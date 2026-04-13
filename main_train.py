import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path
from tqdm import tqdm
import cv2
import json
import os

from src.preprocess.preprocessing import get_loader
from src.model.layers import Solo
from src.model.extra import save_model, load_model, summary

ann_path = Path("images_and_labels/labels")
img_path = Path("images_and_labels/images")

image_w     = 896
image_h     = 512
batch_size  = 16
bbox        = 1 # Technically, the B value from the YOLO paper.
grid_size_x = image_w // (2**6) # Size of the last convolutional layer grid
grid_size_y = image_h // (2**6)
epochs      = 100
global_dtype = jnp.bfloat16

# How many values the model predicts, 20 labels + (bbox * 5) where 4 are coords and 1 is the confidence.
out_classes = 20 + (bbox * 5)

lambda_obj = 5.0
lambda_noobj = 0.5
classes_lambda = 2.0

dataloader, steps = get_loader(
    annotation_path = ann_path,
    images_path     = img_path,
    batch_size      = batch_size,
    epochs          = epochs,
    max_objects     = 3,
    image_w         = image_w,
    image_h         = image_h,
    grid_size_x     = grid_size_x,
    grid_size_y     = grid_size_y,
    out_classes     = out_classes,
    bbox            = bbox,
    n_classes_predict = 20
)

model = Solo(
    input_size  = 3,
    out_classes = out_classes,
    key         = jr.PRNGKey(42),
    dtype       = jnp.bfloat16
)

summary(model)


def loss_fn(logits, labels):
    """
    In the paper they say:
    sum of squared error loss.
    "If the ground truth for some coordinate prediction is ˆt* our gradient is the ground truth value
    (computed from the groundtruth box) minus our prediction: ˆt* − t*.
    This ground truth value can be easily computed by inverting the equations above."

    "During training we use binary cross-entropy loss for the class predictions."
    """
    logits = logits.astype(jnp.float32)
    labels = labels.astype(jnp.float32)
    coords_n = bbox * 5
    preds_coords     = logits[:, :coords_n, :, :]
    preds_class      = logits[:, coords_n:, :, :]
    labels_coords    = labels[:, :coords_n, :, :]
    labels_class     = labels[:, coords_n:, :, :]

    # NOTE: The IOU filter for the bbox is missing. I don't really need it for this demo
    labels_coords_res = jnp.reshape(
        labels_coords,
        shape = (
            labels_coords.shape[0], bbox, 5, labels_coords.shape[-2], labels_coords.shape[-1]
        )
    ) # shape (2, 3, 5, 7, 7)
    preds_coords_res = jnp.reshape(
        preds_coords,
        shape = (preds_coords.shape[0], bbox, 5, preds_coords.shape[-2], preds_coords.shape[-1])
    )

    # Now I only grab the confidence levels.
    mask_obj = labels_coords_res[..., -1, :, :].astype(bool) # The last index of the third dim.
    mask_obj = mask_obj[..., None, :, :]
    mask_noobj = ~mask_obj

    num_objects = jnp.maximum(mask_obj.sum(), 1.0)
    num_noobjects = jnp.maximum(mask_noobj.sum(), 1.0)


    pred_xy = jax.nn.sigmoid(preds_coords_res[..., 0:2, :, :])
    label_xy = labels_coords_res[..., 0:2, :, :]
    xy_loss = ((optax.l2_loss(pred_xy, label_xy) * mask_obj).sum() / num_objects) * lambda_obj

    pred_wh = jax.nn.softplus(preds_coords_res[..., 2:4 , :, :])
    label_wh = labels_coords_res[..., 2:4, :, :]
    wh_loss = (
        ((optax.l2_loss(
            pred_wh, label_wh
        ) * mask_obj).sum()
        ) / num_objects) * lambda_obj

    pred_c = preds_coords_res[..., 4:5, :, :]
    label_c = labels_coords_res[..., 4:5, :, :]
    c_loss_1 = optax.sigmoid_focal_loss(pred_c, label_c)
    c_loss_2 = ((c_loss_1 * mask_noobj).sum() / num_noobjects) * lambda_noobj

    c_loss_1 = ((c_loss_1 * mask_obj).sum() / num_objects) * lambda_obj

    classes_loss = optax.sigmoid_binary_cross_entropy(preds_class, labels_class)
    classes_loss = classes_loss.mean(axis = 1, keepdims = True)
    classes_loss = classes_loss * mask_obj.any(axis = 1)[:, None, :, :]
    classes_loss = (classes_loss.sum() / num_objects) * classes_lambda

    loss = xy_loss + wh_loss + c_loss_1 + c_loss_2 + classes_loss

    return loss, (xy_loss, wh_loss, c_loss_1, c_loss_2, classes_loss)


def vram():
    device = jax.devices()[0]
    stats = device.memory_stats()

    used = stats["bytes_in_use"] / (1024 ** 3)
    peak = stats["peak_bytes_in_use"] / (1024 ** 3)

    return used, peak

def compute_loss(model, data, key):
    inputs = data["data"].astype(global_dtype)
    labels = data["label"]
    # inputs shape is (batch_size,c, h, w)
    logits = jax.vmap(model)(inputs)

    loss = loss_fn(logits, labels)

    return loss

@eqx.filter_jit
def make_steps(model, data, opt_state, optimizer, train_key):
    k1, k2 = jr.split(train_key, 2)
    (loss, losses), grads = eqx.filter_value_and_grad(compute_loss, has_aux = True)(model, data, k1)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)

    return loss, losses, model, opt_state, k2

# warmup = two epochs, decay steps = all epochs
lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
    init_value = 0.0, peak_value = 0.001, end_value = 0.00001, warmup_steps = steps * 2, decay_steps = steps * (epochs // 1.5)
)
# Used to test the architecture. NOTE: Write a test environment to avoid commenting-uncommenting each time.
# lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
#     init_value = 0.0, peak_value = 0.001, end_value = 0.00001, warmup_steps = 2, decay_steps = 10
# )
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate = lr_schedule))

def train(model, opt_state = None, train_key = None, starting_step = 0, log = None, starting_epoch = 1):
    iterable = iter(dataloader)
    if train_key is None:
        train_key = jr.PRNGKey(42)

    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    if log == None:
        log = {
            "train" : {
                "loss": [],
                "XY": [],
                "WH": [],
                "Obj": [],
                "Noobj": [],
                "Classes": [],
                "vram": []
            }
        }

    global_step = 0
    for epoch in range(starting_epoch, epochs +1):
        print(f"\nEpoch {epoch} / {epochs}")
        tbar = tqdm(total = steps, desc = "...", nrows = 2)
        
        loss_epoch = 0.0
        for step in range(starting_step, steps):
            global_step += 1
            data = next(iterable)
            loss, losses, model, opt_state, train_key = make_steps(
                model, data, opt_state, optimizer, train_key
            )
            loss_epoch += loss
            loss = loss_epoch / (step + 1)

            xy_loss, wh_loss, c_loss_1, c_loss_2, classes_loss = losses
            tbar.update(1)
            used, peak = vram()
            current_lr = lr_schedule(global_step)
            tbar.set_description(
                desc = f"Loss: {loss:.3f} | XY: {xy_loss:.3f} | WH: {wh_loss:.3f} |"
                f" Obj Conf: {c_loss_1:.3f} | No Obj Conf: {c_loss_2:.3f} |"
                f" Classes: {classes_loss:.3f} | LR: {current_lr} |"
                f"Vram used: {used:.3f} | Vram peak: {peak:.3f}"
            )
            log["train"]["loss"].append(float(loss))
            log["train"]["XY"].append(float(xy_loss))
            log["train"]["WH"].append(float(wh_loss))
            log["train"]["Obj"].append(float(c_loss_1))
            log["train"]["Noobj"].append(float(c_loss_2))
            log["train"]["Classes"].append(float(classes_loss))
            log["train"]["vram"].append(float(used))

            if step == steps - 1:
                print("Epoch finished, saving...")
                tbar.close()

                tmp = "log.json.tmp"
                with open(tmp, "w") as file:
                    json.dump(log, file)
                os.replace(tmp, "log.json")
                save_model(
                    f"saves/model_{epoch}.eqx",
                    model,
                    opt_state = opt_state,
                    step = step,
                    epoch = epoch,
                    train_key = train_key
                )

train(model)

read    = cv2.imread("train.png")
resized = cv2.resize(read, (image_w, image_h))

img = resized / 255.0
img = jnp.transpose(img, axes = (2, 0, 1))

# model, opt_state, step, epoch, train_key = load_model("saves/model_136.eqx", model, optimizer)

with open("log.json", "r") as file:
    log = json.load(file)

# train(model, opt_state, train_key, 0, log, epoch)

def transform_preds(logits, bbox):
    # Logits shape (25, 8, 14)
    logits = logits.astype(jnp.float32)
    coords_n = bbox * 5
    preds_coords     = logits[:coords_n, :, :]
    preds_class      = logits[coords_n:, :, :]

    preds_coords_res = jnp.reshape(
        preds_coords,
        shape = (bbox, 5, preds_coords.shape[-2], preds_coords.shape[-1])
    )

    pred_xy = jax.nn.sigmoid(preds_coords_res[:, 0:2, :, :])
    pred_wh = jax.nn.softplus(preds_coords_res[:, 2:4 , :, :])
    pred_c = jax.nn.sigmoid(preds_coords_res[..., 4:5, :, :])

    pred_class = jax.nn.softmax(preds_class, axis = 0)
    pred_classes = pred_class.argmax(axis = 0)
    return pred_xy, pred_wh, pred_c, pred_classes

def run_model(model, img, bbox):
    pred = model(img.astype(jnp.bfloat16))

    t_preds = transform_preds(pred, bbox)

    return t_preds

preds = run_model(model, img, bbox)

xy, wh, c, classes = preds
x = xy[:, :1, :, :]
y = xy[:, 1:, :, :]
w = wh[:, :1, :, :]
h = wh[:, 1:, :, :]

c_no_bbox = jax.nn.sigmoid(c[0, 0])
c_mask = c_no_bbox >= 0.61
center_class = classes[c_mask]
map_dict = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor"
}

x_norm = x[0, 0][c_mask]
y_norm = y[0, 0][c_mask]
w_norm = w[0, 0][c_mask]
h_norm = h[0, 0][c_mask] 

real_x = x_norm * image_w
real_y = y_norm * image_h
real_w = w_norm * image_w
real_h = h_norm * image_h

x_min = real_x - (real_w / 2)
y_min = real_y - (real_h / 2)
x_max = real_x + (real_w / 2)
y_max = real_y + (real_h / 2)

print(f"Confidence Max: {jnp.max(c_no_bbox)}")
print(f"Confidence Min: {jnp.min(c_no_bbox)}")
print(f"Confidence Mean: {jnp.mean(c_no_bbox)}")

for i in range(len(center_class)):
    pt1 = (int(x_min[i]), int(y_min[i]))
    pt2 = (int(x_max[i]), int(y_max[i]))
    
    cv2.rectangle(resized, pt1, pt2, (0, 255, 0), 2)
    
    label_name = map_dict.get(int(center_class[i]), "unknown")
    cv2.putText(resized, label_name, (pt1[0], pt1[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Risultato", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
