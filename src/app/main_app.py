from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import shutil

from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict
import optax
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import cv2

from ..model import Solo
from ..preprocess import grab_variables
from ..model.extra import load_model, summary

UPLOAD_DIR = "src/app/static/uploads"

def return_optimizer():
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value = 0.0, peak_value = 0.0001, end_value = 0.00001, warmup_steps = 300, decay_steps = 1000
    )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate = lr_schedule))    
    return optimizer

def setup_model(variables, optimizer):
    model = Solo(
        input_size = 3,
        out_classes = variables["out_classes"],
        key = jr.PRNGKey(42),
        dtype = jnp.bfloat16
    )

    loaded_model, _, _, _, _ = load_model(
        path = "src/model/model.eqx",
        model_base = model,
        optimizer = optimizer,
        is_inexact_init = True
    )

    inference_model = eqx.nn.inference_mode(loaded_model, True)
    return inference_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App starting up.")

    variables: Dict = grab_variables()

    print("Creating dummy optimizer.")
    optimizer = return_optimizer()

    print("Creating Dummy model.")

    model = setup_model(variables, optimizer)

    app.state.model = model
    app.state.variables = variables
    app.state.bbox = app.state.variables["bbox"]

    yield

    del app.state.model
    del app.state.variables
    del app.state.bbox
    del app.state.map
    shutil.rmtree(UPLOAD_DIR, ignore_errors = True)

    print("Cleaned App variables. Now Closing.")

app = FastAPI(lifespan = lifespan)

app.mount("/static", StaticFiles(directory = "src/app/static"), name = "static")

template = Jinja2Templates(directory = "src/app/templates")

os.makedirs(UPLOAD_DIR, exist_ok = True)


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
    pred_c = preds_coords_res[..., 4:5, :, :]

    pred_class = jax.nn.softmax(preds_class, axis = 0)
    pred_classes = pred_class.argmax(axis = 0)
    return pred_xy, pred_wh, pred_c, pred_classes

# @eqx.filter_jit
def run_model(model, img, bbox):
    pred = model(img.astype(jnp.bfloat16))

    t_preds = transform_preds(pred, bbox)

    return t_preds

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return template.TemplateResponse("image_form.html", {"request": request, "file_info": None})

@app.post("/", response_class = HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
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
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    import cv2
    img_read = cv2.imread(f"src/app/static/uploads/{file.filename}")
    resized = cv2.resize(img_read, (app.state.variables["img_w"], app.state.variables["img_h"]))
    img = resized / 255.0
    img = jnp.transpose(img, axes = (2, 0, 1))
    preds = run_model(app.state.model, img, app.state.bbox)

    xy, wh, c, classes = preds
    x = xy[:, :1, :, :]
    y = xy[:, 1:, :, :]
    w = wh[:, :1, :, :]
    h = wh[:, 1:, :, :]

    y_idx, x_idx = jnp.meshgrid(
        jnp.arange(app.state.variables["grid_size_y"]), jnp.arange(app.state.variables["grid_size_x"]),
        indexing = "ij"
    )

    c_no_bbox = c[0, 0]
    c_mask = c_no_bbox >= 0.9
    center_class = classes[c_mask]
    idx_x = x_idx[c_mask]
    idx_y = y_idx[c_mask]

    cell_x = app.state.variables["img_w"] / app.state.variables["grid_size_x"]
    cell_y = app.state.variables["img_h"] / app.state.variables["grid_size_y"] # Size of cell

    x_true = x[0,0][c_mask] # Offset X
    y_true = y[0,0][c_mask] # Offset Y

    real_x = (idx_x + x_true) * cell_x
    real_y = (idx_y + y_true) * cell_y

    real_w = w[0, 0][c_mask]
    real_h = h[0, 0][c_mask]

    x_min = real_x - (real_w / 2)
    x_max = real_x + (real_w / 2)
    y_min = real_y - (real_h / 2)
    y_max = real_y + (real_h / 2)

    for idx in range(len(center_class)):
        pt1 = (int(x_min[idx]), int(y_min[idx])) # (x, y)
        pt2 = (int(x_max[idx]), int(y_max[idx])) # (x, y)
        cv2.rectangle(resized, pt1, pt2, color = (0, 255, 0), thickness = 5)
        label_map = map_dict[int(center_class[idx])]
        cv2.putText(resized, label_map, pt1, color = (0, 255, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0)

    cv2.imwrite(f"src/app/static/uploads/mod_{file.filename}", resized)

    if os.path.exists(f"/static/uploads/{file.filename}"):
        os.remove(f"/static/uploads/{file.filename}")

    file_info = {
        "filename": file.filename,
        "content_type": file.content_type,
        "image_url": f"/static/uploads/mod_{file.filename}"
    }

    return template.TemplateResponse("image_form.html", {"request": request, "file_info": file_info, "prediction": xy.shape})