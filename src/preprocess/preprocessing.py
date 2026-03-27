import grain.python as grain
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import xml.etree.ElementTree as et
import numpy as np
import cv2
from typing import List
import math

@dataclass
class Label:
    filename: str
    labels: jnp.ndarray

class Dataset:
    items: List[Label]
    w: int
    h: int

    def __init__(self, w, h):
        self.items = []
        self.w = w
        self.h = h

    def add(
            self,
            filename: str,
            labels: jnp.ndarray
    ):
        self.items.append(Label(
            filename,
            labels
        ))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        data = self.items[idx]

        def downscale(img):
            read    = cv2.imread(img.filename)
            resized = cv2.resize(read, (self.w, self.h))
            return resized

        img = jnp.array(downscale(data), jnp.float32)
        img = img / 255.0
        img = jnp.transpose(img, axes = (2, 0, 1))

        inputs = {
            "data": jnp.array(img),
            "label": jnp.transpose(data.labels, axes = (2, 0, 1))
        }

        return inputs

def get_loader(
        annotation_path : Path,
        images_path     : Path,
        batch_size      : int,
        epochs          : int,
        max_objects     : int,
        image_w         : int,
        image_h         : int,
        grid_size_x     : int,
        grid_size_y     : int,
        out_classes     : int,
        bbox            : int,
        n_classes_predict: int,
) -> grain.DataLoader | int:
    """
    out_classes = The number of classes plus the amount of bbox * 5 needed.
    n_classes_predict = The number of classes the dataset provides.
    """
    
    df = Dataset(w = image_w, h = image_h)

    map = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19
    }

    # file = "VOC2012/Annotations/2007_000323.xml" # For test purposes

    """
    NOTE
    - Bisogna calcolare x e y come valori 0.0/1.0 dentro la cella specifica
    - Bisogna calcolare w e h come valori non normalizzati, sono la grandezza effettiva della cella.
    - Il risultato deve essere un array di shape (classi + bbox * 5, grid Y, grid X)
        dove classi + bbox * 5 è un'altro array.
    - L'array classi + bbox * 5 è semplicemente un Array di 0.0 di shape (20, )
        tranne 1.0 dove è effettivamente la classe.
    - Array bbox * 5 dove abbiamo (x_center, y_center, w?, h?, confidence) dove:
      - confidence è 1.0, perchè questo array viene applicato SOLO nella cella corretta,
        sennò è tutto 0.0

    Bisogna evitare overwriting.
    Si può iterare sulla dim dei bbox e controllare se c'è un valore confidence == 0.0, la puoi scrivere.
    """

    def grab_data_from_file(file):
        """
        Grabs the needed variables from the .xml file.
        Returns:
        - coords as a list of floats
        - width as a single intager of the width
        - height as a single intager of the height
        - classes as a list of numbers in ohe style for each class found
        """
        tree = et.parse(file).getroot()

        filename    = tree.find(".//filename").text
        class_name  = tree.findall("object/name")
        minX = tree.findall("object/bndbox/xmin")
        maxX = tree.findall("object/bndbox/xmax")
        minY = tree.findall("object/bndbox/ymin")
        maxY = tree.findall("object/bndbox/ymax")

        width   = int(tree.find("size/width").text)
        height  = int(tree.find("size/height").text)

        classes = [i.text for i in class_name]
        class_name = [map.get(i, -1) for i in classes]
        
        minX = [float(i.text) for i in minX]
        maxX = [float(i.text) for i in maxX]
        minY = [float(i.text) for i in minY]
        maxY = [float(i.text) for i in maxY]

        return filename, class_name, minX, maxX, minY, maxY, width, height

    def norm_coords(minX, maxX, minY, maxY, width, height, image_w, image_h):
        """
        Scaling the bbox values from the original to the rescaled size.
        """
        norm_minX = minX * (image_w / width)
        norm_maxX = maxX * (image_w / width)
        norm_minY = minY * (image_h / height)
        norm_maxY = maxY * (image_h / height)
        return norm_minX, norm_maxX, norm_minY, norm_maxY
    
    def pad_data(classes, minX, maxX, minY, maxY, max_objects):
        pad = max_objects - len(classes)
        if pad > 0:
            classes += [-1] * pad
            minX    += [-1] * pad
            maxX    += [-1] * pad
            minY    += [-1] * pad
            maxY    += [-1] * pad
        else:
            classes = classes[:max_objects]
            minX = minX[:max_objects]
            maxX = maxX[:max_objects]
            minY = minY[:max_objects]
            maxY = maxY[:max_objects]
        return classes, minX, maxX, minY, maxY

    def create_label(minX, maxX, minY, maxY, class_number, final_array):
        if minX == -1 or class_number == -1:
            return None, (None, None)

        # We first get which cell it is.
        center_x = (minX + maxX) / 2 # Center X of the BBOX in the IMAGE
        center_y = (minY + maxY) / 2 # Center Y of the BBOX in the IMAGE


        cell_width = image_w / grid_size_x
        cell_height = image_h / grid_size_y

        col_cell = math.floor(center_x / cell_width) # Cell x idx
        row_cell = math.floor(center_y / cell_height) # Cell y idx

        # W and H are the bounding box width and height
        w = maxX - minX
        h = maxY - minY

        normalized_x = center_x / image_w
        normalized_y = center_y / image_h
        normalized_w = w / image_w
        normalized_h = h / image_h

        
        if final_array[row_cell, col_cell, class_number] == 0.0:
            ohe = jnp.zeros(shape = (n_classes_predict, )).at[class_number].set(1.0)
            locality = jnp.array([normalized_x, normalized_y, normalized_w, normalized_h, 1.0])

            cells = jnp.concatenate([
                jnp.tile(locality, bbox),
                ohe
            ], axis = 0)
            return cells, (row_cell, col_cell)
        else:
            return None, (None, None)
    
    for file in tqdm(sorted(annotation_path.iterdir())):
        # NOTE: Gotta check if i'm calculating with the right sizes (pre-rescaling, then scaling).
        
        #####################
        ### Data Retrieve ###
        #####################

        filename, class_name, minX, maxX, minY, maxY, width, height = grab_data_from_file(file)

        #################
        ### Data Norm ###
        #################

        new_minX = []
        new_maxX = []
        new_minY = []
        new_maxY = []
        if type(minX) != list:
            print("Mhh... this one somehow is not a list.")
        for idx, _ in enumerate(minX):
            norm_minX, norm_maxX, norm_minY, norm_maxY = norm_coords(
                minX[idx],
                maxX[idx],
                minY[idx],
                maxY[idx],
                width = width,
                height = height,
                image_w = image_w,
                image_h = image_h
            )
            new_minX.append(norm_minX)
            new_maxX.append(norm_maxX)
            new_minY.append(norm_minY)
            new_maxY.append(norm_maxY)

        ####################
        ### Data Padding ###
        ####################

        classes, minX, maxX, minY, maxY = pad_data(
            class_name,
            new_minX,
            new_maxX,
            new_minY,
            new_maxY,
            max_objects
        )

        ##########################
        ### Array Label Making ###
        ##########################

        final_array = jnp.zeros(shape = (grid_size_y, grid_size_x, out_classes))

        for idx in range(len(minX)):
            cell, pos = create_label(
                minX[idx],
                maxX[idx],
                minY[idx],
                maxY[idx],
                classes[idx],
                final_array
            )
            if cell is not None:
                final_array  = final_array.at[pos].set(cell)
        
        filepath = images_path / filename
        df.add(filepath, final_array)

    ################
    ### Old Code ###
    ################

    # for file in tqdm(sorted(annotation_path.iterdir())):
    #     tree = et.parse(file).getroot()

    #     filename    = tree.find(".//filename").text
    #     class_name  = tree.findall("object/name")
    #     minX = tree.findall("object/bndbox/xmin")
    #     maxX = tree.findall("object/bndbox/xmax")
    #     minY = tree.findall("object/bndbox/ymin")
    #     maxY = tree.findall("object/bndbox/ymax")

    #     width   = int(tree.find("size/width").text)
    #     height  = int(tree.find("size/height").text)

    #     classes = [i.text for i in class_name]
    #     class_name = [map.get(i, -1) for i in classes]

    #     def norm_coords(minX, maxX, minY, maxY):
    #         norm_minX = minX * (image_w / width)
    #         norm_maxX = maxX * (image_w / width)
    #         norm_minY = minY * (image_h / height)
    #         norm_maxY = maxY * (image_h / height)
    #         return norm_minX, norm_maxX, norm_minY, norm_maxY
        
    #     new_minX = []
    #     new_maxX = []
    #     new_minY = []
    #     new_maxY = []
    #     if type(minX) == list:
    #         for idx, _ in enumerate(minX):
    #             c_minX = float(minX[idx].text)
    #             c_maxX = float(maxX[idx].text)
    #             c_minY = float(minY[idx].text)
    #             c_maxY = float(maxY[idx].text)
    #             norm_minX, norm_maxX, norm_minY, norm_maxY = norm_coords(c_minX, c_maxX, c_minY, c_maxY)
    #             new_minX.append(norm_minX)
    #             new_maxX.append(norm_maxX)
    #             new_minY.append(norm_minY)
    #             new_maxY.append(norm_maxY)
    #     else:
    #         norm_minX, norm_maxX, norm_minY, norm_maxY = norm_coords(minX, maxX, minY, maxY)
    #         new_minX.append(norm_minX)
    #         new_maxX.append(norm_maxX)
    #         new_minY.append(norm_minY)
    #         new_maxY.append(norm_maxY)

    #     def pad_data(classes, minX, maxX, minY, maxY):
    #         pad = max_objects - len(classes)
    #         if pad > 0:
    #             classes += [-1] * pad
    #             minX    += [-1] * pad
    #             maxX    += [-1] * pad
    #             minY    += [-1] * pad
    #             maxY    += [-1] * pad
    #         else:
    #             classes = classes[:max_objects]
    #             minX = minX[:max_objects]
    #             maxX = maxX[:max_objects]
    #             minY = minY[:max_objects]
    #             maxY = maxY[:max_objects]
    #         return classes, minX, maxX, minY, maxY
        
    #     classes, minX, maxX, minY, maxY = pad_data(class_name, new_minX, new_maxX, new_minY, new_maxY)

    #     def get_cell_labels(minX, maxX, minY, maxY, class_number):
    #         """
    #         For every idx found, it's an entirely new object. So we build them separately.
    #         Considering we're inside a main loop where each file is searched, we're actually doing
    #         one operation for every one found. coords are padded to the max_objects, which is the
    #         bbox value B we need.
    #         Meaning that in this function, we're making the ohe for each object one each call.

    #         """
    #         if minX == -1 or class_number == -1:
    #             return None, (None, None)
    #         center_x = ((minX + maxX) / 2) / image_w
    #         center_y = ((minY + maxY) / 2) / image_h
    #         center_cell_x = center_x * grid_size_x
    #         center_cell_y = center_y * grid_size_y

    #         center_img_x = center_cell_x - int(center_cell_x)
    #         center_img_y = center_cell_y - int(center_cell_y)

    #         w = (maxX - minX) / image_w
    #         h = (maxY - minY) / image_h

    #         ohe = jnp.zeros(shape = (n_classes_predict,)).at[class_number].set(1.0)
    #         locality = jnp.array([center_img_x, center_img_y, w, h, 1.0])

    #         cells = jnp.concatenate([
    #             jnp.tile(locality, bbox),
    #             ohe
    #         ], axis = 0)
    #         return cells, (int(center_cell_y), int(center_cell_x))

    #     object_label = jnp.zeros(shape = (grid_size_y, grid_size_x, out_classes))

    #     # if type(minX) == list:
    #     for idx, _ in enumerate(minX):
    #         label, pos = get_cell_labels(
    #             minX[idx],
    #             maxX[idx],
    #             minY[idx],
    #             maxY[idx],
    #             classes[idx]
    #         )
    #         if label is not None:
    #             object_label = object_label.at[pos].set(label)
    #         else:
    #             label, pos = get_cell_labels(new_minX, new_maxX, new_minY, new_maxY, classes[0])
    #             if label is not None:
    #                 object_label = object_label.at[pos].set(label)

    
    #     filepath    = images_path / filename

    #     df.add(filepath, object_label)

    sampler = grain.IndexSampler(
        num_records     = len(df),
        shard_options   = grain.NoSharding(),
        shuffle         = True,
        num_epochs      = epochs,
        seed            = 42
    )

    dataloader = grain.DataLoader(
        data_source = df,
        sampler     = sampler,
        operations  = [
            grain.Batch(
                batch_size = batch_size,
                drop_remainder = True
            )
        ],
        worker_count = 0
    )
    avaiable_steps = len(df) // batch_size
    print(f"Estimated batches: {avaiable_steps}")

    return dataloader, avaiable_steps