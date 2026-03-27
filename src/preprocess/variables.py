import toml
from typing import Dict

def grab_variables() -> Dict:
    try:
        variables   = toml.load("config.toml")
    except Exception as e:
        raise e
    img_vars    = variables["images"]
    model_vars  = variables["model"]

    global_variables: Dict = {
        "ann_path"      : img_vars.get("ann_path", "VOC2012/Annotations"),
        "img_path"      : img_vars.get("img_path", "VOC2012/JPEGImages"),
        "img_w"         : img_vars.get("image_w", 896),
        "img_h"         : img_vars.get("image_h", 512),
        "bbox"          : model_vars.get("bbox", 1),
        "grid_size_x"   : model_vars.get("grid_size_x", 1),
        "grid_size_y"   : model_vars.get("grid_size_y", 1),
        "epochs"        : model_vars.get("epochs", 1),
        "out_classes"   : model_vars.get("out_classes", 1),
        "lambda_obj"    : model_vars.get("lambda_obj", 1),
        "lambda_noobj"  : model_vars.get("lambda_noobj", 1),
    }

    return global_variables