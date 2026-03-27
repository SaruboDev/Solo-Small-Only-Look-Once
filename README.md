# Solo (Small Only Look Once)
A Small CNN Model (20M parameters) similar to the **YOLO v1** model structure.

In this repo, i have translated the YOLO v1 structure in a relatively small model structure (**20M paramters**), i have built my labels by using the min-max x/y coordinates from the VOC 2012 dataset and the image size, the pre-processing basically rescales the images, normalizes the coordinate values and the bounding box width and height.

After that the pre-process gets which cells in the final grid the found object is centered in, and checks if the confidence in that one specific cell has already been occupied by another object (if it's 1.0 already, we skip the object).

I have then wrapped the model for simple inference usage with **FastAPI** and a minimal web/HTML interface.

## Key points in this architecture:
- Built with a grid based structure, with 1 bbox for each cell.
- Capable of classifying 20 classes from the VOC 2012 dataset.
- Training VRAM usage <3.3 GB, on a RTX 3060 12GB GPU.

## About me
I am a Junior Developer who's studying to specialize in ML/DL, building skills beyond simply using pre-made architectures or libraries.

I have already built and completed a few projects:
- An **Encoder Transformer** (Here! https://github.com/SaruboDev/BertJax) with **300M parameters**, pre-trained on an RTX 3060 12GB GPU and under 8GB of training VRAM used.
- An **Autodiff engine with Dynamic Computational Graphs** in Python, complete with SGD and Adam optimizers, Softmax and Linear layers! (Here! https://github.com/SaruboDev/Neutron-Python)
