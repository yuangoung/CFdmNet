Datasets：0.8 DEM and Multispectral Images for Retrogressive Thaw Slump Susceptibility Mapping[DS/OL]. V1. Science Data Bank, 2026. https://doi.org/10.57760/sciencedb.37659.

<img width="6516" height="3991" alt="Figure_3" src="https://github.com/user-attachments/assets/caa149c2-b256-41eb-98c1-f50d316eee2e" />

This repository provides the implementation of CFdmNet for retrogressive thaw slump susceptibility mapping. The code was developed in Python with PyTorch. The required packages include PyTorch, NumPy, OpenCV, rasterio, GDAL, and tqdm. GPU is recommended for model training and inference on large remote sensing images. The dataset should be arranged according to the folder structure described in this repository, including multispectral images, DEM data, and binary RTS labels. Training is performed by running the training script with the corresponding configuration file and data directory. After training, the inference script can be used to load the trained weights and produce RTS susceptibility maps. For large GF-7 scenes, the inference procedure adopts a sliding-window strategy and merges overlapping predictions in the logit domain before converting them into probability values.

<img width="1824" height="1356" alt="Figure_5" src="https://github.com/user-attachments/assets/6c7a7950-a0b9-4f9e-867c-03eea49a10c3" />
