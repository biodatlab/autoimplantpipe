# AutoimplantPipe

This is a repository for the paper _"An End-to-End Pipeline for Automatic Patient-Specific Cranial Implant Design:
From CT Scans to Titanium Implants"_. The repository contains the following implementations:

1. **Automatic skull segmentation** predict 3D skulls from a given CT Scans in grayscale
2. **Autoimplant** predicting complete skull from a defective skull

## Notebooks

Notebooks folder contains notebook for automatic skull segmentation and autoimplant inferences including

- `01_autosegmentation.ipynb` is an example notebook for performing segmentation from a given Nifti file
- `02_autoimplant_prediction.ipynb` is an example for performing autoimplant inference

We provide the pretrained model for both segmentation and autoimplant
[here](#Results-and-Models) and the SkullBreak data example for autoimplant inference [here](./skullbreak_parietotemporal_001.nii.gz).

## Training scripts

`scripts` folder includes the script for training models and baseline models.

- `train_skull_segmentation_pl.ipynb` is a notebook for training a segmentation model with Pytorch Lightning
- `train_autoimplant.py` is a script for training an autoimplant model

## Results and Models

Dice score and model checkpoints for segmentation and autoimplant models.

| Model |  Dice Score | Checkpoint |
| ----- | :-----------: | ---- |
| Unet + Post Process (Segmentation) | 0.9100 | [link](https://drive.google.com/file/d/1__LxfFFNa7lquG8mT2unGBgNqRvVVFlj/view?usp=share_link) |
| PCA   | 0.7773 | - |
| 3DUNetCNN off-the-shelf | 0.9205 | - |
| 3DUNetCNN SkullBreak | 0.9464 | [link](https://drive.google.com/file/d/1Zvj3xa1E2pHV-Ykvqa70S5IOhiMWVL39/view?usp=share_link) |
| 3DUNetCNN in-house | 0.9711 | - |
| 3DUNetCNN SkullBreak + in-house | 0.9715 | [link](https://drive.google.com/file/d/1XrgC84nhVJVHKtgC5jGLhXckup2A5BMK/view?usp=share_link) |

## Installation

Download the repository using `git`:

```sh
git clone https://github.com/biodatlab/autoimplantpipe
cd autoimplantpipe
```

Install dependencies and library using `pip`:

```sh
pip install -r requirements.txt  # install dependencies
pip install .  # install `autoimplant_pipeline` library
```
