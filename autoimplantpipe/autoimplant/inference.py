from typing import List, Dict
import numpy as np
import torch
import nibabel as nib
from skimage.morphology import binary_erosion, binary_dilation
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Resized,
    Invertd,
)
from autoimplantpipe.autoimplant.models import AutoImplantUNet


def merge_batch(batch_pred: list, post_pred: Compose):
    """Merge batch prediction"""
    post_pred_batch = [post_pred(i) for i in decollate_batch(batch_pred)]
    pred_img = post_pred_batch[0]["pred"][0, :, :, :].detach().numpy()
    return pred_img


def load_autoimplant_model(model_path: str, device=torch.device("cpu")):
    """
    Load a autoimplant model from a trained pth file.
    """
    autoimpant_model = AutoImplantUNet().to(device)
    autoimpant_model.load_state_dict(torch.load(model_path))
    return autoimpant_model


def predict_autoimplant(predict_dict: dict, model):
    """
    Perform autoimplant inference with a given test file dictionary and pretrained model

    test_file: dict, {"image": "path to nifti file", "label": "path to label file"}
    model: UNet, loaded pretrain model

    Example
    =======
    >>> predict_dict = {"image": "...", "label": ...}
    >>> autoimplant_model = load_autoimplant_model(model_path)
    >>> image_output = predict_autoimplant(predict_dict, autoimplant_model) # save with original headers
    """
    model.eval()
    pred_keys = list(predict_dict.keys())
    data_transforms = Compose(
        [
            LoadImaged(keys=pred_keys),
            EnsureChannelFirstd(keys=pred_keys),
            CropForegroundd(keys=pred_keys, source_key="image"),
            Orientationd(keys=pred_keys, axcodes="RAS"),
            Resized(
                keys=pred_keys, spatial_size=(176, 224, 144), mode=["area", "nearest"]
            ),
        ]
    )

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=data_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )

    test_files = [predict_dict]
    data_ds = Dataset(data=test_files, transform=data_transforms)
    data_loader = DataLoader(data_ds, batch_size=1, num_workers=2)

    # perform autoimplant inference over given dataloader

    for batch_data in data_loader:
        inputs, labels = (
            batch_data["image"],
            batch_data["label"],
        )
        batch_data["pred"] = model(inputs)

    image_output = merge_batch(batch_data, post_transforms)
    return image_output


def save_prediction_array_to_nifti(
    prediction_data: np.array, orig_defective_nii_path: str, output_path: str
):
    """
    Save autoimplant prediction array with original NifTI file
    to a given output path
    """
    # read original data, compute implant, add implant back to original data
    orig_defect_img = nib.load(orig_defective_nii_path)
    orig_defect_data = orig_defect_img.get_fdata()

    implant_data = prediction_data - orig_defect_data
    implant_data[implant_data < 0] = 0
    implant_data = binary_erosion(implant_data).astype(int)
    implant_data = binary_erosion(implant_data).astype(int)
    implant_data = binary_dilation(implant_data).astype(int)
    implant_data = binary_dilation(implant_data).astype(int)
    defect_skull_data = orig_defect_data.copy()
    defect_skull_data[implant_data > 0] = 2  # assign to implant
    defect_skull_data = defect_skull_data.clip(0, 2)

    ni_img = nib.Nifti1Image(
        defect_skull_data.astype(np.uint8),
        affine=orig_defect_img.affine,
        header=orig_defect_img.header,
    )  # add affine and header
    ni_img.header.set_data_dtype(np.uint8)  # set datatype to uint8
    nib.save(ni_img, output_path)  # save file
