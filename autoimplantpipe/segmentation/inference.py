import os.path as op
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from skimage.morphology import remove_small_objects
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from autoimplantpipe.utils import save_nii_file


def load_segmentation_model(model_path: str, device=torch.device("cpu")):
    """
    Load a segmentation model from a trained pth file.
    """
    segmentation_model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    segmentation_model.load_state_dict(torch.load(model_path))
    return segmentation_model


def generate_transforms():
    """
    Generate transforms for inference
    """
    test_org_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.625), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-43, a_max=453, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        ]
    )  # predict the output to output_dir with additional _seg postfix
    return test_org_transforms, post_transforms


def remove_small_islands(img: np.array, percentage_remove: float = 10):
    """
    Remove small islands from an image.
    percentage_remove: float, is a percentage of island we want to remove. Default 10.
    """
    unique, counts = np.unique(img, return_counts=True)
    unique_dict = dict(zip(unique, counts))
    min_size = unique_dict[1] * percentage_remove / 100
    small_removed_img = remove_small_objects(
        img.astype(bool), min_size=min_size
    ).astype(int)
    return small_removed_img


def predict_skull_segmentation(
    prediction_path: str,
    segmentation_model,
    output_dir: str = "out",
    output_postfix: str = "seg",
    device=torch.device("cpu"),
    percentage_islands_remove: float = 10,
):
    """
    Predict skull segmentation from a given Nifti file.

    >>> prediction_path = "example/CT1.nii" # path to predict
    >>> segmentation_model = load_segmentation_model("model/unet_segmentation_300ep_lr1e-4.pth")  # load model
    >>> segmentation_path = predict_skull_segmentation(
            predict_path=prediction_path,
            segmentation_model=segmentation_model,
            output_dir="out"
        )
    """
    test_org_transforms, post_transforms = generate_transforms()

    # create dataset and dataloadere from a given file
    test_data = [{"image": prediction_path}]
    test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=1)

    # perform segmentation inference
    segmentation_model.eval()
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, segmentation_model
            )
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
    output_img = test_data[0]["pred"][1, :, :, :].clip(0, 1).astype(np.uint8)
    if percentage_islands_remove > 0:
        output_img = remove_small_islands(output_img, percentage_islands_remove)
    raw_img = nib.load(prediction_path)
    output_path = op.join(
        output_dir, Path(prediction_path).stem + f"_{output_postfix}.nii"
    )
    save_nii_file(raw_img, output_img, output_path)
    return output_path
