import nibabel as nib
import numpy as np
import nrrd
import SimpleITK as sitk
from pathlib import Path


def _load_skull(filename: str):
    """
    Load skull image with raw or NIfTI format.
    """
    ext = Path(filename).suffix
    if ext == ".nrrd":
        return nib.Nifti1Image(nrrd.read(filename)[0], np.eye(4))
    elif ext in [".nii", ".nii.gz"]:
        return nib.load(filename)
    else:
        return False
    # TODO: implement ext == ".stl"
    return False


def _evaluate(y_true: np.array, y_pred: np.array):
    """
    Calculate 3D metric of segmentation between `y_true` and `y_pred`.
    The metrics include Average Hausdorff Distance, Hausdorff Distance,
    Dice and Jaccard Coefficient, Volume Similarity, and classification errors.
    """
    quality = dict()
    labelPred = sitk.GetImageFromArray(y_pred, isVector=False) > 0.5
    labelTrue = sitk.GetImageFromArray(y_true, isVector=False) > 0.5

    # Hausdorff Distance
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue, labelPred)
    quality["avg_hausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    # Dice,Jaccard,Volume Similarity
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue, labelPred)
    quality["dice"] = dicecomputer.GetDiceCoefficient()
    quality["jaccard"] = dicecomputer.GetJaccardCoefficient()
    quality["volume_similarity"] = dicecomputer.GetVolumeSimilarity()

    # Classification errors
    quality["false_negative"] = dicecomputer.GetFalseNegativeError()
    quality["false_positive"] = dicecomputer.GetFalsePositiveError()

    return quality


def evaluate_skulls(pred_file: str, target_file: str, verbose: bool = False):
    """
    Calculate 3D metric of segmentation between `target_file` and `pred_file`.
    The metrics include Average Hausdorff Distance, Hausdorff Distance,
    Dice and Jaccard Coefficient, Volume Similarity, and classification errors.

    Support '.nrrd', '.nii', or '.nii.gz' file format.
    >>> pred_file = "predicted_skull/000.nrrd"
    >>> target_file = "complete_skull/000.nrrd"
    """
    pred = _load_skull(pred_file)
    target = _load_skull(target_file)

    labelPred = sitk.GetImageFromArray(pred.get_fdata(), isVector=False) > 0.5
    labelTrue = sitk.GetImageFromArray(target.get_fdata(), isVector=False) > 0.5

    result = dict()

    # Hausdorff Distance
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue, labelPred)
    result["avg_hausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    result["hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    # Dice,Jaccard,Volume Similarity
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue, labelPred)
    result["dice"] = dicecomputer.GetDiceCoefficient()
    result["jaccard"] = dicecomputer.GetJaccardCoefficient()
    result["volume_similarity"] = dicecomputer.GetVolumeSimilarity()

    # Classification errors
    result["false_negative"] = dicecomputer.GetFalseNegativeError()
    result["false_positive"] = dicecomputer.GetFalsePositiveError()

    if verbose:
        print("Predicted Skull:", Path(pred_file).name)
        print("Complete Skull:", Path(target_file).name)
        print("")
        print(f"Average Hausdorff Distance: { result['avg_hausdorff']:.2f }")
        print(f"Hausdorff Distance:         { result['hausdorff']:.2f }")
        print(f"DSC:                        { result['dice'] * 100:.4f }%")
        print(f"IoU:                        { result['jaccard'] * 100:.4f }%")
        print(f"Volume Similarity:          { result['volume_similarity']:.4f }")
        print(f"False Negative:             { result['false_negative'] * 100:.2f }%")
        print(f"False Positive:             { result['false_positive'] * 100:.2f }%")

    return result
