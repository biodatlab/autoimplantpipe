from typing import List
import os
import os.path as op
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob
import vtk
import nibabel as nib


def show_slices(slices: list):
    """
    Display image slices.
    """
    fig, axes = plt.subplots(1, len(slices), figsize=(15, 20))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def create_dict_datasets(images: List[str], labels: List[str]):
    """
    Create a list of data dictionary with keys 'image' and 'label'
    from a list of images and labels.
    """
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]
    return data_dicts


def convert_nifti_to_stl(
    nii_path: str, stl_path: str, label: int = 1, decimate_factor: float = 0
):
    """
    Read a nifti file including a binary map of a segmented organ with label id = label.
    Convert it to a smoothed mesh of type stl.
    nii_path: str, Input nifti binary map
    stl_path: str, Output mesh name in stl format
    label: int, segmented label id
        default as 1
    decimate_factor: float, [0,1) reduce traingles factor (0.5 = reduce 50% of triangles)
        default as 0
    """

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path)
    reader.Update()

    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(
        0, label
    )  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf.GetOutput())
    else:
        smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(30)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.Update()

    # get only the largest mesh
    remover = vtk.vtkConnectivityFilter()
    remover.SetInputConnection(smoother.GetOutputPort())
    remover.SetExtractionModeToLargestRegion()
    remover.Update()

    # reduce triangles
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputConnection(remover.GetOutputPort())
    if 0 <= decimate_factor < 1:
        decimator.SetTargetReduction(decimate_factor)
        decimator.Update()

    # save the output
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(decimator.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(stl_path)
    writer.Write()


def convert_nrrd_to_nifti(nrrd_path: str, nii_path: str):
    """
    Read a nrrd file and convert it to nifti format.
    """
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(nrrd_path)
    reader.Update()
    info = reader.GetInformation()
    image = reader.GetOutput()

    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(nii_path)
    writer.SetInformation(info)
    writer.Write()


def save_nii_file(raw_img: any, output_data: any, output_path: str):
    """
    Save a nifti file from a numpy array.
    """
    ni_img = nib.Nifti1Image(  # create a nifti image from the numpy array
        output_data.astype(np.uint8), affine=raw_img.affine, header=raw_img.header
    )
    ni_img.header.set_data_dtype(np.uint8)
    nib.save(ni_img, output_path)


def separate_complete_defective_from_labels_file(
    paths: list, skull_type: str, folder_name: str
):
    """
    Create a defective or complete skull labels file from the original labels file.
    """
    for data_name in paths:
        img = nib.load(data_name)  # load the original labels file
        img_data = img.get_fdata()
        output_skull = img_data.copy()
        if skull_type == "defective":
            output_skull[img_data == 2] = 0
        if skull_type == "complete":
            output_skull[img_data == 2] = 1
        output_path = (
            op.join(
                folder_name,
                Path(data_name).parent.name,
                op.splitext(op.basename(data_name))[0],
            )
            + f"_{skull_type}.nii"
        )
        save_nii_file(img, output_skull, output_path)  # save the new labels file


def create_csv_from_input_files(input_path: str, output_csv_name: str):
    """
    Add complete and defective files from the new labels files to the original dataset.
    Create a csv file.

    >>> input_path = "CP_reconstruct_2/229-278" #path to the new labels files
    >>> output_name = "cp_48_defect_complete_path_221220" #output csv file name
    """
    folder_name = op.dirname(input_path)  # get the folder name
    labels_paths = sorted(glob(op.join(input_path, "*", "*labels.nii")))
    for id_path in labels_paths:
        id = Path(id_path).parent.name  # get the skull id
        os.makedirs(
            f"{folder_name}/{id}", exist_ok=True
        )  # create a new folder for each skull id
    skull_types = ["complete", "defective"]
    for skull_type in skull_types:
        separate_complete_defective_from_labels_file(
            labels_paths, skull_type, folder_name
        )
    complete_skull_paths = sorted(glob(f"{folder_name}/*/*complete.nii"))
    defective_skull_paths = sorted(glob(f"{folder_name}/*/*defective.nii"))
    df_csv = pd.DataFrame(
        {
            "defect_skull_path": defective_skull_paths,
            "complete_skull_path": complete_skull_paths,
        }
    )  # create a dataframe with the complete and defective skull paths
    df_csv["skull_id"] = df_csv.defect_skull_path.map(lambda x: Path(x).parent.name)
    df_csv = df_csv[["skull_id", "defect_skull_path", "complete_skull_path"]]
    df_csv.to_csv(
        f"{output_csv_name}.csv", index=False
    )  # save the dataframe as a csv file
