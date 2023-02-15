#! /usr/bin/env python
from setuptools import setup
from autoimplantpipe import __version__

if __name__ == "__main__":
    setup(
        name="autoimplantpipe",
        version=__version__,
        description="A Python package for autoimplant pipeline",
        url="https://github.com/biodatlab/autoimplantpipe",
        download_url="https://github.com/biodatlab/autoimplantpipe.git",
        author="Napasara Asawalertsak, Titipat Achakulvisut, Thatchapatt Kesornsri, Zaw Htet Aung, Pornnapas Manowongpichate, Peeranat Buabang",
        author_email="",
        license="",
        install_requires=[
            "nibabel",
            "matplotlib>=3.1.3",
            "SimpleITK>=2.2.0",
            "itkwidgets",
            "vtk",
            "scikit-image",
            "scikit-learn",
            "torch",
            "monai",
            "pandas",
            "wandb",
            "pynrrd",
        ],
        packages=[
            "autoimplantpipe"
        ],
        keywords=[
            "Python",
            "Autoimplant",
            "Skull segmentation",
            "Skull registration",
            "Skull autoimplant",
        ],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        platforms="any",
        project_urls={
            "Source": "https://github.com/biodatlab/autoimplantpipe",
            "Documentation": "https://github.com/biodatlab/autoimplantpipe",
            "Bug Reports": "https://github.com/biodatlab/autoimplantpipe/issues",
        },
    )
