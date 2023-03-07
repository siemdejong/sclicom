"""Provide datasets and transforms."""
import platform

from .pmc_tile_dataset import PMCHHGImageDataModule, PMCHHGImageDataset
from .pmchhg_h5_dataset import PMCHHGH5DataModule, PMCHHGH5Dataset
from .transforms import Dlup2DpatTransform

if platform.system() == "Windows":
    import dpat

    dpat.install_windows(r"D:\apps\vips-dev-8.14\bin")

__all__ = [
    "PMCHHGImageDataModule",
    "PMCHHGImageDataset",
    "PMCHHGH5DataModule",
    "PMCHHGH5Dataset",
    "Dlup2DpatTransform",
]
