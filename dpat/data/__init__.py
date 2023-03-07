"""Provide datasets and transforms."""
import platform

from .pmc_tile_dataset import PMCHHGImageDataModule, PMCHHGImageDataset  # noqa: F401
from .pmchhg_h5_dataset import PMCHHGH5DataModule, PMCHHGH5Dataset  # noqa: F401
from .transforms import Dlup2DpatTransform  # noqa: F401

if platform.system() == "Windows":
    import dpat

    dpat.install_windows(r"D:\apps\vips-dev-8.14\bin")
