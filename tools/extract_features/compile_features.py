"""Compile features extracted by pretrained feature extractor."""
import logging
import pathlib
import tempfile

# The pyvips import from dlup (dpat dependency) conflicts with torchvision.
# Import dpat first.
from dpat.data import PMCHHGH5Dataset, PMCHHGImageDataset  # isort: skip
from dpat.extract_features.models import SwAV, SimCLR  # isort: skip # noqa: F401

from torchvision.transforms import Compose, ToTensor  # isort: skip

logger = logging.getLogger(__name__)


def main():
    """Compile features using a trained model and a datamodule."""
    model = SimCLR.load_from_checkpoint(
        "/scistor/guest/sjg203/projects/pmc-hhg/snellius-simclr.ckpt"
    )

    # Make a temporary file make the dataset read from.
    concatenated_file = tempfile.TemporaryFile()

    base_file = (
        "/scistor/guest/sjg203/projects/pmc-hhg/"
        "medulloblastoma+pilocytic-astrocytoma_"
        "pmchhg_{fold}-subfold-0-fold-0.csv"
    )

    read_files = [base_file.format(fold=fold) for fold in ["train", "val", "test"]]

    for f in read_files:
        with open(f, "rb") as infile:
            concatenated_file.write(infile.read())

    # Go back to the beginning of the file.
    concatenated_file.seek(0)

    dataset = PMCHHGImageDataset(
        root_dir="/scistor/guest/sjg203/projects/pmc-hhg/images-tif",
        image_paths_and_targets=concatenated_file,
        mpp=0.2,
        tile_size_x=224,
        tile_size_y=224,
        tile_overlap_x=0,
        tile_overlap_y=0,
        tile_mode="overflow",
        crop=False,
        mask_factory="no_mask",
        mask_foreground_threshold=None,
        mask_root_dir=None,
        transform=Compose([ToTensor()]),
    )

    _ = PMCHHGH5Dataset.from_pmchhg_data_and_model(
        filename="simclr-13-3-2023.hdf5",
        dir_name=pathlib.Path("/scistor/guest/sjg203/projects/pmc-hhg/features"),
        dataset=dataset,
        model=model,
        num_classes=2,
        dsetname_format=["case_id", "img_id"],
        transform=None,
        cache=False,
        mode="a",
        skip_if_exists=True,
        skip_feature_compilation=False,
        batch_size=256,
    )

    concatenated_file.close()


if __name__ == "__main__":
    main()
