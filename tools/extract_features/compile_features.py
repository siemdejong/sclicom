"""Compile features extracted by pretrained feature extractor."""
import logging
import pathlib
import tempfile

import torch
from torchvision.transforms import Compose, Normalize, ToTensor  # isort: ignore

# The pyvips import from dlup (dpat dependency) conflicts with torchvision.
# Import dpat first.
from dpat.data import PMCHHGH5Dataset, PMCHHGImageDataset
from dpat.extract_features.models import SimCLR

logger = logging.getLogger(__name__)


def main():
    """Compile features using a trained model and a datamodule."""
    # model = SimCLR.load_from_checkpoint(
    #     "/gpfs/home2/sdejong/pmchhg/dpat/lightning_logs/version_2460498/checkpoints/last.ckpt"  # noqa: E501
    # )
    model = SimCLR(pretrained=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

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
        tile_mode="skip",
        crop=False,
        mask_factory="load_from_disk",
        mask_foreground_threshold=1,
        mask_root_dir="/gpfs/home2/sdejong/pmchhg/masks/",
        clinical_context=True,
        transform=Compose(
            [
                ToTensor(),
                Normalize(
                    # Change to NORMALIZE if not using masks.
                    **PMCHHGImageDataset.NORMALIZE_MASKED
                ),
            ]
        ),
    )

    _ = PMCHHGH5Dataset.from_pmchhg_data_and_model(
        filename="simclr-20-3-2023.hdf5",
        dir_name=pathlib.Path("/gpfs/home2/sdejong/pmchhg/features/"),
        dataset=dataset,
        model=model,
        num_classes=2,
        dsetname_format=["case_id", "img_id"],
        transform=None,
        cache=False,
        mode="a",
        skip_if_exists=False,
        skip_feature_compilation=False,
        batch_size=512,
        num_workers=10,
        clinical_context=True,
    )

    concatenated_file.close()


if __name__ == "__main__":
    main()
