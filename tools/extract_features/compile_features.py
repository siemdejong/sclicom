"""Compile features extracted by pretrained feature extractor."""
import pathlib

# The pyvips import from dlup (dpat dependency) conflicts with torchvision.
# Import dpat first.
from dpat.data import H5Dataset, PMCHHGImageDataset  # isort: skip
from dpat.extract_features.models import SwAV  # isort: skip

from torchvision.transforms import Compose, ToTensor  # isort: skip


def main():
    """Compile features using a trained model and a datamodule."""
    model = SwAV.load_from_checkpoint(
        "/scistor/guest/sjg203/projects/pmc-hhg/dpat/"
        "checkpoints/swav-epoch=0-step=196.ckpt"
    )

    dataset = PMCHHGImageDataset(
        root_dir="/scistor/guest/sjg203/projects/pmc-hhg/images-tif",
        image_paths_and_targets="/scistor/guest/sjg203/projects/pmc-hhg/images-tif/"
        "splits/"
        "medulloblastoma+pilocytic-astrocytoma_pmc-hhg_train-subfold-0-fold-0.csv",
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

    H5Dataset.from_pmchhg_data_and_model(
        filename="test.hdf5",
        dir_name=pathlib.Path("/scistor/guest/sjg203/projects/pmc-hhg/features/"),
        dataset=dataset,
        model=model.backbone,
        num_classes=2,
        dsetname_format=["case_id", "img_id", "tile_region_index"],
        transform=None,
        cache=False,
        mode="a",
        skip_if_exists=True,
    )


if __name__ == "__main__":
    main()
