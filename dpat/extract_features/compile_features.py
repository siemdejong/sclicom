"""Compile features extracted by pretrained feature extractor."""
import pathlib

import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from dpat.data import PMCHHGImageDataset
from dpat.extract_features.models import SwAV


class HDF5Dataset:
    """Dataset that saves feature vectors in HDF5 format."""

    def __compile_features(self):
        self.model.eval()
        dataloader_tiles = DataLoader(self.dataset)

        mode = "w" if self.overwrite else "w-"

        try:
            f = h5py.File(str(self.dir / self.filename), mode)

            # TODO: extract features in batches. It currently takes a long time.
            for i, (tile, _) in tqdm(
                enumerate(dataloader_tiles),
                total=len(self.dataset),
                desc="Extracting HDF5 features",
            ):
                metadata = self.dataset.get_metadata(i)

                with torch.no_grad():
                    features = self.model.backbone(tile).flatten()

                dsetname = (
                    f"/{metadata['case_id']}"
                    f"/{metadata['img_id']}"
                    f"/{metadata['tile_region_index']}"
                )
                dset = f.create_dataset(
                    name=dsetname, shape=features.shape, dtype="f", data=features
                )

                dset.attrs.update(metadata["meta"])
        except FileExistsError:
            raise FileExistsError(
                "Preventing overwrite. Use the overwrite keyword to overwrite features."
            )
        except KeyboardInterrupt:
            pass
        finally:
            f.close()

    def __init__(
        self,
        filename: str,
        dir: pathlib.Path,
        dataset: PMCHHGImageDataset,
        model: nn.Module,
        overwrite: bool = False,
    ) -> None:
        """Build an HDF5 dataset.

        Structure
        ---------
        /case/image/tile
        Every tile has `target`, `tile_mpp`, `tile_x`, and `tile_y` attached as
        attributes.

        Parameters
        ----------
        name : str
            Filename of the dataset.
        dir : pathlib.Path
            Directory to save the file to.
        dataset : PMCHHGImageDataset
            The dataset to extract hdf5 features from. Must be of PMCHHGImageDataset.
        model : nn.Module
            The model to use to extract the features. Must have a `backbone` attribute.
        overwrite : bool, default=False
            If True, overwrite items from the file given by `dir`/`filename`.
        """
        self.filename = filename
        self.dir = dir
        self.dataset = dataset
        self.model = model
        self.overwrite = overwrite

        self.__compile_features()


if __name__ == "__main__":
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

    HDF5Dataset(
        "test.hdf5",
        dir=pathlib.Path("/scistor/guest/sjg203/projects/pmc-hhg/features"),
        dataset=dataset,
        model=model,
        overwrite=True,
    )
