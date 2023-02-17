"""Provide Pytorch datasets and Pytorch Lightning datamodules."""
import logging
import pathlib
from functools import partial

import lightning.pytorch as pl
import pandas as pd
import torchvision
from dlup import UnsupportedSlideError
from dlup.data.dataset import ConcatDataset, SlideImage, TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from torch.utils.data import DataLoader, Dataset

from dpat.data.transforms import (
    AvailableTransforms,
    ContrastiveTransform,
    Dlup2DpatTransform,
)

logger = logging.getLogger(__name__)


class PMCHHGImageDataset(Dataset):
    """Dataset for the PMC-HHG project.

    Make tiles of all selected images.
    """

    def __init__(
        self,
        root_dir: str,
        image_paths_and_targets: str,
        mpp: float,
        tile_size_x: int,
        tile_size_y: int,
        tile_overlap_x: int,
        tile_overlap_y: int,
        tile_mode: str,
        crop: bool,
        # mask_factory: str,
        # mask_foreground_threshold: Union[float, None],
        # mask_root_dir: str,
        transform: torchvision.transforms.Compose,
    ):
        """Create dataset.

        Parameters
        ----------
        root_dir : str
            Directory where the images are stored.
        image_paths_and_targets_file : str
            Path to file containing image paths and targets,
            created by `dpat splits create`.
        mpp : float
            float stating the microns per pixel that you wish the tiles to be.
        tile_size_x : int
            Tuple of integers that represent the size in pixels of output tiles in
            x-direction.
        tile_size_y : int
            Tuple of integers that represent the size in pixels of output tiles in
            y-direction.
        tile_overlap : int
            Tuple of integers that represents the overlap of tiles in the x-direction.
        tile_overlap : int
            Tuple of integers that represents the overlap of tiles in the x-direction.
        tile_mode : skip|overflow
            See `dlup.tiling.TilingMode` for more information
        crop : bool
             If overflowing tiles should be cropped.
        mask_factory : str
            How to load masks. Must be `load_from_disk` or `no_mask`.
        mask_foreground_threshold : float
            Threshold to check against. The foreground percentage should be strictly
            larger than threshold. If None anything is foreground. If 1, the region must
            be completely foreground. Other values are in between, for instance if 0.5,
            the region must be at least 50% foreground.
        transform : `torchvision.transforms.Compose
            Transform to be applied to the sample.
        """
        super().__init__()

        self.root_dir = pathlib.Path(root_dir)

        path_image_paths_and_targets = pathlib.Path(image_paths_and_targets)
        self.df = pd.read_csv(path_image_paths_and_targets, header=None)
        self.relative_img_paths = self.df[0]

        self.df = self.df.set_index(0)

        tile_mode = TilingMode[tile_mode]

        self.transform = Dlup2DpatTransform(transform)

        # TODO: instantiate MaskGetter

        # Build dataset
        single_img_datasets: list = []
        logger.info("Building dataset...")
        for _, img_path in enumerate(self.relative_img_paths):
            absolute_img_path = self.root_dir / img_path
            try:
                _ = SlideImage.from_file_path(absolute_img_path)
            except UnsupportedSlideError:
                logger.info(f"{absolute_img_path} is unsupported. Skipping image.")
                continue

            # TODO: get mask

            single_img_datasets.append(
                TiledROIsSlideImageDataset.from_standard_tiling(
                    path=absolute_img_path,
                    mpp=mpp,
                    tile_size=(tile_size_x, tile_size_y),
                    tile_overlap=(tile_overlap_x, tile_overlap_y),
                    tile_mode=tile_mode,
                    crop=crop,
                    # mask=mask,
                    # mask_threshold=self.foreground_threshold,
                    transform=self.transform,
                )
            )

        self.dlup_dataset = ConcatDataset(single_img_datasets)
        logger.info("Built dataset successfully.")

    def num_samples(self) -> int:
        """Size of the dataset."""
        return len(self.dlup_dataset)

    def __getitem__(self, index):
        """Get one tile and its target, along with metadata."""
        sample = self.dlup_dataset[index]
        relative_path = str(pathlib.Path(sample["path"]).relative_to(self.root_dir))
        (case_id, img_id, target) = self.df.loc[relative_path, [1, 2, 3]]
        return_object = {
            "x": sample["image"],
            "y": int(target),
            "slide_id": img_id,
            "patient_id": case_id,
            "paths": str(relative_path),
            "root_dir": str(self.root_dir),
            "meta": {
                "tile_x": sample["coordinates"][0],
                "tile_y": sample["coordinates"][1],
                "tile_mpp": sample["mpp"],
                # "tile_w": sample["region_size"][0],
                # "tile_h": sample["region_size"][1],
                # "tile_region_index": sample["region_index"],
            },
        }
        return return_object

    def __len__(self) -> int:
        """Size of the dataset."""
        return self.num_samples()


class PMCHHGImageDataModule(pl.LightningDataModule):
    """Datamodule for structuring building of Pytorch DataLoader."""

    def __init__(
        self,
        root_dir: str,
        train_img_paths_and_targets: str,
        val_img_paths_and_targets: str,
        test_img_paths_and_targets: str,
        mpp: float,
        tile_size_x: int,
        tile_size_y: int,
        tile_overlap_x: int,
        tile_overlap_y: int,
        tile_mode: str,
        crop: bool,
        # mask_factory: str,
        # mask_foreground_threshold: float,
        # mask_root_dir: str,
        num_workers: int,
        batch_size: int,
        transform: list[AvailableTransforms],
    ) -> None:
        """Create datamodule.

        Raises
        ------
        ValueError
            If `transform` is unavailable.
        """
        super().__init__()
        self.save_hyperparameters()
        # self.prepare_data_per_node = True

        self.num_workers = num_workers
        self.root_dir = pathlib.Path(root_dir)
        self.train_path = pathlib.Path(train_img_paths_and_targets)
        self.val_path = pathlib.Path(val_img_paths_and_targets)
        self.test_path = pathlib.Path(test_img_paths_and_targets)
        self.mpp = mpp
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.tile_overlap_x = tile_overlap_x
        self.tile_overlap_y = tile_overlap_y
        self.crop = crop
        self.tile_mode = tile_mode
        # self.mask_factory = mask_factory
        # self.mask_foreground_threshold = mask_foreground_threshold
        # self.mask_root_dir = mask_root_dir
        self.batch_size = batch_size

        assert tile_size_x == tile_size_y

        if "contrastive" in transform:
            self.transform = ContrastiveTransform()
        else:
            raise ValueError(
                "Please set transform to a list with transforms from"
                f" {AvailableTransforms._member_names_}"
            )

    def prepare_data(self):
        """Prepare data."""
        # TODO: if storing the data somewhere in the cloud
        # and it is downloaded in prepare_data,
        # use setup() to make the splits with dpat.splits.create_splits.
        pass

    def setup(self, stage):
        """Split dataset and apply stage transform.

        Is done on every device.
        """
        self.dataset = partial(
            PMCHHGImageDataset(
                root_dir=self.root_dir,
                mpp=self.mpp,
                tile_size_x=self.tile_size_x,
                tile_size_y=self.tile_size_y,
                tile_overlap_x=self.tile_overlap_x,
                tile_overlap_y=self.tile_overlap_y,
                tile_mode=self.tile_mode,
                crop=self.crop,
                # mask_factory=self.mask_factory,
                # mask_foreground_threshold=self.mask_foreground_threshold,
                # mask_root_dir=self.mask_root_dir,
                transform=self.transform,
            )
        )

        if stage == "fit":
            self.train_dataset, self.val_dataset = [
                self.dataset(image_paths_and_targets=paths)
                for paths in [self.train_path, self.val_path]
            ]
        elif stage == "test":
            self.test_dataset = self.dataset(images_paths_and_targets=self.train_path)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Before transfer to device, do ..."""
        # Using self.trainer.train/validation/test,
        # different transformations can be applied here.

        # batch['x'] = transforms(batch['x'])
        # return batch
        pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """After transfer to device, do ..."""
        # batch['x'] = gpu_transforms(batch['x'])
        # return batch
        pass

    def train_dataloader(self):
        """Build train dataloader."""
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        """Build validation dataloader."""
        return DataLoader(
            self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size
        )

    def test_dataloader(self):
        """Build test dataloader."""
        return DataLoader(
            self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size
        )

    def teardown(self, stage):
        """Cleanup."""
        pass
        # clean up after fit or test
        # called on every process in DDP
