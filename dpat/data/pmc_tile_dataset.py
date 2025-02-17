"""Provide Pytorch datasets and Pytorch Lightning datamodules."""
import logging
import pathlib
from io import BytesIO
from typing import Any, Literal, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchvision
from dlup import UnsupportedSlideError
from dlup.data.dataset import ConcatDataset, SlideImage, TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.swav_transform import SwaVTransform
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dpat.data.transforms import AvailableTransforms, Dlup2DpatTransform
from dpat.types import MaskFactory, SizedDataset

logger = logging.getLogger(__name__)


class MaskGetter:
    """Set parameters required for getting a mask."""

    def __init__(
        self, mask_factory: MaskFactory, mask_root_dir: Union[str, None] = None
    ):
        """Initialize MaskGetter.

        Parameters
        ----------
        mask_factory : "no_mask"
            Which factory to use to create masks.
        mask_root_dir : str, None (default=None)
            Root dir where to place/find the masks.
        """
        self.mask_options = {
            "load_from_disk": self.load_from_disk,
            "no_mask": self.no_mask,
        }
        self.mask_factory = mask_factory

        self.mask_root_dir: Union[pathlib.Path, None]
        if self.mask_factory == "load_from_disk" and mask_root_dir is not None:
            assert pathlib.Path(mask_root_dir).is_dir()
            self.mask_root_dir = pathlib.Path(mask_root_dir)
        else:
            self.mask_root_dir = None

        self.current_slide_image = None
        self.current_idx: Union[int, None] = None

    def return_mask_from_config(
        self, img: pathlib.Path, idx: int, relative_img_path: pathlib.Path
    ):
        """Return a mask with the given mask_factory."""
        self.current_idx = idx
        mask = self.mask_options[self.mask_factory](
            img=img, relative_img_path=relative_img_path
        )
        return mask

    def load_from_disk(self, *args, **kwargs):
        """Load mask from disk.

        Read a .png saved by DLUP and converts into a npy object.
        """
        mask_file_path = self.mask_root_dir / (
            kwargs["relative_img_path"].stem + "-mask.png"
        )
        mask = np.array(Image.open(mask_file_path)).astype(np.uint8)
        return mask

    def no_mask(self, *args, **kwargs):
        """Return no mask."""
        return None


class PMCHHGImageDataset(Dataset):
    """Dataset for the PMC-HHG project.

    Make tiles of all selected images. The tiles are at the requested
    mpp resolution.
    """

    # Precalculated. Assumed as "domain knowledge".
    NORMALIZE = {"mean": [0.0692, 0.1651, 0.0257], "std": [0.1059, 0.1531, 0.0329]}

    # Calculated on medulloblastoma+pilocytic astrocytoma with images until case 140.
    NORMALIZE_MASKED = {
        "mean": [0.0842, 0.2062, 0.0275],
        "std": [0.1152, 0.1471, 0.0344],
    }

    def __init__(
        self,
        root_dir: str,
        image_paths_and_targets: Union[pathlib.Path, str, BytesIO],
        clinical_context: bool = False,
        mpp: float = 0.2,
        tile_size_x: int = 224,
        tile_size_y: int = 224,
        tile_overlap_x: int = 0,
        tile_overlap_y: int = 0,
        tile_mode: str = "skip",
        crop: bool = False,
        mask_factory: MaskFactory = "no_mask",
        mask_foreground_threshold: Union[float, None] = 1,
        mask_root_dir: Union[str, None] = None,
        transform: Union[torchvision.transforms.Compose, None] = None,
    ):
        """Create dataset.

        Parameters
        ----------
        root_dir : str
            Directory where the images are stored.
        image_paths_and_targets_file : str
            Path to file containing image paths and targets,
            created by `dpat splits create`.
        mpp : float, default=0.2
            float stating the microns per pixel that you wish the tiles to be.
        tile_size_x : int, default=224
            Tuple of integers that represent the size in pixels of output tiles in
            x-direction.
        tile_size_y : int, default=224
            Tuple of integers that represent the size in pixels of output tiles in
            y-direction.
        tile_overlap_x : int, default=0
            Tuple of integers that represents the overlap of tiles in the x-direction.
        tile_overlap_y : int, default=0
            Tuple of integers that represents the overlap of tiles in the x-direction.
        tile_mode : skip|overflow, default=overflow
            See `dlup.tiling.TilingMode` for more information
        crop : bool, default=False
             If overflowing tiles should be cropped.
        mask_factory : str, default="no_mask"
            How to load masks. Must be `load_from_disk` or `no_mask`.
        mask_foreground_threshold : float|None, default=None
            Threshold to check against. The foreground percentage should be strictly
            larger than threshold. If None anything is foreground. If 1, the region must
            be completely foreground. Other values are in between, for instance if 0.5,
            the region must be at least 50% foreground.
        transform : `torchvision.transforms.Compose, default=None
            Transform to be applied to the sample.
        clinical context : bool, default=False
            Fetch the clinical context from the labels file.
        """
        super().__init__()

        if mask_factory == "load_from_disk":
            assert mask_root_dir is not None, f"If {mask_factory=}, "
            "mask_root_dir must be set."

        self.root_dir = pathlib.Path(root_dir)

        self.clinical_context = clinical_context

        if isinstance(image_paths_and_targets, str):
            image_paths_and_targets = pathlib.Path(image_paths_and_targets)
        self.image_paths_and_targets = image_paths_and_targets
        self.df = pd.read_csv(image_paths_and_targets, header=None)
        self.relative_img_paths = self.df[0]

        # Reset for if it needs to be reused somewhere else, e.g. in PMCHHGH5Dataset.
        if not isinstance(self.image_paths_and_targets, (pathlib.Path, str)):
            self.image_paths_and_targets.seek(0)

        self.df = self.df.set_index(0)

        tile_mode = TilingMode[tile_mode]

        self.transform = Dlup2DpatTransform(transform)

        self.foreground_threshold: Union[float, None]
        if mask_factory != "no_mask":
            self.foreground_threshold = mask_foreground_threshold
        else:
            # DLUP dataset erroneously requires a float instead of optional None
            self.foreground_threshold = 0.1

        self.mask_getter = MaskGetter(
            mask_factory=mask_factory, mask_root_dir=mask_root_dir
        )

        # Build dataset
        single_img_datasets: list = []
        logger.info("Building dataset...")
        logger.info(f"Reading images from {self.root_dir}")
        for idx, relative_img_path in enumerate(self.relative_img_paths):
            absolute_img_path = self.root_dir / relative_img_path
            try:
                img = SlideImage.from_file_path(absolute_img_path)
            except UnsupportedSlideError:
                logger.info(f"{absolute_img_path} is unsupported. Skipping image.")
                continue

            mask = self.mask_getter.return_mask_from_config(
                img=img, idx=idx, relative_img_path=pathlib.Path(relative_img_path)
            )

            single_img_datasets.append(
                TiledROIsSlideImageDataset.from_standard_tiling(
                    path=absolute_img_path,
                    mpp=mpp,
                    tile_size=(tile_size_x, tile_size_y),
                    tile_overlap=(tile_overlap_x, tile_overlap_y),
                    tile_mode=tile_mode,
                    crop=crop,
                    mask=mask,
                    mask_threshold=self.foreground_threshold,
                    transform=self.transform,
                )
            )

        self.dlup_dataset = ConcatDataset(single_img_datasets)
        logger.info("Built dataset successfully.")

    def num_samples(self) -> int:
        """Size of the dataset."""
        return len(self.dlup_dataset)

    @staticmethod
    def path_to_image(dataset: Dataset, index: int):
        """Return filename of file at the index.

        Used by lightly `index_to_filename`.

        Parameters
        ----------
        dataset : `Dataset`
            Dataset with the images.
        index : int
            Index in the `dataset`.
        """
        sample = dataset[index]
        return sample["path"]

    def get_metadata(
        self,
        index: int,
        from_tile_rois_slide_image_dataset: Union[
            TiledROIsSlideImageDataset, None
        ] = None,
    ) -> Any:  # TODO: Add strong typing.
        """Get metadata of a sample.

        Parameters
        ----------
        index : int
            Index of tile dataset to fetch metadata from.
        from_tile_rois_slide_image_dataset : `TiledROIsSlideImageDataset`
            Specific inner dataset to read tiles from.
        """
        if from_tile_rois_slide_image_dataset is not None:
            sample = from_tile_rois_slide_image_dataset[index]
        else:
            sample = self.dlup_dataset[index]

        relative_path = str(pathlib.Path(sample["path"]).relative_to(self.root_dir))
        case_id, img_id, target = self.df.loc[relative_path, [1, 2, 3]]

        if self.clinical_context:
            location = self.df.loc[relative_path, [4]]

        metadata = {
            "img_id": img_id,
            "case_id": case_id,
            "tile_region_index": sample["region_index"],
            "meta": {
                "target": int(target),
                "tile_mpp": sample["mpp"],
                "tile_x": sample["coordinates"][0],
                "tile_y": sample["coordinates"][1],
                "location": location,
                # 'tile_w': sample["region_size"][0],
                # 'tile_h': sample["region_size"][1],
            },
        }

        return metadata

    def __getitem__(self, index):
        """Get one tile and its target."""
        sample = self.dlup_dataset[index]

        relative_path = str(pathlib.Path(sample["path"]).relative_to(self.root_dir))
        target = self.df.loc[relative_path, 3]
        return sample["image"], target

    def __len__(self) -> int:
        """Size of the dataset."""
        return self.num_samples()


def online_mean_and_std(
    dataset: SizedDataset, batch_size: int = 64, device: str = "cuda"
) -> tuple[Tensor, Tensor]:
    """Calculate mean and std in an online fashion.

    Calculates the mean and std by looping through the dataset two times.

    Parameters
    ----------
    dataset : `Dataset`
        Dataset to calculate the mean and std for. __getitem__ must return
        an rgb image as the first return value.
    batch_size : int, default=32
        Batch size of the dataloader.
    device : cpu/cuda
        Device to transfer the data to.

    References
    ----------
    https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9
    """
    dataloader: DataLoader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False
    )

    _sum: torch.Tensor = torch.zeros(3, device=device)
    _sum_sq: torch.Tensor = torch.zeros(3, device=device)
    total_pixels: int = 0

    for batch in tqdm(dataloader, desc="Calculating sum/sumsq", unit="batch"):
        data = batch[0].to(device)
        b, _, h, w = data.shape
        nb_pixels = b * h * w
        _sum += torch.sum(data, dim=[0, 2, 3])
        _sum_sq += torch.sum(data**2, dim=[0, 2, 3])
        total_pixels += nb_pixels

    mean = _sum / total_pixels
    var = (_sum_sq / total_pixels) - (mean**2)
    std = torch.sqrt(var)

    return mean, std


class PMCHHGImageDataModule(pl.LightningDataModule):
    """Datamodule for structuring building of Pytorch DataLoader."""

    def __init__(
        self,
        model: Union[Literal["swav", "simclr"], None],
        root_dir: str,
        train_img_paths_and_targets: str,
        val_img_paths_and_targets: str,
        test_img_paths_and_targets: str,
        clinical_context: bool = False,
        mpp: float = 0.2,
        tile_size_x: int = 224,
        tile_size_y: int = 224,
        tile_overlap_x: int = 0,
        tile_overlap_y: int = 0,
        tile_mode: str = "skip",
        crop: bool = False,
        mask_factory: MaskFactory = "no_mask",
        mask_foreground_threshold: Union[float, None] = 1,
        mask_root_dir: Union[str, None] = None,
        num_workers: int = 4,
        batch_size: int = 512,
        color_jitter: bool = True,
        transform: Union[list[AvailableTransforms], None] = None,
        **kwargs,
    ) -> None:
        """Create datamodule.

        Parameters
        ----------
        model : "swav"
            Model that the dataset is used with. Some models need a special collate
            function.
        root_dir : str
            Directory where the images are stored.
        train_img_paths_and_targets : str
            Path to file containing training image paths and targets,
            created by `dpat splits create`.
        val_img_paths_and_targets : str
            Path to file containing validation image paths and targets,
            created by `dpat splits create`.
        test_img_paths_and_targets : str
            Path to file containing testing image paths and targets,
            created by `dpat splits create`.
        clinical_context : bool, default=False
            Export clinical context with tiles.
        mpp : float
            float stating the microns per pixel that you wish the tiles to be.
        tile_size_x : int
            Tuple of integers that represent the size in pixels of output tiles in
            x-direction.
        tile_size_y : int
            Tuple of integers that represent the size in pixels of output tiles in
            y-direction.
        tile_overlap_x : int
            Tuple of integers that represents the overlap of tiles in the x-direction.
        tile_overlap_y : int
            Tuple of integers that represents the overlap of tiles in the x-direction.
        tile_mode : skip|overflow
            See `dlup.tiling.TilingMode` for more information.
        crop : bool
            If overflowing tiles should be cropped.
        num_workers : int
            Num workers for the dataloader.
        batch_size : int
            Batch size.
        transform : `torchvision.transforms.Compose`
            Transform to be applied to the sample.

        Raises
        ------
        ValueError
            If `transform` is unavailable.
        """
        super().__init__()
        # self.prepare_data_per_node = True

        self.model = model.lower() if model is not None else model
        self.num_workers = num_workers
        self.root_dir = pathlib.Path(root_dir)
        self.train_path = pathlib.Path(train_img_paths_and_targets)
        self.val_path = pathlib.Path(val_img_paths_and_targets)
        self.test_path = pathlib.Path(test_img_paths_and_targets)
        self.clinical_context = clinical_context
        self.mpp = mpp
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.tile_overlap_x = tile_overlap_x
        self.tile_overlap_y = tile_overlap_y
        self.crop = crop
        self.tile_mode = tile_mode
        self.mask_factory = mask_factory
        self.mask_foreground_threshold = mask_foreground_threshold
        self.mask_root_dir = mask_root_dir
        self.batch_size = batch_size
        self.color_jitter = color_jitter

        assert tile_size_x == tile_size_y

        if transform is not None:
            if self.mask_factory != "no_mask":
                normalize = PMCHHGImageDataset.NORMALIZE_MASKED
            else:
                normalize = PMCHHGImageDataset.NORMALIZE

            transform_kwargs = dict(
                normalize=normalize,
                rr_prob=1,
                rr_degrees=180,
                cj_prob=1 if self.color_jitter else 0,
            )
            if self.model == "swav":
                self.transform = SwaVTransform(**transform_kwargs)
            elif self.model == "simclr":
                self.transform = SimCLRTransform(**transform_kwargs)
            else:
                raise NotImplementedError(
                    f"Special transforms for model {self.model} are not implemented."
                )
        else:
            self.transform = transform

    @property
    def collate_fn(self):
        """Return the collate function related to the model."""
        if self.model in ["swav", "simclr"]:
            collate_fn = MultiViewCollate()
        else:
            collate_fn = None

        return collate_fn

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
        dataset_kwargs = dict(
            root_dir=self.root_dir,
            clinical_context=self.clinical_context,
            mpp=self.mpp,
            tile_size_x=self.tile_size_x,
            tile_size_y=self.tile_size_y,
            tile_overlap_x=self.tile_overlap_x,
            tile_overlap_y=self.tile_overlap_y,
            tile_mode=self.tile_mode,
            crop=self.crop,
            mask_factory=self.mask_factory,
            mask_foreground_threshold=self.mask_foreground_threshold,
            mask_root_dir=self.mask_root_dir,
            transform=self.transform,
        )

        if stage == "train":
            self.train_dataset = LightlyDataset.from_torch_dataset(
                PMCHHGImageDataset(
                    image_paths_and_targets=self.train_path, **dataset_kwargs
                )
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Build train dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
