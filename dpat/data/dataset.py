import pathlib

import pandas as pd
import torchvision
from dlup import UnsupportedSlideError
from dlup.data.dataset import ConcatDataset, SlideImage, TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from torch.utils.data import Dataset

from dpat.data.transformations import Dlup2DpatTransform


class PMCHHGImageDataset(Dataset):
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
        """
        Parameters
        ----------
        root_dir : str
            Directory where the images are stored.
        image_paths_and_targets_file : str
            Path to file containing image paths and targets, created by `dpat splits create`.
        mpp : float
            float stating the microns per pixel that you wish the tiles to be.
        tile_size_x : int
            Tuple of integers that represent the size in pixels of output tiles in x-direction
        tile_size_y : int
            Tuple of integers that represent the size in pixels of output tiles in y-direction
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
            Threshold to check against. The foreground percentage should be strictly larger than threshold.
            If None anything is foreground. If 1, the region must be completely foreground.
            Other values are in between, for instance if 0.5, the region must be at least 50% foreground.
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
        print(f"Building dataset...")
        for idx, img_path in enumerate(self.relative_img_paths):
            absolute_img_path = self.root_dir / img_path
            try:
                img = SlideImage.from_file_path(absolute_img_path)
            except UnsupportedSlideError:
                print(f"{absolute_img_path} is unsupported. Skipping image.")
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
        print(f"Built dataset successfully.")

    def num_samples(self) -> int:
        """Size of the dataset."""
        return len(self.dlup_dataset)

    def __len__(self) -> int:
        """Size of the dataset."""
        return self.num_samples()

    def __getitem__(self, index):
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
