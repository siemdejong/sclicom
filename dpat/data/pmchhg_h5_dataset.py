"""Provide H5 dataset to read and compile features."""
import logging
import pathlib
from collections import Counter
from contextlib import contextmanager
from pprint import pformat
from typing import Generator, Type, TypeVar, Union

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from dpat.types import SizedMetaDataDataset

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="PMCHHGH5Dataset")
H5ItemObject = dict[str, Union[torch.Tensor, torch.LongTensor, float, int]]


class _H5ls:
    """List all datasets in a given H5 file."""

    def __init__(
        self,
        file_path: pathlib.Path,
        h5type: Union[h5py.Group, h5py.Dataset] = h5py.Dataset,
        depth: int = 3,
    ) -> None:
        """Index H5 file."""
        logger.info(f"Indexing H5 datasets of {file_path}...")

        self.file_path = file_path
        self.names: list[str] = []
        self.h5type = h5type
        self.depth = depth

        with h5py.File(self.file_path, "r") as file:
            file.visititems(self.get_datasets)
        logger.info("Indexing complete.")

    def get_datasets(self, name: str, node: h5py.HLObject) -> None:
        """h5py visititems function to index groups."""
        if isinstance(node, self.h5type) and name not in self.names:
            if len(name.split("/")) == self.depth:
                self.names.append(name)

    def __getitem__(self, index: int) -> str:
        return self.names[index]

    def __iter__(self) -> Generator[str, None, None]:
        yield from self.names

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        """Represent H5ls, showing its origin."""
        return f"H5ls(file_path={self.file_path})"

    def __str__(self) -> str:
        """Prettyformat the print output."""
        return pformat(self.names)


@contextmanager
def careful_hdf5(*args, **kwargs) -> h5py.File:
    """Open an HDF5 file carefully.

    If file already exists, note to give the overwrite keyword.
    If interrupted while open, flush buffers and close the file.

    Raises
    ------
    FileExistsError : if file exists and mode does not allow writing to it.
    """
    try:
        f = h5py.File(*args, **kwargs)
        yield f
    except FileExistsError:
        raise FileExistsError(
            "Preventing overwrite. Use the overwrite keyword to overwrite features."
        )
    except KeyboardInterrupt:
        logger.info("Interrupted...")
    finally:
        f.flush()
        f.close()


def feature_batch_extract(
    f: h5py.File,
    model: nn.Module,
    dataset: SizedMetaDataDataset,
    dsetname_format: list[str],
    skip_if_exists: bool = True,
) -> None:
    """Extract features and store them in a H5 dataset.

    Parameters
    ----------
    f : h5py.File
        File to store the datasets in.
    model : nn.Module
        Model to calculate the features with.
    dataset : SizedMetaDataDataset
        Dataset with metadata and length, with tiles to extract features from.
    dsetname_format : list[str]
        List of strings which determine where the individual items of the dataset
        will be stored in the hierarchical h5 structure.
        The *i*th H5 datasets in the hdf5 file is created at the location which is
        computed like `dataset.get_metadata(i)[dsetname_format...]` where
        `dsetname_format` is a list of strings choosing the metadata to be used to
        name the hierarchical groups and datasets.
    skip_if_exists : bool, default=True
        Skip if dataset objects in hdf5 file already exist.
    """
    # TODO: extract features in batches. It currently takes a long time.
    model.eval()
    dataloader_tiles = DataLoader(dataset=dataset)

    for i, (tile, _) in tqdm(
        enumerate(dataloader_tiles),
        total=len(dataset),
        desc="Extracting HDF5 features",
        unit="tiles",
    ):
        metadata = dataset.get_metadata(i)
        dsetname = "".join([f"/{metadata[dsetname]}" for dsetname in dsetname_format])

        if dsetname in f and skip_if_exists:
            continue

        with torch.no_grad():
            output: torch.Tensor = model(tile)
            features = output.view(-1)

        # E.g. interpolated /case_id/img_id/tile_id.
        dset = f.create_dataset(
            name=dsetname, shape=features.shape, dtype="f", data=features
        )

        dset.attrs.update(metadata["meta"])


def stack_features(
    tempfile: h5py.File, file: h5py.File, skip_if_exists: bool = True
) -> None:
    """Stack features."""
    h5ls_tempfile = _H5ls(tempfile.filename)
    img_groups = np.unique([str(pathlib.Path(item).parent) for item in h5ls_tempfile])

    logger.info("Stacking features.")

    for img_group in tqdm(img_groups, desc="Stacking features, image", unit="images"):
        # Filter which tiles are in the current image group.
        # Make sure every tile is unique.
        tiles_in_img_group = np.unique(
            [
                tile
                for tile in h5ls_tempfile
                if str(pathlib.Path(tile).parent) == img_group
            ]
        )

        assert len(tiles_in_img_group) == len(tempfile[img_group])

        # Assuming the content of the vectors has remained the same,
        # skip concatenating img groups already concatenated.
        # Else, make room for the newly concatenated dataset.
        if img_group in file:
            if len(file[img_group]["data"]) == len(tiles_in_img_group):
                if skip_if_exists:
                    continue
            del file[img_group]

        all_features = []
        all_tile_mpp = []
        all_tile_region_index = []
        all_tile_x = []
        all_tile_y = []
        all_target = []

        for tile in tqdm(
            tiles_in_img_group, desc="Concatenating tile", unit="tile", leave=False
        ):
            all_features.append(tempfile[tile][()])
            all_tile_region_index.append(int(str(tile).split("/")[-1]))
            all_tile_mpp.append(tempfile[tile].attrs["tile_mpp"])
            all_tile_x.append(tempfile[tile].attrs["tile_x"])
            all_tile_y.append(tempfile[tile].attrs["tile_y"])
            all_target.append(tempfile[tile].attrs["target"])

        assert len(set(all_target)) == 1  # All targets must be equal within one img.

        img_group_h5 = file.create_group(img_group)
        img_group_h5.create_dataset("data", data=all_features)
        img_group_h5.create_dataset("all_tile_mpp", data=all_tile_mpp)
        img_group_h5.create_dataset("all_tile_region_index", data=all_tile_region_index)
        img_group_h5.create_dataset("all_tile_x", data=all_tile_x)
        img_group_h5.create_dataset("all_tile_y", data=all_tile_y)
        img_group_h5.create_dataset("all_target", data=all_target)


def compile_features(
    model: nn.Module,
    dataset: SizedMetaDataDataset,
    dir_name: pathlib.Path,
    filename: str,
    dsetname_format: list[str],
    mode: h5py.File.mode = "a",
    skip_if_exists: bool = True,
) -> pathlib.Path:
    """Compile features vectors to H5 format with metadata.

    Parameters
    ----------
    model : `nn.Module`
        Pytorch module to calculate the feature vectors with.
        E.g. this is the backbone attribute of a trained model with SwAV.
    dataset : `MetaDataDataset`
        Must be some dataset where `get_metadata` is defined and yields a metadata
        dict.
    dir_name : `pathlib.Path`
        Directory to store the h5 file.
    filename : `pathlib.Path`
        Name of the h5 file.
    dsetname_format : list[str]
        List of strings which determine where the individual items of the dataset
        will be stored in the hierarchical h5 structure.
        The *i*th H5 datasets in the hdf5 file is created at the location which is
        computed like `dataset.get_metadata(i)[dsetname_format...]` where
        `dsetname_format` is a list of strings choosing the metadata to be used to
        name the hierarchical groups and datasets.
    mode : h5py.File.mode, default="a"
        Mode to open hdf5 file with, using h5py.
        See `h5py.File` for possible modes.
    skip_if_exists : bool, default=True
        Return filename if overwrite

    Returns
    -------
    filepath : `pathlib.Path`
        Path to the compiled h5 file.
    """
    tempfilepath = dir_name / ("temp_" + filename)

    filepath = dir_name / filename

    with careful_hdf5(name=tempfilepath, mode=mode) as tile_file:
        feature_batch_extract(
            tile_file, model, dataset, dsetname_format, skip_if_exists
        )

        with careful_hdf5(name=filepath, mode=mode) as stacked_feature_file:
            stack_features(
                tempfile=tile_file,
                file=stacked_feature_file,
                skip_if_exists=skip_if_exists,
            )

    return filepath


class PMCHHGH5Dataset(Dataset):
    """Dataset for packed HDF5 files to pass to a PyTorch dataloader."""

    def __init__(
        self,
        path: pathlib.Path,
        num_classes: int,
        metadata_keys: Union[list[str], None] = None,
        cache: bool = False,
        transform: torchvision.transforms.Compose = None,
        load_encoded: bool = False,
    ) -> None:
        """Initialize an H5 dataset.

        Parameters
        ----------
        path : `pathlib.Path`
            Location where the images are stored on disk.
        transform : `torchvision.transforms.Compose`, default=None
            Transform to apply.
        load_encoded : bool, default=False
            Whether the images within the h5 file are encoded or saved as bytes
            directly.
        """
        self.file_path = path
        self.num_classes = num_classes
        self.cache = cache
        self.transform = transform
        self.load_encoded = load_encoded

        self.hdf5: Union[h5py.File, None] = None

        self.dataset_indices = _H5ls(self.file_path, h5py.Group, 2)
        if metadata_keys is None:
            self.metadata_keys = ["target"]
        else:
            self.metadata_keys = metadata_keys

        if cache:
            raise NotImplementedError
        if load_encoded:
            raise NotImplementedError

    def get_dataset_at_index(self, index) -> h5py.Dataset:
        """Get dataset at index."""
        # Convert numerical index to string index.
        hdf5_index = self.dataset_indices[index]
        if self.hdf5 is not None:
            return self.hdf5[hdf5_index]
        else:
            return None

    @classmethod
    def from_pmchhg_data_and_model(
        cls: Type[T],
        filename: str,
        dir_name: pathlib.Path,
        dataset: SizedMetaDataDataset,
        model: nn.Module,
        num_classes: int,
        dsetname_format: list[str],
        transform: torchvision.transforms = None,
        cache: bool = False,
        mode: h5py.File.mode = "a",
        skip_if_exists: bool = True,
        skip_feature_compilation: bool = False,
    ) -> T:
        """Build an H5 dataset from PMCHHG dataset with a pretrained model.

        Example structure
        -----------------
        /case/image/tile
        Every tile has `target`, `tile_mpp`, `tile_x`, and `tile_y` attached as
        attributes.

        Parameters
        ----------
        filename : str
            Filename of the dataset.
        dir_name : pathlib.Path
            Directory to save the file to.
        dataset : `SizedMetaDataDataset`
            The dataset to extract hdf5 features from. Must be of
            `SizedMetaDataDataset`.
        model : nn.Module
            The model to use to extract the features.
            E.g. this is the backbone attribute of a trained model with SwAV.
        transform : `torchvision.transforms`
            Must be a torchvision transform.
        cache : bool, default=False
            Cache features during training.
        mode : h5py.File.mode, default="a"
            Mode to open hdf5 file with, using h5py.
            See `h5py.File` for possible modes.
        skip_if_exists : bool, default=True
            If True, does not write new items to h5 file if it already exists,
            but returns file path.
        skip_feature_compilation : bool, default=False
            Skip feature compilation. Asserts a file at hdf5 exists.
        """
        if not skip_feature_compilation:
            filepath = compile_features(
                model=model,
                dataset=dataset,
                dir_name=dir_name,
                filename=filename,
                dsetname_format=dsetname_format,
                mode=mode,
                skip_if_exists=skip_if_exists,
            )
        else:
            filepath = dir_name / filename
            assert pathlib.Path(filepath).exists()
            # TODO: Test if file is compatible with downstream tasks.

        h5_dataset = cls(
            path=filepath,
            num_classes=num_classes,
            metadata_keys=[
                "all_" + key for key in dataset.get_metadata(0)["meta"].keys()
            ],
            cache=cache,
            transform=transform,
            load_encoded=False,
        )

        return h5_dataset

    def compute_sampler_weights(self) -> list[float]:
        """Compute weights to oversample the minority class.

        Pytorch's WeightedRandomSampler requires weights for every
        sample to give probabilities of every sample being drawn.
        """
        all_targets = list(map(lambda tile: self[tile]["target"], range(len(self))))
        counts = Counter(all_targets)
        weights = [1 / counts[i] for i in all_targets]
        return weights

    def __getitem__(self, index: int) -> H5ItemObject:
        """Get feature vectors and metadata at the index.

        Metadata must is assumed to contain at least a "target" key.

        Each worker (which are forked after the init) need to have their own file
        handle [1]. Therefore, the file is opened in __getitem__ once.
        The file is 'weakly' closed [2] when all processes exit and thus del the
        file objects.

        Parameters
        ----------
        index : int
            Index of feature 'tile' in the dataset.

        Returns
        -------
        data_obj : dict
            data : torch.Tensor
                Feature vector of one tile.
            target : torch.Tensor (onehot)
                Target prediction as a one hot vector to support multiclasses.
            tile_mpp : float
                Microns per pixel of the tile.
            tile_x : int
                X-coordinate of the tile in the broader picture.
            tile_y : int
                Y-coordinate of the tile in the broader picture.

        References
        ----------
        [1] https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug
        [2] https://docs.h5py.org/en/stable/high/file.html
        """
        if self.hdf5 is None:
            self.hdf5 = h5py.File(self.file_path, "r")

        dataset = self.get_dataset_at_index(index)
        metadata = {
            key: dataset[key][()] for key in self.metadata_keys if key in dataset
        }

        data = dataset["data"][()]

        if self.transform:
            data = self.transform(data)
        else:
            data = torch.tensor(data)

        # Select only one, as the target is equal for the whole image.
        target = torch.tensor(dataset["all_target"])[0]

        case_id, img_id = str(dataset.name).split("/")[1:]

        data_obj = dict(
            data=data,
            target=target,  # All targets are equal, so just choose one.
            tile_region_index=metadata["all_tile_region_index"],
            tile_mpp=metadata["all_tile_mpp"],
            tile_x=metadata["all_tile_x"],
            tile_y=metadata["all_tile_y"],
            case_id=case_id,
            img_id=img_id,
        )

        return data_obj

    def __len__(self):
        """Calculate length of h5 dataset."""
        return len(self.dataset_indices)


class PMCHHGH5DataModule(pl.LightningDataModule):
    """H5 Datamodule for use with Pytorch Lightning and the PMC-HHG dataset."""

    def __init__(
        self,
        train_path: pathlib.Path,
        val_path: pathlib.Path,
        test_path: pathlib.Path,
        num_workers: int = 0,
        num_classes: int = 2,
        balance: bool = True,
    ):
        """Create PMCHHGH5Dataset DataModule.

        Note that batch sizes are forced to be one (bag).

        Parameters
        ----------
        train_path : pathlib.Path
            Path to train hdf5 file.
        val_path : pathlib.Path
            Path to validation hdf5 file.
        test_path : pathlib.Path
            Path to test hdf5 file.
        num_workers : int, default=0
            Number of workers for the datalaoders.
        num_classes : int, default=2,
            Number of classes for prediction.
        balance : bool, default=True
            Balance the training set using minority oversampling.
        """
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.num_workers = num_workers
        self.num_classes = num_classes

        self.balance = balance

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
        metadata_keys = [
            "all_tile_mpp",
            "all_tile_region_index",
            "all_tile_x",
            "all_tile_y",
        ]
        if stage == "fit":
            self.train_dataset = PMCHHGH5Dataset(
                self.train_path, self.num_classes, metadata_keys
            )
            self.val_dataset = PMCHHGH5Dataset(
                self.val_path, self.num_classes, metadata_keys
            )
        elif stage == "test":
            self.test_dataset = PMCHHGH5Dataset(
                self.test_path, self.num_classes, metadata_keys
            )

    def train_dataloader(self):
        """Build train dataloader."""
        if self.balance:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.compute_sampler_weights(),
                num_samples=len(self.train_dataset),
            )
        return DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=None if self.balance else True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Build validation dataloader."""
        return DataLoader(
            self.val_dataset, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        """Build test dataloader."""
        return DataLoader(
            self.test_dataset, num_workers=self.num_workers, pin_memory=True
        )
