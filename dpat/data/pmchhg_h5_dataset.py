"""Provide H5 dataset to read and compile features."""
import logging
import pathlib
from pprint import pformat
from typing import Union

import h5py
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dpat.types import SizedMetaDataDataset

logger = logging.getLogger(__name__)


H5ItemObject = dict[str, Union[torch.Tensor, torch.LongTensor, float, int]]


class _H5ls:
    """List all datasets in a given H5 file."""

    def __init__(self, file_path: pathlib.Path) -> None:
        """Index H5 file."""
        logger.info(f"Indexing H5 datasets of {file_path}...")

        self.file_path = file_path
        self.names: list = []

        with h5py.File(self.file_path, "r") as file:
            file.visititems(self.get_datasets)
        logger.info("Indexing complete.")

    def get_datasets(self, name: str, node: h5py.HLObject) -> None:
        """h5py visititems function to index datasets."""
        if isinstance(node, h5py.Dataset) and name not in self.names:
            self.names.append(name)

    def __repr__(self) -> str:
        """Represent H5ls, showing its origin."""
        return f"H5ls(file_path={self.file_path})"

    def __str__(self) -> str:
        """Prettyformat the print output."""
        return pformat(self.names)


def compile_features(
    model: nn.Module,
    dataset: SizedMetaDataDataset,
    dir_name: pathlib.Path,
    filename: pathlib.Path,
    dsetname_format: list[str],
    overwrite: bool = False,
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
    overwrite : bool, default=False
        Overwrite items in the hdf5 file.
    skip_if_exists : bool, default=True
        Return filename if overwrite

    Returns
    -------
    filepath : `pathlib.Path`
        Path to the compiled h5 file.
    """
    model.eval()
    dataloader_tiles = DataLoader(dataset)

    filepath = dir_name / filename
    if skip_if_exists and filepath.exists():
        return filepath

    mode = "w" if overwrite else "w-"

    try:
        f = h5py.File(str(filepath), mode)

        # TODO: extract features in batches. It currently takes a long time.
        for i, (tile, _) in tqdm(
            enumerate(dataloader_tiles),
            total=len(dataset),
            desc="Extracting HDF5 features",
            unit="tiles",
        ):
            metadata = dataset.get_metadata(i)

            with torch.no_grad():
                output: torch.Tensor = model(tile)
                features = output.view(-1)

            dsetname = "".join(
                [f"/{metadata[dsetname]}" for dsetname in dsetname_format]
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
        f.flush()
        logger.info("Interrupted...")
    finally:
        f.close()

    return filepath


class H5Dataset(Dataset):
    """Dataset for packed HDF5 files to pass to a PyTorch dataloader.

    TODO: ADD MOVE COMPILATION TO HERE.
    """

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

        self.hdf5 = None

        self.dataset_indices = _H5ls(self.file_path)
        if metadata_keys is None:
            self.metadata_keys = ["target"]
        else:
            self.metadata_keys = metadata_keys

        if cache:
            raise NotImplementedError
        if load_encoded:
            raise NotImplementedError

    def get_dataset_at_index(self, index):
        """Get dataset at index."""
        # Convert numerical index to string index.
        hdf5_index = self.dataset_indices[index]
        dataset = self.hdf5[hdf5_index]

        return dataset

    @classmethod
    def from_pmchhg_data_and_model(
        cls,
        filename: pathlib.Path,
        dir_name: pathlib.Path,
        dataset: SizedMetaDataDataset,
        model: nn.Module,
        num_classes: int,
        dsetname_format: list[str],
        transform: torchvision.transforms = None,
        cache: bool = False,
        overwrite: bool = False,
        skip_if_exists: bool = True,
    ) -> "H5Dataset":
        """Build an H5 dataset from PMCHHG dataset with a pretrained model.

        Example structure
        -----------------
        /case/image/tile
        Every tile has `target`, `tile_mpp`, `tile_x`, and `tile_y` attached as
        attributes.

        Parameters
        ----------
        filename : pathlib.Path
            Filename of the dataset.
        dir_name : pathlib.Path
            Directory to save the file to.
        dataset : PMCHHGImageDataset
            The dataset to extract hdf5 features from. Must be of PMCHHGImageDataset.
        model : nn.Module
            The model to use to extract the features.
            E.g. this is the backbone attribute of a trained model with SwAV.
        transform : `torchvision.transforms`
            Must be a torchvision transform.
        cache : bool, default=False
            Cache features during training.
        overwrite : bool, default=False
            If True, overwrite items from the file given by `dir`/`filename`.
        skip_if_exists : bool, default=True
            If True, does not write new items to h5 file if it already exists,
            but returns file path.
        """
        filepath = compile_features(
            model=model,
            dataset=dataset,
            dir_name=dir_name,
            filename=filename,
            dsetname_format=dsetname_format,
            overwrite=overwrite,
            skip_if_exists=skip_if_exists,
        )

        h5_dataset = cls(
            path=filepath,
            num_classes=num_classes,
            metadata_keys=list(dataset.get_metadata(0).keys()),
            cache=cache,
            transform=transform,
            load_encoded=False,
        )

        return h5_dataset

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
        metadata = {key: dataset.attrs[key] for key in self.metadata_keys}

        data = dataset[()]

        if self.transform:
            data = self.transform(data)
        else:
            data = torch.Tensor(data)

        target = dataset.attrs["target"]
        target = torch.Tensor(target).long()  # one_hot needs integers.
        target = nn.functional.one_hot(target, num_classes=self.num_classes)

        data_obj = dict(
            data=data,
            target=target,
            tile_mpp=metadata["tile_mpp"],
            tile_x=metadata["tile_x"],
            tile_y=metadata["tile_y"],
        )

        return data_obj

    def __len__(self):
        """Calculate length of h5 dataset."""
        return len(self.dataset_indices)
