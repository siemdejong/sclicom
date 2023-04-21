"""Provide H5 dataset to read and compile features."""
import logging
import pathlib
import signal
import sys
from collections import Counter
from io import BytesIO
from pprint import pformat
from typing import Generator, Type, TypeVar, Union

import h5py
import lightning.pytorch as pl
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from dpat.data import PMCHHGImageDataset

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

        with h5py.File(self.file_path) as file:
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


class CarefulHDF5:
    """Open an HDF5 file carefully.

    If file already exists, note to give the overwrite keyword. If
    interrupted while open, flush buffers and close the file.
    """

    def __init__(self, *args, **kwargs):
        """Save args and kwargs for the h5py File object initialization."""
        self.killed = False
        self.args = args
        self.kwargs = kwargs

    def __exit__(self, *args, **kwargs):
        """Gracefully close the h5py file."""
        try:
            self.f.flush()
            self.f.close()
        except ValueError:
            logger.error(
                "Something went wrong with closing the file. "
                "Please inspect the file and delete if broken."
            )
            exit()

        if self.killed:
            sys.exit(0)

        signal.signal(signal.SIGINT, self.old_sigint)
        signal.signal(signal.SIGTERM, self.old_sigterm)

    def _handler(self, *args, **kwargs):
        """Kill switch."""
        logging.error("Received SIGINT or SIGTERM! Finishing this block, then exiting.")
        self.killed = True
        self.__exit__()

    def __enter__(self, *args, **kwargs):
        """Start listening for SIGINT/SIGTERM and safely open h5 file file."""
        self.old_sigint = signal.signal(signal.SIGINT, self._handler)
        self.old_sigterm = signal.signal(signal.SIGTERM, self._handler)

        try:
            self.f = h5py.File(*self.args, **self.kwargs)
            return self.f
        except FileExistsError:
            logger.error(
                "Preventing overwrite. Use the overwrite keyword to overwrite features."
            )
            exit()
        except KeyboardInterrupt:
            logger.info("Interrupted...")
            exit()


def generate_embeddings(model, dataloader, desc: str = "Extracting features"):
    """Generate vector reps of tiles."""
    embeddings = []
    with torch.no_grad():
        for img in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            img = img["image"]
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)

    embeddings_concatenated = torch.cat(embeddings, 0)
    return embeddings_concatenated


def feature_batch_extract(
    f: h5py.File,
    model: nn.Module,
    dataset: PMCHHGImageDataset,
    batch_size: int,
    dsetname_format: list[str],
    num_workers: int = 0,
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
    batch_size : int
        Batch size to use to feed data to the model.
    dsetname_format : list[str]
        List of strings which determine where the individual items of the dataset
        will be stored in the hierarchical h5 structure.
        The *i*th H5 datasets in the hdf5 file is created at the location which is
        computed like `dataset.get_metadata(i)[dsetname_format...]` where
        `dsetname_format` is a list of strings choosing the metadata to be used to
        name the hierarchical groups and datasets.
    num_workers : int, default=0
        Number of workers preparing the data for the GPU.
    skip_if_exists : bool, default=True
        Skip if dataset objects in hdf5 file already exist.
    """
    # TODO: extract features in batches. It currently takes a long time.
    model.eval()

    # PMCHHGDataset.dlup_dataset.datasets contain tiles belonging to one image.
    all_img_dataset = dataset.dlup_dataset.datasets

    length_all_img_datasets = len(all_img_dataset)
    for img_dataset in tqdm(
        all_img_dataset, total=length_all_img_datasets, desc="Image."
    ):
        # Check if we can skip this image, because it is already in the dataset.
        if skip_if_exists:
            metadata = dataset.get_metadata(0, img_dataset)
            dsetname = "".join(
                [f"/{metadata[dsetname]}" for dsetname in dsetname_format]
            )

            if dsetname in f:
                continue

        tile_metadata = []
        for i in range(len(img_dataset)):
            metadata = dataset.get_metadata(i, img_dataset)
            tile_metadata.append(metadata)

        dsetname = "".join([f"/{metadata[dsetname]}" for dsetname in dsetname_format])

        dataloader = DataLoader(
            img_dataset, batch_size=batch_size, num_workers=num_workers
        )
        embeddings = generate_embeddings(model, dataloader, desc="Tile batches")

        img_group_h5 = f.create_group(dsetname)
        img_group_h5.create_dataset("data", data=embeddings.cpu())
        img_group_h5.create_dataset(
            "all_tile_mpp", data=[tile["meta"]["tile_mpp"] for tile in tile_metadata]
        )
        img_group_h5.create_dataset(
            "all_tile_region_index",
            data=[tile["tile_region_index"] for tile in tile_metadata],
        )
        img_group_h5.create_dataset(
            "all_tile_x", data=[tile["meta"]["tile_x"] for tile in tile_metadata]
        )
        img_group_h5.create_dataset(
            "all_tile_y", data=[tile["meta"]["tile_y"] for tile in tile_metadata]
        )
        img_group_h5.create_dataset(
            "all_target", data=[tile["meta"]["target"] for tile in tile_metadata]
        )
        img_group_h5.create_dataset(
            "all_location",
            data=[tile["meta"]["location"] for tile in tile_metadata],
            dtype=h5py.string_dtype("utf-8"),
        )


def compile_features(
    model: nn.Module,
    dataset: PMCHHGImageDataset,
    dir_name: pathlib.Path,
    filename: str,
    dsetname_format: list[str],
    mode: h5py.File.mode = "a",
    skip_if_exists: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
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
    num_workers : int, default=0
        Number of workers for the dataloader.

    Returns
    -------
    filepath : `pathlib.Path`
        Path to the compiled h5 file.
    """
    dir_name.mkdir(exist_ok=True, parents=True)

    filepath = dir_name / filename

    with CarefulHDF5(name=filepath, mode=mode) as file:
        feature_batch_extract(
            file,
            model,
            dataset,
            batch_size,
            dsetname_format,
            num_workers,
            skip_if_exists,
        )

    return filepath


class PMCHHGH5Dataset(Dataset):
    """Dataset for packed HDF5 files to pass to a PyTorch dataloader."""

    def __init__(
        self,
        path: pathlib.Path,
        num_classes: int,
        metadata_keys: Union[list[str], None] = None,
        paths_and_targets: Union[pathlib.Path, str, BytesIO, None] = None,
        cache: bool = False,
        transform: Union[torchvision.transforms.Compose, None] = None,
        load_encoded: bool = False,
        clinical_context: bool = False,
    ) -> None:
        """Initialize an H5 dataset.

        Parameters
        ----------
        path : `pathlib.Path`
            Location where the images are stored on disk.
        transform : `torchvision.transforms.Compose`, default=None
            Transform to apply.
        paths_and_targets : Path|str|BytesIO
            File to be read by pandas for the img paths/case_id/img_id/targets.
        load_encoded : bool, default=False
            Whether the images within the h5 file are encoded or saved as bytes
            directly.
        clinical_context : bool, default=False
            Export clinical context when fetching items.
        """
        self.file_path = path
        self.num_classes = num_classes
        self.cache = cache
        self.transform = transform
        self.load_encoded = load_encoded
        self.paths_and_targets = paths_and_targets

        self.clinical_context = clinical_context

        self.df = pd.read_csv(paths_and_targets, header=None)
        self.case_ids = self.df[1]
        self.img_ids = self.df[2]
        self.dataset_indices_filter = [
            f"{case_id}/{img_id}"
            for case_id, img_id in zip(self.case_ids, self.img_ids)
        ]
        self.df = self.df.set_index(0)

        # Reset for if it needs to be reused somewhere else, e.g. in PMCHHGH5Dataset.
        # TODO: isinstance(..., BytesIO), but it doesn't recognize it as bytes.
        if not isinstance(self.paths_and_targets, (pathlib.Path, str)):
            self.paths_and_targets.seek(0)  # type: ignore

        self.hdf5: Union[h5py.File, None] = None

        self._h5ls = _H5ls(self.file_path, h5py.Group, 2)
        self.dataset_indices = [
            path for path in self._h5ls if path in self.dataset_indices_filter
        ]

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
        dataset: PMCHHGImageDataset,
        model: nn.Module,
        num_classes: int,
        dsetname_format: list[str],
        transform: torchvision.transforms = None,
        cache: bool = False,
        mode: h5py.File.mode = "a",
        skip_if_exists: bool = True,
        skip_feature_compilation: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        clinical_context: bool = False,
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
        dataset : `PMCHHGImageDataset`
            The dataset to extract hdf5 features from.
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
        num_workers : int, default=0
            Number of workers for the dataloader.
        clinical_context : bool, default=False
            Export clinical context when fetching items.
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
                batch_size=batch_size,
                num_workers=num_workers,
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
            paths_and_targets=dataset.image_paths_and_targets,
            cache=cache,
            transform=transform,
            load_encoded=False,
            clinical_context=clinical_context,
        )

        return h5_dataset

    def compute_sampler_weights(self) -> list[float]:
        """Compute weights to oversample the minority class.

        Pytorch's WeightedRandomSampler requires weights for every
        sample to give probabilities of every sample being drawn.
        """
        # TODO: Make weights depend on location?
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
            case_id : str
                Case identifier. E.g. "PMG_HHG_1".
            img_id : str
                Image identifier.
            cc : str
                Location. E.g. "frontal lobe".

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
            cc=metadata["all_location"].astype(str)[0],
        )

        return data_obj

    def __len__(self):
        """Calculate length of h5 dataset."""
        return len(self.dataset_indices)


class PMCHHGH5DataModule(pl.LightningDataModule):
    """H5 Datamodule for use with Pytorch Lightning and the PMC-HHG dataset."""

    def __init__(
        self,
        file_path: pathlib.Path,
        train_path: pathlib.Path,
        val_path: pathlib.Path,
        test_path: Union[pathlib.Path, None] = None,
        clinical_context: bool = False,
        num_workers: int = 0,
        num_classes: int = 2,
        balance: bool = True,
    ):
        """Create PMCHHGH5Dataset DataModule.

        Note that batch sizes are forced to be one (bag).

        Parameters
        ----------
        file_path : pathlib.Path
            Path to hdf5 file containing all needed tile embeddings.
        train_path : pathlib.Path
            Path to paths and targets for the train fold
            (see dpat.data.PMCHHGImageDataset).
        val_path : pathlib.Path
            Path to paths and targets for the val fold
            (see dpat.data.PMCHHGImageDataset).
        test_path : pathlib.Path
            Path to paths and targets for the test fold
            (see dpat.data.PMCHHGImageDataset).
        clinical_context : bool, default=False
            Export clinical context with tiles.
        num_workers : int, default=0
            Number of workers for the datalaoders.
        num_classes : int, default=2,
            Number of classes for prediction.
        balance : bool, default=True
            Balance the training set using minority oversampling.
        """
        super().__init__()
        self.file_path = file_path
        self.train_paths_and_targets = train_path
        self.val_paths_and_targets = val_path
        self.test_paths_and_targets = test_path
        self.clinical_context = clinical_context
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
            "all_location",
        ]
        dataset_kwargs = dict(
            path=self.file_path,
            num_classes=self.num_classes,
            metadata_keys=metadata_keys,
            clinical_context=self.clinical_context,
        )
        if stage == "fit":
            self.train_dataset = PMCHHGH5Dataset(
                paths_and_targets=self.train_paths_and_targets, **dataset_kwargs
            )
            self.val_dataset = PMCHHGH5Dataset(
                paths_and_targets=self.val_paths_and_targets, **dataset_kwargs
            )
        elif stage == "test":
            if self.test_paths_and_targets is None:
                raise ValueError("Please provide a path to the test paths and targets.")
            self.test_dataset = PMCHHGH5Dataset(
                paths_and_targets=self.test_paths_and_targets, **dataset_kwargs
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
