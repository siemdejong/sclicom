"""Type stubs."""

from typing import Any, Protocol, Sized

from torch.utils.data import Dataset


class SizedDataset(Sized, Dataset):
    """Type stub for a dataset where __len__ is implemented."""

    pass


class MetaDataProtocol(Protocol):
    """Protocol for classes that implement `get_metadata`."""

    def get_metadata(self, index: int) -> dict[str, Any]:
        """Get metadata at index of `Dataset`."""
        ...


class MetaDataDataset(MetaDataProtocol, Dataset):
    """Type stub for a dataset where `get_metadata` is implemented."""

    pass


class SizedMetaDataDataset(SizedDataset, MetaDataDataset):
    """Type stub for a dataset with size and metadata."""

    pass
