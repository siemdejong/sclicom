"""Type stubs."""

from torch.utils.data import Dataset


class SizedDataset(Dataset):
    """Type stub for a dataset where __len__ is implemented."""

    def __len__(self):
        """Calculate length of `Dataset`."""
        ...


class SizedMetaDataDataset(SizedDataset):
    """Type stub for a dataset where `get_metadata` is implemented."""

    def get_metadata(self, index: int):
        """Get metadata at index of `Dataset`."""
        ...
