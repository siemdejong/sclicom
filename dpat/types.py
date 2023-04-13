"""Type stubs."""

from typing import Any, Literal, Protocol, Sized, Union

import torch
from torch import nn
from torch.utils.data import Dataset

MaskFactory = Literal["no_mask", "load_from_disk", "entropy_masker"]

ExampleInputArray = Union[Union[torch.Tensor, tuple, dict], None]


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


class OutputProtocol(Protocol):
    """Protocol for modules that implement a Linear layer."""

    output: nn.Linear


class VisionBackbone(nn.Module, OutputProtocol):
    """Type stub for a vision model with the fc.in_features attribute."""

    pass


class LLMOutputProtocol(Protocol):
    """Protocol for Large Language Model outputs."""

    last_hidden_state: torch.Tensor


class LLMOutput(nn.Module, LLMOutputProtocol):
    """Type stub for a LLM output with last_hidden_states attribute."""

    pass
