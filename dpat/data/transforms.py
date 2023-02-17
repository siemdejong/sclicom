"""Provide transforms."""
from enum import Enum
from typing import Any


class Dlup2DpatTransform:
    """Transform DLUP dataset objects for downstream use.

    Essentially, it ensures the image object is of type PIL.Image.Image,
    and ensures that the paths are strings.
    """

    def __init__(self, transform):
        """Create transform.

        Parameters
        ----------
        transform :
            transform to apply.
        """
        self.transform = transform

    def __call__(self, sample):
        """Do transform."""
        # torch collate functions can not handle a pathlib.path object
        # and want a string instead
        sample["path"] = str(sample["path"])
        # Openslide returns RGBA, but most neural networks want RGB
        sample["image"] = self.transform(sample["image"].convert("RGB"))
        return sample


class ContrastiveTransform:
    """Contrastive transform for use with SimCLR."""

    def __init__(self, base_transforms, n_views=2) -> None:
        """Create transform."""
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x: Any) -> list[Any]:
        """Do transform.

        Apply the base transforms n times and put the transformed objects in a list.

        Parameters
        ----------
        x : Any
            parameter to apply the base transforms to.
        """
        return [self.base_transforms(x) for _ in range(self.n_views)]


class AvailableTransforms(Enum):
    """Available transforms."""

    dlup2dpat = "dlup2dpat"
    contrastive = "contrastive"
