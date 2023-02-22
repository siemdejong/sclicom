"""Provide transforms."""
from enum import Enum


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
        sample["image"] = sample["image"].convert("RGB")
        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])
        return sample


class AvailableTransforms(Enum):
    """Available transforms."""

    dlup2dpat = "dlup2dpat"
