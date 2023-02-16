from enum import Enum


class Dlup2DpatTransform:
    """
    A small class to transform the objects returned by a DLUP dataset to the expected
    object by DPAT. Essentially, it ensures the image object is of type PIL.Image.Image,
    and ensures that the paths are strings.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        # torch collate functions can not handle a pathlib.path object
        # and want a string instead
        sample["path"] = str(sample["path"])
        # Openslide returns RGBA, but most neural networks want RGB
        sample["image"] = self.transform(sample["image"].convert("RGB"))
        return sample


class ContrastiveTransform:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class AvailableTransforms(Enum):
    dlup2dpat = "dlup2dpat"
    contrastive = "contrastive"
