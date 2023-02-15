class Dlup2DpatTransform:
    """
    A small class to transform the objects returned by a DLUP dataset to the expected object by DPAT.
    Essentially, it ensures the image object is of type PIL.Image.Image, and ensures that the paths are strings.
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
