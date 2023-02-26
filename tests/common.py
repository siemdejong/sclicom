"""Provide utilities for tests."""
import pathlib

import numpy as np
import pytest
from PIL import Image


def create_img(shape: tuple[int, int, int]) -> Image.Image:
    """Create a random image.

    Parameters
    ----------
    shape : tuple of three ints
        Shape of the output image.
    """
    img = np.random.rand(*shape) * 255
    img = Image.fromarray(img.astype("uint8")).convert("RGB")
    return img


@pytest.fixture(scope="session")
def image_files(tmp_path_factory) -> list[pathlib.Path]:
    """Create an image file."""
    img = create_img((10, 10, 3))
    filenames = []
    tmp_dir = tmp_path_factory.mktemp("data")
    for num in range(100):
        fn = tmp_dir / f"img-{num}.bmp"
        img.save(fn)
        filenames.append(fn)

    return filenames
