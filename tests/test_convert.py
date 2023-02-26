"""Tests for the convert package."""

import pathlib

import numpy as np
import pytest
from PIL import Image

from dpat.convert import img_to_tiff


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
def image_file(tmp_path_factory) -> pathlib.Path:
    """Create an image file."""
    img = create_img((10, 10, 3))
    fn = tmp_path_factory.mktemp("data") / "img.bmp"
    img.save(fn)
    return fn


@pytest.fixture
def tiff_kwargs():
    """Create tiff metadata."""
    kwargs = {"resolution_unit": 3, "x_resolution": 5e4, "y_resolution": 5e4}
    return kwargs


class TestConvert:
    """Tests for the convert package."""

    @pytest.mark.parametrize("extension", ["tiff", "tif"])
    def test_img_to_tiff(self, image_file, tmp_path, extension, tiff_kwargs):
        """Test if image with new filename is created."""
        img_to_tiff(image_file, tmp_path, extension, **tiff_kwargs)

        Image.open(tmp_path / (image_file.stem + f".{extension}"))

    def test_large_convert(self, image_file, tmp_path, tiff_kwargs):
        """Test if a large image warns about a decompression bomb."""
        Image.MAX_IMAGE_PIXELS = 50
        with pytest.warns(Image.DecompressionBombWarning):
            img_to_tiff(image_file, tmp_path, "tiff", **tiff_kwargs)
        img_to_tiff(image_file, tmp_path, "tiff", trust_source=True, **tiff_kwargs)

    def test_bomb_convert(self, image_file, tmp_path, tiff_kwargs):
        """Test if a super large image raises a decompression bomb warning."""
        Image.MAX_IMAGE_PIXELS = 1
        with pytest.raises(Image.DecompressionBombError):
            img_to_tiff(image_file, tmp_path, "tiff", **tiff_kwargs)
        img_to_tiff(image_file, tmp_path, "tiff", trust_source=True, **tiff_kwargs)
