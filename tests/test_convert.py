"""Tests for the convert package."""

import pytest
from PIL import Image

from dpat.convert import img_to_tiff

from .common import image_files


@pytest.fixture
def tiff_kwargs():
    """Create tiff metadata."""
    kwargs = {"resolution_unit": 3, "x_resolution": 5e4, "y_resolution": 5e4}
    return kwargs


class TestConvert:
    """Tests for the convert package."""

    @pytest.mark.parametrize("extension", ["tiff", "tif"])
    def test_img_to_tiff(self, image_files, tmp_path, extension, tiff_kwargs):
        """Test if image with new filename is created."""
        img_to_tiff(image_files[0], tmp_path, extension, **tiff_kwargs)

        Image.open(tmp_path / (image_files[0].stem + f".{extension}"))

    def test_large_convert(self, image_files, tmp_path, tiff_kwargs):
        """Test if a large image warns about a decompression bomb."""
        Image.MAX_IMAGE_PIXELS = 50
        with pytest.warns(Image.DecompressionBombWarning):
            img_to_tiff(image_files[0], tmp_path, "tiff", **tiff_kwargs)
        img_to_tiff(image_files[0], tmp_path, "tiff", trust_source=True, **tiff_kwargs)

    def test_bomb_convert(self, image_files, tmp_path, tiff_kwargs):
        """Test if a super large image raises a decompression bomb warning."""
        Image.MAX_IMAGE_PIXELS = 1
        with pytest.raises(Image.DecompressionBombError):
            img_to_tiff(image_files[0], tmp_path, "tiff", **tiff_kwargs)
        img_to_tiff(image_files[0], tmp_path, "tiff", trust_source=True, **tiff_kwargs)
