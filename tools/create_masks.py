"""Create binary tissue masks."""

from pathlib import Path
from typing import Union

import click
import numpy as np
from dlup import SlideImage
from dlup.background import AvailableMaskFunctions, get_mask
from PIL import Image
from tqdm import tqdm


@click.command()
@click.argument("path", type=Path)  # , help="File to convert to mask.")
@click.argument("output_dir", type=Path)  # , help="Output directory.")
@click.option(
    "--mask-func",
    default="entropy_masker",
    show_default=True,
    help="Mask function provided by dlup.",
)
@click.option(
    "--ext",
    show_default=None,
    help="Extension to filter on. If None, all files are considered.",
)
@click.option(
    "--skip-if-exists",
    is_flag=True,
    default=True,
    show_default=True,
    help="Skip existing masks.",
)
def create_masks(
    path: Path,
    output_dir: Path,
    mask_func: str,
    ext: Union[str, None] = None,
    skip_if_exists: bool = True,
) -> None:
    """Create a binary mask of the given image.

    Parameters
    ----------
    path : Path
        Path to the input. Can be a directory or single file.
    output_dir : Path
        Directory where to store the binary mask.
    mask_func : str
        Mask function of dlup to use.
        See dlup.background.AvailableMaskFunctions for available mask functions.
    ext : str, default=None
        Extension to filter images by. If None, considers all files.
    skip_if_exists : bool
        Skip existing masks."
    """
    if path.is_dir():
        pattern = f"*.{ext}" if ext is not None else "*"
        file_paths = list(path.glob(pattern))
    else:
        file_paths = [path]

    output_dir.mkdir(parents=True, exist_ok=True)

    mask_func = AvailableMaskFunctions[mask_func]

    for file_path in tqdm(file_paths):
        if file_path.is_dir():
            continue
        output_file_path = output_dir / (file_path.stem + "-mask.png")
        if output_file_path.exists() and skip_if_exists:
            continue

        image = SlideImage.from_file_path(file_path)

        mask = get_mask(slide=image, mask_func=mask_func)
        Image.fromarray(mask.astype(dtype=np.bool_)).save(output_file_path)


if __name__ == "__main__":
    create_masks()
