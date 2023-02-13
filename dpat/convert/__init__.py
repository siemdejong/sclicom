from PIL import Image
from typing import Literal, TypedDict, Optional
import pathlib
from tqdm import tqdm
from functools import partial
from enum import Enum
from multiprocessing import Pool


class AvailableImageFormats(Enum):
    tiff = "tiff"
    tif = "tif"


class ToTIFFParams(TypedDict):
    """Allowed parameters for `img_to_tiff`.

    Attributes
    ----------
    resolution_unit : int
        Resolution unit, indicating no (1), inch (2) or cm (3) as unit.
    xresolution : float
        Number of pixels per resolution unit in X-direction.
    yresolution : float
        Number of pixels per resolution unit in Y-direction.
    """

    resolution_unit: Literal[1, 2, 3]
    x_resolution: float
    y_resolution: float


class ToOtherParams(ToTIFFParams):
    pass


def img_to_tiff(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    extension: AvailableImageFormats,
    trust_source: bool = False,
    **kwargs: ToTIFFParams,
) -> None:
    """Convert any image to TIFF.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the input image.
    output_dir : pathlib.Path
        Directory to output image in.
    extension : str
        Output file extension.
    trust_source : bool
        Trust source. If True, PIL.Image.DecompressionBombError is ignored.
    kwargs : `ToTIFFParams`
        Extra arguments to `PIL.Image.save`: refer to the `ToTIFFParams` documentation for a list of all possible arguments.
    """
    # assert not input_path.endswith([".tiff", ".tif"]), "Input image is already in TIFF format."

    # To ignore decompression bomb DOS attack error.
    if trust_source:
        print("Source is trusted. PIL.Image.DecompressionBombError ignored.")
        Image.MAX_IMAGE_PIXELS = None

    # Skip existing files.
    output_fn = pathlib.Path(output_dir / (input_path.stem + f".{extension}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    bmp_image = Image.open(input_path)
    bmp_image.save(output_fn, **kwargs)
    bmp_image.close()


def filter_existing(
    input_paths: list[pathlib.Path],
    output_dirs: list[pathlib.Path],
    extension: AvailableImageFormats,
    kwargs_per_path: Optional[list[ToOtherParams]] = None,
):
    skip_count = 0
    for i, (input_path, output_dir) in enumerate(zip(input_paths, output_dirs)):
        output_fn = pathlib.Path(output_dir / (input_path.stem + f".{extension}"))
        if output_fn.exists():
            input_paths.pop(i)
            output_dirs.pop(i)
            kwargs_per_path.pop(i)
            skip_count += 1

    print(f"Skipping {skip_count} images, as they already exist.")
    print("Use '--skip-existing false' to not overwrite existing images.")

    return input_paths, output_dirs, kwargs_per_path


def _wrapper(args, worker):
    input_path, output_dir, kwargs = args
    worker(input_path, output_dir, **kwargs)


def batch_convert(
    input_paths: list[pathlib.Path],
    output_dirs: list[pathlib.Path],
    output_ext: AvailableImageFormats,
    num_workers: int = 4,
    chunks: int = 30,
    kwargs_per_path: Optional[list[ToOtherParams]] = None,
    trust_source=False,
    skip_existing=True,
) -> None:
    """Convert all images found in the input directory.

    Parameters
    ----------
    input_paths : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.
    output_ext : str
        Output extension
    trust_source : bool
        Trust source. If True, PIL.Image.DecompressionBombError is ignored.
    skip_existing : bool
        Skip existing output files.
    kwargs : `ToOtherParams`
        Extra arguments to `convert_func`.
    """

    if output_ext in ["tiff", "tif"]:
        convert_func = partial(
            img_to_tiff,
            extension=output_ext,
            skip_existing=skip_existing,
            trust_source=trust_source,
        )

    if skip_existing:
        input_paths, output_dirs, kwargs_per_path = filter_existing(
            input_paths,
            output_dirs,
            output_ext,
            kwargs_per_path,
        )

    wrapper = partial(_wrapper, worker=convert_func)

    nmax = len(input_paths)
    chunksize = max(nmax // chunks, 1)
    with Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    wrapper,
                    zip(input_paths, output_dirs, kwargs_per_path),
                    chunksize,
                ),
                total=nmax,
                desc="Converting images",
            )
        )
