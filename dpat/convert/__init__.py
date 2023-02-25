"""Convert image to other formats."""
import logging
import pathlib
import signal
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Literal, TypedDict, Union

from PIL import Image
from tqdm import tqdm

from dpat.exceptions import DpatDecompressionBombError

logger = logging.getLogger(__name__)


class AvailableImageFormats(Enum):
    """Available image formats to convert to."""

    tiff = "tiff"
    tif = "tif"


class ToOtherParams(TypedDict):
    """Class encapsulating target format classes for typing."""

    pass


class ToTIFFParams(ToOtherParams):
    """Allowed parameters for `img_to_tiff`.

    TIFF files [1] contain resolution information in the header.

    Attributes
    ----------
    resolution_unit : int
        Resolution unit, indicating no (1), inch (2) or cm (3) as unit.
    xresolution : float
        Number of pixels per resolution unit in X-direction.
    yresolution : float
        Number of pixels per resolution unit in Y-direction.

    References
    ----------
    [1] https://www.fileformat.info/format/tiff/corion.htm
    """

    resolution_unit: Literal[1, 2, 3]
    x_resolution: float
    y_resolution: float


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
        Extra arguments to `PIL.Image.save`: refer to the `ToTIFFParams` documentation
        for a list of all possible arguments.
    """
    # To ignore decompression bomb DOS attack error.
    if trust_source:
        Image.MAX_IMAGE_PIXELS = None

    bmp_image = Image.open(input_path)
    output_fn = pathlib.Path(output_dir / (input_path.stem + f".{extension}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    bmp_image.save(output_fn, **kwargs)
    bmp_image.close()


def filter_existing(
    input_paths: list[pathlib.Path],
    output_dirs: list[pathlib.Path],
    extension: AvailableImageFormats,
    kwargs_per_path: Union[list[ToOtherParams], None] = None,
) -> tuple[list[pathlib.Path], list[pathlib.Path], Union[list[ToOtherParams], None]]:
    """Filter existing output paths.

    Pops out items of all input lists that correspond to an existing output file.

    Parameters
    ----------
    input_paths : list of pathlib.Path
        Paths to images.
    output_dirs : list of pathlib.Path
        Output directories.
    extension : AvailableImageFormats
        Extension to format the file to.
    kwargs_per_path : list of ToOtherParams or None, optional
        Any other list of kwargs to be filtered with the other lists.
    """
    filter_ids: list[int] = []
    for i, (input_path, output_dir) in enumerate(zip(input_paths, output_dirs)):
        output_fn = output_dir / (input_path.stem + f".{extension}")
        if output_fn.exists():
            filter_ids.append(i)

    filtered_input_paths = [
        input_path for i, input_path in enumerate(input_paths) if i not in filter_ids
    ]
    filtered_output_dirs = [
        output_dir for i, output_dir in enumerate(output_dirs) if i not in filter_ids
    ]

    if kwargs_per_path is not None:
        filtered_kwargs_per_path = [
            kwargs for i, kwargs in enumerate(kwargs_per_path) if i not in filter_ids
        ]
    else:
        filtered_kwargs_per_path = None

    skip_count = len(input_paths) - len(filtered_input_paths)
    if skip_count:
        logger.info(
            f"Skipping {skip_count}/{len(input_paths)} images, as they were already"
            f" converted to {extension}. Remove --skip-existing to overwrite existing"
            " images."
        )

    return filtered_input_paths, filtered_output_dirs, filtered_kwargs_per_path


def _wrapper(args, worker):
    """Pass kwargs to Pool.imap_unordered."""
    try:
        input_path, output_dir, kwargs = args
        worker(input_path, output_dir, **kwargs)
    except ValueError:
        worker(*args)


def batch_convert(
    input_paths: list[pathlib.Path],
    output_dirs: list[pathlib.Path],
    output_ext: AvailableImageFormats,
    num_workers: int = 4,
    chunks: int = 30,
    kwargs_per_path: Union[list[ToOtherParams], None] = None,
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
            input_paths, output_dirs, output_ext, kwargs_per_path
        )

    if not input_paths:
        logger.info("No images to convert.")
        return

    wrapper = partial(_wrapper, worker=convert_func)

    nmax = len(input_paths)
    chunksize = max(nmax // chunks, 1)
    with Pool(
        num_workers, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)
    ) as pool:
        try:
            list(
                tqdm(
                    pool.imap_unordered(
                        wrapper,
                        zip(input_paths, output_dirs, kwargs_per_path)
                        if kwargs_per_path
                        else zip(input_paths, output_dirs),
                        chunksize,
                    ),
                    total=nmax,
                    desc="Converting images",
                )
            )
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        except Image.DecompressionBombError:
            raise DpatDecompressionBombError
