import logging
import pathlib
import signal
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Literal, Optional, TypedDict

from PIL import Image
from tqdm import tqdm

from dpat.exceptions import DpatDecompressionBombError

logger = logging.getLogger(__name__)


class AvailableImageFormats(Enum):
    tiff = "tiff"
    tif = "tif"


class ToTIFFParams(TypedDict):
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
    kwargs_per_path: Optional[list[ToOtherParams]] = None,
):
    filtered_input_paths = []
    filtered_output_dirs = []
    filtered_kwargs_per_path = []

    for input_path, output_dir, kwargs in zip(
        input_paths, output_dirs, kwargs_per_path
    ):
        input_path: pathlib.Path
        output_dir: pathlib.Path
        output_fn: pathlib.Path = pathlib.Path(
            output_dir / (input_path.stem + f".{extension}")
        )
        if not output_fn.exists():
            filtered_input_paths.append(input_path)
            filtered_output_dirs.append(output_dir)
            filtered_kwargs_per_path.append(kwargs)

    skip_count = len(input_paths) - len(filtered_input_paths)
    logger.info(
        f"Skipping {skip_count}/{len(input_paths)} images, as they were already converted to {extension}. "
        "Remove --skip-existing to overwrite existing images."
    )

    return filtered_input_paths, filtered_output_dirs, filtered_kwargs_per_path


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
                        zip(input_paths, output_dirs, kwargs_per_path),
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
