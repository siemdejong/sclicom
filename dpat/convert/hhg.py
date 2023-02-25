"""Convert functions related to the PMC-HHG project."""

import pathlib
from typing import Literal

from dpat.convert import (
    AvailableImageFormats,
    ToOtherParams,
    ToTIFFParams,
    batch_convert,
)


def hhg_batch_convert(
    input_dir: str,
    output_dir: str,
    output_ext: AvailableImageFormats,
    trust: bool,
    skip_existing: bool,
    num_workers: int,
    chunks: int,
):
    """Extract metadata from filenames used in the PMC-HHG project.

    Images with
    - 200slow are 0.2 mpp
    - 300slow are 0.25 mpp
    - 300fast are 1 mpp

    As tiff requires a resolution unit of cm, x- and y-resolutions are calculated as
    1e4 um / x mmp, where x are the values above.
    """
    ROOT_DIR = input_dir
    OUTPUT_DIR = output_dir
    OUTPUT_EXT = output_ext
    TRUST_SOURCE = trust
    SKIP_EXISTING = skip_existing
    NUM_WORKERS = num_workers
    CHUNKS = chunks

    resolution_unit: Literal[1, 2, 3]
    x_resolution: float
    y_resolution: float

    paths = []
    kwargs_per_path: list[ToOtherParams] = []
    output_dirs = []
    for scan_program in ["200slow", "300slow", "300fast"]:
        if scan_program == "200slow":
            resolution_unit = 3
            x_resolution = 5e4
            y_resolution = 5e4
        elif scan_program == "300slow":
            resolution_unit = 3
            x_resolution = 4e4
            y_resolution = 4e4
        elif scan_program == "300fast":
            resolution_unit = 3
            x_resolution = 1e4
            y_resolution = 1e4

        add_paths = list(pathlib.Path(ROOT_DIR).glob(f"**/*{scan_program}*.bmp"))
        paths += add_paths
        output_dirs += [
            pathlib.Path(OUTPUT_DIR) / path.relative_to(ROOT_DIR).parent
            for path in add_paths
        ]

        if output_ext in ["tiff", "tif"]:
            kwargs: ToTIFFParams = dict(
                resolution_unit=resolution_unit,
                x_resolution=x_resolution,
                y_resolution=y_resolution,
            )
            kwargs_per_path += [kwargs] * len(add_paths)

    batch_convert(
        input_paths=paths,
        output_dirs=output_dirs,
        output_ext=OUTPUT_EXT,
        kwargs_per_path=kwargs_per_path,
        trust_source=TRUST_SOURCE,
        skip_existing=SKIP_EXISTING,
        num_workers=NUM_WORKERS,
        chunks=CHUNKS,
    )
