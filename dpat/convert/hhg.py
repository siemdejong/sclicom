import pathlib

from dpat.convert import AvailableImageFormats, batch_convert


def hhg_batch_convert(
    input_dir: str,
    output_dir: str,
    output_ext: AvailableImageFormats,
    trust: bool,
    skip_existing: bool,
    num_workers: int,
    chunks: int,
):
    ROOT_DIR = input_dir
    OUTPUT_DIR = output_dir
    OUTPUT_EXT = output_ext
    TRUST_SOURCE = trust
    SKIP_EXISTING = skip_existing
    NUM_WORKERS = num_workers
    CHUNKS = chunks

    paths = []
    kwargs_per_path = []
    output_dirs = []
    for scan_program in ["200slow", "300slow", "300fast"]:
        if scan_program == "200slow":
            resolution_unit: int = 3
            x_resolution: float = 5e4
            y_resolution: float = 5e4
        elif scan_program == "300slow":
            resolution_unit: int = 3
            x_resolution: float = 4e4
            y_resolution: float = 4e4
        elif scan_program == "300fast":
            resolution_unit: int = 3
            x_resolution: float = 1e4
            y_resolution: float = 1e4

        kwargs = dict(
            resolution_unit=resolution_unit,
            x_resolution=x_resolution,
            y_resolution=y_resolution,
        )
        add_paths = list(pathlib.Path(ROOT_DIR).glob(f"**/*{scan_program}*.bmp"))
        paths += add_paths
        output_dirs += [
            pathlib.Path(OUTPUT_DIR) / path.relative_to(ROOT_DIR).parent
            for path in add_paths
        ]
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