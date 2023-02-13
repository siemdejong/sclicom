import argparse
from dpat.convert import AvailableImageFormats, batch_convert
import pathlib

class BulkConvertArguments(argparse.Namespace):
    input_dir: str
    output_dir: str
    output_ext: str
    trust: bool
    skip_existing: bool

def bulk_convert(args: BulkConvertArguments):

    ROOT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    OUTPUT_EXT = args.output_ext
    TRUST_SOURCE = args.trust
    SKIP_EXISTING = args.skip_existing

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
        output_dirs += [pathlib.Path(OUTPUT_DIR) / path.relative_to(ROOT_DIR).parent for path in add_paths]
        kwargs_per_path += [kwargs] * len(add_paths)

    batch_convert(
        input_paths=paths,
        output_dirs=output_dirs,
        output_ext=OUTPUT_EXT,
        kwargs_per_path=kwargs_per_path,
        trust_source=TRUST_SOURCE,
        skip_existing=SKIP_EXISTING,
    )

def register_parser(parser: argparse._SubParsersAction):
    """Register hhg commands to a root parser."""

    # Convert HHG images from any image to another format.
    convert_parser: argparse.ArgumentParser = parser.add_parser("convert", help="Convert HHG images from any image to another format.")
    convert_subparsers = convert_parser.add_subparsers(help="Convert subparser.")
    convert_subparsers.required = True
    convert_subparsers.dest = "subcommand"

    bulk_convert_parser = convert_subparsers.add_parser("bulk", help="Convert images in bulk.")
    bulk_convert_parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Input directory where to find the images to be converted.",
        default='.',
    )
    bulk_convert_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory where place converted files.",
        default="./converted",
    )
    bulk_convert_parser.add_argument(
        "--output-ext",
        "-e",
        type=str,
        choices=AvailableImageFormats.__members__,
        required=True,
        help="Extension to convert to."
    )
    bulk_convert_parser.add_argument(
        "--trust",
        "-t",
        type=bool,
        default=False,
        help="Trust the source of the images."
    )
    bulk_convert_parser.add_argument(
        "--skip-existing",
        "-s",
        type=bool,
        default=True,
        help="Skip existing output files."
    )
    bulk_convert_parser.set_defaults(subcommand=bulk_convert)