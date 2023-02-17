"""DPAT Command-line interface.

This is the file which builds the main parser.
"""
import click

from dpat.cli.logging import config_logging
from dpat.convert import AvailableImageFormats
from dpat.convert.hhg import hhg_batch_convert
from dpat.splits.create_splits import create_splits

config_logging()


@click.group()
def cli():
    """Click group to attach all cli subcommands to."""
    pass


@cli.group()
def convert():
    """Click group to attach all convert commands to."""
    pass


@convert.command()
@click.option(
    "-i",
    "--input-dir",
    default=".",
    show_default=True,
    help="Input directory where to find the images to be converted.",
)
@click.option(
    "-o",
    "--output-dir",
    default="./converted",
    show_default=True,
    help="Output directory where place converted files.",
)
@click.option(
    "-e",
    "--output-ext",
    type=click.Choice(AvailableImageFormats.__members__, case_sensitive=False),
    required=True,
    help="Extension to convert to.",
)
@click.option(
    "-w",
    "--num-workers",
    default=4,
    show_default=True,
    help="Number of workers that convert the images in parallel.",
)
@click.option(
    "-c",
    "--chunks",
    default=30,
    show_default=True,
    help="Number of chunks distributed to every worker.",
)
@click.option(
    "--trust",
    is_flag=True,
    default=False,
    show_default=True,
    help="Trust the source of the images.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    show_default=True,
    help="Skip existing output files.",
)
def batch(*args, **kwargs):
    """Click command passing args to the batch converter."""
    hhg_batch_convert(*args, **kwargs)


@cli.group()
def splits():
    """Click group to attach all splits commands to."""
    pass


@splits.command()
@click.option(
    "-i",
    "--input-dir",
    "image_dir",
    required=True,
    show_default=True,
    help="Input directory where to find the images.",
)
@click.option(
    "-l", "--labels", "path_to_labels_file", required=True, help="Path to labels file."
)
@click.option("-n", "--name", "dataset_name", required=True, help="Name of dataset.")
@click.option(
    "-o",
    "--output-dir",
    "save_to_dir",
    default="splits",
    show_default=True,
    help="Directory where to put the splits.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite folds in output dir, if available.",
)
@click.option(
    "-y",
    "--include",
    "include_pattern",
    default=["*.*"],
    show_default=True,
    multiple=True,
    help="Glob pattern to include files from `input-dir`",
)
@click.option(
    "-x",
    "--exclude",
    "exclude_pattern",
    default=[""],
    show_default=True,
    multiple=True,
    help="Glob pattern to exclue files from `input-dir`, included with `--include`",
)
@click.option(
    "-f",
    "--filter",
    "filter_diagnosis",
    default=None,
    show_default=True,
    multiple=True,
    help="Filter a diagnosis. For multiple diagnoses, use `-f 1 -f 2`.",
)
def create(*args, **kwargs):
    """Click command passing args to the split creator."""
    create_splits(*args, **kwargs)
