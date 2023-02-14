import argparse

from dpat.splits.create_splits import create_splits


class CreateSplitsArguments(argparse.Namespace):
    input_dir: str
    output_dir: str
    path_to_labels_file: str
    dataset_name: str
    filter_diagnosis: list[str]


def cli_create_splits(args: CreateSplitsArguments) -> None:
    input_dir = args.input_dir
    output_dir = args.output_dir
    path_to_labels_file = args.path_to_labels_file
    dataset_name = args.dataset_name
    filter_diagnosis = args.filter_diagnosis
    create_splits(
        input_dir, path_to_labels_file, dataset_name, output_dir, filter_diagnosis
    )


def register_parser(parser: argparse._SubParsersAction):
    """Register splits commands to a root parser."""

    # Create splits
    splits_parser: argparse.ArgumentParser = parser.add_parser(
        "splits", help="Create data splits."
    )
    splits_subparsers = splits_parser.add_subparsers(help="Splits subparser.")
    splits_subparsers.required = True
    splits_subparsers.dest = "subcommand"

    create_splits_parser = splits_subparsers.add_parser(
        "create", help="Create data splits."
    )
    create_splits_parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Input directory where to find the images.",
        default=".",
    )
    create_splits_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Name of dataset.",
        default="splits",
    )
    create_splits_parser.add_argument(
        "--labels",
        "-l",
        type=str,
        required=True,
        help="Path to labels file.",
        default="labels.csv",
        dest="path_to_labels_file",
    )
    create_splits_parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="Name of dataset.",
        dest="dataset_name",
    )
    create_splits_parser.add_argument(
        "--filter",
        "-f",
        type=str,
        help="Filter a diagnosis. For multiple diagnoses, use `-f 1 -f 2`.",
        dest="filter_diagnosis",
        action="append",
    )
    create_splits_parser.set_defaults(subcommand=cli_create_splits)
