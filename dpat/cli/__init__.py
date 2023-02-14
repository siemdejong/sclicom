"""DPAT Command-line interface. This is the file which builds the main parser."""
import argparse


def main():
    """
    Console script for dlup.
    """
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    root_subparsers = root_parser.add_subparsers(help="Possible DPat CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # Prevent circular import
    from dpat.cli.convert import register_parser as register_convert_subcommand
    from dpat.cli.create_splits import register_parser as register_splits_subcommand

    # Whole slide images related commands.
    register_convert_subcommand(root_subparsers)
    register_splits_subcommand(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main()
