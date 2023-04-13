"""Utilities for configuration."""

from pathlib import Path


def get_default_config_by_name(name: str) -> str:
    """Return the absolute path from a path relative to the invoking script."""
    absolute_path = str(
        (Path(__file__).parent / f"../../dpat/configs/defaults/{name}.yaml").resolve()
    )
    return absolute_path
