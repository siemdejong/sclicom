"""Provides logging configurations for the CLI."""
import logging
from typing import Union


def config_logging(
    name: str = "dpat",
    level: Union[int, str] = logging.INFO,
    handler: logging.Handler = logging.StreamHandler(),
    format: str = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
):
    """Configure simple logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = handler
    ch.setLevel(level)

    formatter = logging.Formatter(format)

    ch.setFormatter(formatter)

    logger.addHandler(ch)
