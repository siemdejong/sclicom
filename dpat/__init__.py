import logging
import os
import platform

import yaml

logging.getLogger("dpat").addHandler(logging.NullHandler())

# vips must be installed separately for Windows.
# vips already includes OpenSlide.
# Provide the path to vips\bin in project.ini.
# https://github.com/libvips/pyvips
if platform.system() == "Windows":
    try:
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"Provide a config.yml file to {os.getcwd()}.")

    try:
        PYVIPS_PATH = config["PATHS"]["vips"]
        os.environ["PATH"] = PYVIPS_PATH + ";" + os.environ["PATH"]
    except KeyError:
        raise Exception(f"Please check and set PATHS.vips to in config.yml.")

    try:
        import pyvips  # isort:skip
        import openslide
    except OSError:
        raise ImportError(
            "Make sure to download and extract vips point PATHS.vips in config.yml to vips/bin."
        )

import dlup
