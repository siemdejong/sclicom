import os
import platform

import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# vips must be installed separately for Windows.
# vips already includes OpenSlide.
# Provide the path to vips\bin in project.ini.
# https://github.com/libvips/pyvips
if platform.system() == "Windows":
    PYVIPS_PATH = config["PATHS"]["vips"]
    os.environ["PATH"] = PYVIPS_PATH + ";" + os.environ["PATH"]
    try:
        import pyvips  # isort:skip
        import openslide
    except OSError:
        raise ImportError(
            "Make sure to install vips and set PATHS.vips in project.ini."
        )

import dlup
