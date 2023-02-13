import os
import platform
import configparser

config = configparser.ConfigParser()
config.read('project.ini')

# vips must be installed separately for Windows.
# vips already includes OpenSlide.
# Provide the path to vips\bin in project.ini.
# https://github.com/libvips/pyvips
if platform.system() == "Windows":
    PYVIPS_PATH = config["PATHS"]["vips"]
    os.environ["PATH"] = PYVIPS_PATH + ";" + os.environ["PATH"]
    import pyvips
    import openslide

import dlup