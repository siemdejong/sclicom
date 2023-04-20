"""Use the trained N2V model to denoise images."""

import argparse
import os
import sys
from glob import glob

import csbdeep.io
import numpy as np
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

# We import all our dependencies.
from n2v.models import N2V

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--baseDir", help="directory in which all your network will live", default="models"
)
parser.add_argument("--name", help="name of your network", default="N2V")
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--fileName", help="name of your data file", default="*.tif")
parser.add_argument(
    "--output", help="The path to which your data is to be saved", default="."
)
parser.add_argument("--dims", help="dimensions of your data", default="YX")
parser.add_argument(
    "--tile",
    help="will cut your image [TILE] times in every dimension to make it fit GPU"
    "memory",
    default=1,
    type=int,
)

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

assert ("T" not in args.dims) or (args.dims[0] == "T")

# A previously trained model is loaded by creating a new N2V-object without providing a
# 'config'.
model_name = args.name
basedir = args.baseDir
model = N2V(config=None, name=model_name, basedir=basedir)


tiles = (args.tile, args.tile)

if "Z" in args.dims or "C" in args.dims:
    tiles = (1, args.tile, args.tile)

if "Z" in args.dims and "C" in args.dims:
    tiles = (1, args.tile, args.tile, 1)

datagen = N2V_DataGenerator()
imgs = datagen.load_imgs_from_directory(
    directory=args.dataPath, dims=args.dims, filter=args.fileName
)


files = glob(os.path.join(args.dataPath, args.fileName))
files.sort()

for i, img in enumerate(imgs):
    img_ = img

    if "Z" in args.dims:
        myDims = "TZYXC"
    else:
        myDims = "TYXC"

    if "C" not in args.dims:
        img_ = img[..., 0]
        myDims = myDims[:-1]

    myDims_ = myDims[1:]

    # if we have a time dimension we process the images one by one
    if args.dims[0] == "T":
        outDims = myDims
        pred = img_.copy()
        for j in range(img_.shape[0]):
            pred[j] = model.predict(img_[j], axes=myDims_, n_tiles=tiles)
    else:
        outDims = myDims_
        img_ = img_[0, ...]
        # Denoise the image.
        pred = model.predict(img_, axes=myDims_, n_tiles=tiles)
    outpath = args.output
    filename = os.path.basename(files[i]).replace(".tif", "_N2V.tif")
    outpath = os.path.join(outpath, filename)
    csbdeep.io.save_tiff_imagej_compatible(outpath, pred.astype(np.float32), outDims)
