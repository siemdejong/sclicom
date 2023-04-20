Adding new data
===============

When adding new data to the dataset, do the following:

#.  Update ``labels.csv`` with the new data.
#.  (Optionally) convert the images to to another filetype (e.g. tif), see :ref:`convert-images`.
#.  (Optionally) create masks with the masking tool, see :ref:`create-masks`.
#.  Run e.g. ``dpat splits create -i INPUT_DIR -l labels.csv -f medulloblatoma -f "pilocytic astrocytoma" -n pmchhg --overwrite`` to create new splits for the new dataset.
#.  (Optionally) pretrain a feature extractor with ``python tools/extract_features/train.py``.
#.  Locate the pretrained model, image directory, mask directory, splits files, and optionally an already calculated HDF5 file, and adapt ``tools/extract_features/compile_features.py`` to reflect the locations.
    Optionally, change the number of classes or other parameters.
    Run ``python tools/extract_features/compile_features.py`` to compile the features.
    If step 3 was skipped, the feature extractor does not take into account the new image data.
