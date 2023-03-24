Making splits
=============
dpat provides a CLI to create train/val/test splits.
Labels must be present at case level in a file containing the following structure:

.. code-block::
    :caption: labels.csv

    case_id,diagnosis,location,
    PMC_HHG_3,medulloblastoma,fourth ventricle,
    PMC_HHG_9,pilocytic astrocytoma,suprasellar,
    etc.

The CLI can be used as

.. code-block::

    dpat splits create -i IMAGE_DIR -o OUTPUT_DIR -l PATH_TO_LABELS_FILE -n NAME

Run ``dpat splits create --help`` to see all options.

By default, 5 folds are created, stratified by `case_id`.
Within every fold, random train/val (80\%/20\%) splits are created from the (super) train set.
The output is 15 files.
train/val/test sets for all 5 folds.
The files can be used for further training.

To filter diagnoses that exactly match diseases, use e.g. ``-f medulloblastoma -f "pilocytic astrocytoma"``.
To filter filenames that match certain values, use a glob pattern.
E.g. ``-y slow.tiff`` to only include images ending with ``slow.tiff``.
To exclude filenames that match certaine values, use a glob pattern with ``-x``.
Exclusion is performed on the set specified by inclusion.
