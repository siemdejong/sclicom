.. _create-masks:

Create masks
==============
To create masks of all images from directory `INPUT_DIR`, and output the masks as PNG in `OUTPUT_DIR`, run

.. code-block::

    dpat mask create INPUT_DIR OUTPUT_DIR

A DLUP compatible mask function can be selected with ``--mask_func DLUP_MASK_FUNC``.
``DLUP_MASK_FUNC`` defaults to ``entropy_masker``, which is currently only available in the `siemdejong/dlup@entropy_masker <https://github.com/siemdejong/dlup/tree/entropy_masker>`_ repo.
The input images can be filtered by extension with ``--ext EXT``.
If images already have masks and recomputing is not needed, use ``--skip-if-exists``.
