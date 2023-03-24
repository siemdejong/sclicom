Convert images
==============
To convert all images from directory `INPUT_DIR`, and output the images as TIFF in `OUTPUT_DIR`, run

.. code-block::

    dpat convert batch -i INPUT_DIR -o OUTPUT_DIR -e tiff

Large images need to be trusted against decompression bomb DOS attack.
Use the ``--trust`` flag.
To skip images that were already converted to the target extension, use ``--skip-existing``.

.. note::

    If converting to tiff, the input images are assumed to contain the reference to the scanning program, which must be in {200slow, 300slow, 300fast}.
