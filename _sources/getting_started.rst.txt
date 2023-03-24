Getting started
===============

Prerequisites
-------------

Conda
^^^^^
For package management, it is advised to use a conda package manager.
The author recommends Miniforge or Mambaforge.

.. _prerequisites_libvips:

vips
^^^^^
This project depends on ``dlup`` (automatically installed), which depends on vips.
On Windows, vips needs to be installed locally.
Download the latest libvips Windows binary and unzip somewhere.
On Linux/macOS, vips is included with the installation steps below.

OpenSlide
^^^^^^^^^
Vips comes with OpenSlide. It is not needed to install OpenSlide separately.


.. _prerequisites_CUDA:

CUDA
^^^^
To do deep learning on CUDA enabled accelerators, follow installation instructions on pytorch.org. Run nvidia-smi to see if CUDA is available.

Installation
------------

Run the following commands from a conda enabled shell (such as Miniforge Prompt, if Miniforge/Mambaforge is installed).

#.  Clone this repository and change directories

    .. code::

    git clone https://github.com/siemdejong/dpat.git dpat && cd dpat

#.  Create a new conda environment and activate it.

    .. code::

        conda create -n <env_name>
        conda activate <env_name>

#.  Install dependencies from ``environment.yml``.

    .. code::

        conda env update -f environment.yml

#.  If you use this library for deep learning and want to use CUDA-enabled Pytorch,
    follow instructions on `pytorch.org <https://pytorch.org/get-started>`_.
    Make sure CUDA is available, see :ref:`prerequisites_CUDA`.
#.  Install dpat in editable mode with

    .. code-block::

        pip install -e .

#.  Verify installation

    .. code-block::
        python -c "import dpat"

#.  Windows only: if you use this library for use in scripts, make sure libvips is available, see :ref:`prerequisites_libvips`.
    If using this library in a script, make sure to properly install the package with

    .. code:: python

        import dpat
        dpat.install_windows("path/to/vips/bin")

    every time the package is used.
#.  Check if CUDA is available for the installed Pytorch distribution.
    In a Python shell, execute

    .. code:: python

        import torch
        torch.cuda.is_available()

    If ``False`` is returned, install Pytorch following its documentation.
