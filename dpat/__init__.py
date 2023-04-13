"""dpat.

Deep Learning for Pathology on Higher Harmonic Generation Microscopy
Images.
"""

import logging
import os
import platform

logging.getLogger("dpat").addHandler(logging.NullHandler())

__version__ = "4.9.2"


def install_windows(vipsbin: str):
    """Install dpat for windows.

    Requests vips, such that it can import pyvips [1], openslide [2]
    and dlup [3] in the right order.

    Vips must be installed separately for Windows. Vips already includes OpenSlide.
    Provide the path to vips/bin.

    Parameters
    ----------
    vipsbin : str
        `path/to/vips/bin`.

    Examples
    --------
    >>> import dpat
    >>> dpat.install_windows("D:/apps/vips-dev-8.14/bin")

    References
    ----------
    [1] https://github.com/libvips/pyvips
    [2] https://openslide.org/
    [3] https://github.com/NKI-AI/dlup
    """
    assert platform.system() == "Windows", "install_windows() is for Windows only."

    os.environ["PATH"] = vipsbin + ";" + os.environ["PATH"]

    try:
        import pyvips  # noqa:F401
    except OSError:
        raise ImportError(f"Please check if vips is installed at {vipsbin}")

    import dlup  # noqa:F401
