"""Provide datasets and transforms."""
import platform

if platform.system() == "Windows":
    import dpat

    dpat.install_windows(r"D:\apps\vips-dev-8.14\bin")
