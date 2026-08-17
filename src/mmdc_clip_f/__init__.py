"""Reproducible core implementation for MMDC-CLIP-F."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mmdc-clip-f")
except PackageNotFoundError:  # source checkout without installation
    __version__ = "0.1.0"

__all__ = ["__version__"]
