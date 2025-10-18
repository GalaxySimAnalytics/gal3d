"""
Three-dimensional galaxy shape profile fitting and analysis package.

More information about gal3d can be found at https://github.com/GalaxySimAnalytics/gal3d

"""

from ._info import __version__, logo, print_gal3d_info
from .configuration import config
from .log import logger

__all__ = ["logo","config","logger","__version__","print_gal3d_info"]
