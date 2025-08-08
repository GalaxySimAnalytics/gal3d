import os

from ._info import __version__, logo_color
from .config import config
from .log import logger

#logger.log(2025, logo_color)
#logger.log(2025, ("\n"+f"This is gal3d, version: {__version__}".rjust(64, ' ') + "\n"))
# Only print in script context
