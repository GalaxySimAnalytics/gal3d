import os
from .configuration import logger, config_parser
from ._info import __version__,logo_color



logger.log(2025, logo_color)
logger.log(2025, ("\n"+f"This is gal3d, version: {__version__}".rjust(64, ' ') + "\n"))

#if logger.level <= 20:
#    logger.info("Verbose mode is on")
