
from .configuration import logger, config, config_parser





__version__ = '1.0.0'


logger.log(25,(f"This is gal3d, version: {__version__}".rjust(64,' ')+ "\n"))
if logger.level <=20:
    logger.info("Verbose mode is on")