
from .configuration import logger, logo_color





__version__ = '1.0.0'





logger.log(25,logo_color)
logger.log(25,(f"This is gal3d, version: {__version__}".rjust(64,' ')+ "\n"))