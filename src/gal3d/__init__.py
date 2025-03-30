
import logging

from .gal3d_main import Galaxy3d


def _setup_logging():
    logger = logging.getLogger('gal3d')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter('[%(levelname)s-%(name)s]: %(message)s')
    )
    logger.addHandler(ch)
    return logger



def set_logging_level(level = logging.INFO):
    """
    Set to logging.INFO for more verbose output, or logging.WARNING for less.
    """
    logger = logging.getLogger('gal3d')
    logger.setLevel(level)





logger = _setup_logging()

__version__ = '1.0.0'

logger.debug(f"import gal3d version: {__version__}")