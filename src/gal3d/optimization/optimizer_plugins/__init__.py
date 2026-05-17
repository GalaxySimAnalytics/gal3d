"""
Module for optimizer plugins, defining and manipulating optimization algorithms.
"""

from logging import getLogger

# default optimizer
from .optimize_scipy import OptimizerScipy

logger = getLogger("gal3d.optimization.optimizer_plugins")

__all__ = ["OptimizerScipy"]

# optional optimizers
try:
    from .optimize_nlopt import OptimizerNLopt

    __all__ += ["OptimizerNLopt"]
except ImportError:
    logger.debug("NLopt not available, skipping OptimizerNLopt plugin. Use 'pip install nlopt' to enable it.")

try:
    from .optimize_optimagic import OptimizerOptimagic

    __all__ += ["OptimizerOptimagic"]
except ImportError:
    logger.debug(
        "Optimagic not available, skipping OptimizerOptimagic plugin. Use 'pip install optimagic' to enable it."
    )

try:
    from .optimize_lmfit import OptimizerLMFit

    __all__ += ["OptimizerLMFit"]
except ImportError:
    logger.debug("LMFit not available, skipping OptimizerLMFit plugin. Use 'pip install lmfit' to enable it.")
