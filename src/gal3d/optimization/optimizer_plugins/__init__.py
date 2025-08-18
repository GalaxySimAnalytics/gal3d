"""
Module for optimizer plugins, defining and manipulating optimization algorithms.
"""
# default optimizer
from .optimize_scipy import OptimizerScipy

__all__ = ["OptimizerScipy"]

# optional optimizers
try:
    from .optimize_nlopt import OptimizerNLopt

    __all__ += ["OptimizerNLopt"]
except ImportError:
    pass

try:
    from .optimize_optimagic import OptimizerOptimagic
    __all__ += ["OptimizerOptimagic"]
except ImportError:
    pass

try:
    from .optimize_emcee import OptimizerEmcee
    __all__ += ["OptimizerEmcee"]
except ImportError:
    pass
