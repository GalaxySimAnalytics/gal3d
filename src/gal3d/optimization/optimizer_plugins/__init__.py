
# default optimizer
from .optimize_scipy import OptimizerScipy

# optional optimizers
try:
    from .optimize_nlopt import OptimizerNLopt
except ImportError:
    pass
    
try:
    from .optimize_optimagic import OptimizerOptimagic
except ImportError:
    pass

try:
    from .optimize_emcee import OptimizerEmcee
except ImportError:
    pass