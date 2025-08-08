from gal3d.config import config

if config.general.use_cython:
    from .util_cy import *
else:
    from .util_nb import *