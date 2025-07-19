from gal3d import config

if config['general']['use_cython']:
    from .util_cy import *
else:
    from .util_nb import *