from gal3d import config



if config['general']['use_cython']:
    from .ellipsoid_cy import Ellipsoid
    from .ellipsoid_s_cy import Ellipsoid_S
else:
    from .ellipsoid_nb import Ellipsoid
    from .ellipsoid_s_nb import Ellipsoid_S
    


