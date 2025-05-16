from ..configuration import config
if config['general']['use_cython']:
    from .array_operate_cy import *
else:
    from .array_operate_nb import *