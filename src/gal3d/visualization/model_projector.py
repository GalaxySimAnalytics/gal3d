
import logging
import abc
from typing import List,NoReturn
from functools import wraps
import logging
import os

import numpy as np

from ..util.func_cache import CacheDict
from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty




Update_plugin_stub = True

__all__ = ['ModelProjectorBase','ModelProjector']

logger = logging.getLogger("gal3d.visualization.model_projector")

_ModelProjectorPlugins=dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py','.pyi')


class ModelProjectorBase(abc.ABC):
    
    def __init_subclass__(cls, **kwargs):

        _ModelProjectorPlugins[cls.__name__] = cls
        logger.info(f"Find ModelProjectorPlugin: {cls.__name__} and load successfully")
        if Update_plugin_stub:
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(ModelProjector,ModelProjectorBase,_ModelProjectorPlugins, output_path)
            logger.info(f"✅ Updated stub: {output_path}")
        
    
    def __init__(self,cache_len: int = 100):
        self._image_cache = CacheDict(cache_len=cache_len)


    def ImageCache(func):
        @wraps(func)
        def wrapper(self, x_range,y_range,nbins, z_range, rotation,**kwargs):
            recod = (x_range[0],x_range[1],y_range[0],y_range[1],nbins,z_range[0],z_range[1],rotation.tobytes())
            if recod in self._image_cache:
                logger.info(f"Get image from cache for config: {recod}")
                return self._image_cache[recod]
            else:
                logger.info(f"Cache image, register config: {recod}")
                self._image_cache[recod] = func(self, x_range,y_range,nbins, z_range, rotation,**kwargs)
            return self._image_cache[recod]
        return wrapper
    
    @ImageCache
    def image(self,x_range,y_range, nbins: int = 100, z_range=(-20,20), rotation=np.eye(3),**kwargs):
        
        return self._image( x_range, y_range, nbins, z_range=z_range, rotation=rotation,**kwargs)
    
    
    @abc.abstractmethod
    def _image(self, x_range,y_range, nbins: int = 100, z_range=(-20,20), rotation=np.eye(3),**kwargs):
        pass

    def image_xz(self, x_range,y_range,nbins: int = 100, z_range=(-20,20), ):
        return self.image(x_range,y_range,nbins, z_range, rotation = np.array([[1.,0,0],[0,0,1.],[0,1.,0.]]).T)
    
    def image_yz(self, x_range,y_range,nbins: int = 100, z_range=(-20,20), ):
        return self.image(x_range,y_range,nbins, z_range, rotation = np.array([[0,1.,0.],[0,0,1.],[1.,0,0.]]).T)
    
    
class ModelProjector:
    
    @staticmethod
    def get_plugin( model, model_cric = None,cache_len = 100, plugin: str | None = None,**kwargs) -> ModelProjectorBase:
        
        assert ((plugin is None) or isinstance(plugin,str))
        plugin = 'ProjectorLineIntegration' if plugin is None else plugin
        
        return _ModelProjectorPlugins[plugin](model,model_cric,cache_len,**kwargs)
                    
                    
    @classproperty
    def available_plugins(cls) -> List[str]:
        return list(_ModelProjectorPlugins.keys())
    
        
    
    