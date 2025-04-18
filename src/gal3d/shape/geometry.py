
import os
import logging
from typing import List

import numpy as np
from numpy.typing import ArrayLike,NDArray


from .with_parameter import WithParameter, abstractmethod, Parameters
from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty



Update_plugin_stub = True

__all__ = ['Geometry','GeometryBase']

logger = logging.getLogger("gal3d.shape.geometry")

_GeometryPlugins=dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py','.pyi')



class GeometryBase(WithParameter):
    """ 
    
    
    """
        
    def __init_subclass__(cls, **kwargs):
             
        
        if not super().__init_subclass__():
            logger.info(f"Find GeometryPlugin: {cls.__name__} but fail to load")
            return
            
        _GeometryPlugins[cls.__name__] = cls
        logger.info(f"Find GeometryPlugin: {cls.__name__} and load successfully")
        if Update_plugin_stub:
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(Geometry,GeometryBase,_GeometryPlugins, output_path)
            logger.info(f"✅ Updated stub: {output_path}")
        
    @abstractmethod
    def __call__(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the geometry function at the given positions.
        """
        pass
    
    @abstractmethod
    def jacobian(self, pos: NDArray[np.float64]) -> tuple:
        """
        Computes the Jacobian of the geometry function at the given positions for each parameters.
        """
        pass
    
    @abstractmethod
    def ray_intersect(self, pos: NDArray[np.float64]) -> tuple:
        """ 
        Computes the intersection between the ray from center and the surface of geometry.
        
        Parameter:
            pos: position
        
        Return:
            tuple[ pos, distance]
        """
        pass
    
    
    @abstractmethod
    def line_intersect(self, pos1: NDArray[np.float64], pos2: NDArray[np.float64]) -> NDArray[np.float64]:
        """ Computes the intersection between given line segment and the surface of geometry """    
        pass
    
    @abstractmethod
    def f_ray_d(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """ Ray distance in unit of the ray distance of the surface, 1 means on the surface """
        pass
    
    
    def ray_point(self, pos: NDArray[np.float64]) -> tuple:
        return self.ray_intersect(pos)[0]
    
    
    def ray_dist(self, pos: NDArray[np.float64]) -> tuple:
        return self.ray_intersect(pos)[1]
    
    
    @staticmethod
    @abstractmethod
    def quick_call(*args,**kwargs) -> NDArray[np.float64]:
        """ Quick version of call, with given parameters, useful in error function"""
        pass
    
    @staticmethod
    @abstractmethod
    def quick_f_ray_d(*args,**kwargs) -> NDArray[np.float64]:
        """ Quickly evaluates the distance fraction of the geometry function with given parameters and positions, useful in error function"""
        pass
        
    
    @staticmethod
    def quick_ray_dist(*args,**kwargs) -> NDArray[np.float64]:
        """ Quickly computes the distance between points and ray points on the surface of the geometry, useful in error function"""
        pass
    
    @staticmethod
    def quick_line_intersect(*args,**kwargs) -> NDArray[np.float64]:
        """ Quickly computes the intersection between given line segment and the geometry"""
        pass
    
    @staticmethod
    def quick_jacobian(*args,**kwargs) -> tuple:
        """
        Quickly computes the Jacobian of the geometry function at the given positions for each parameters.
        """
        pass




class Geometry:
    @staticmethod
    def _updata_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(Geometry,GeometryBase,_GeometryPlugins, output_path)
        logger.info(f"✅ Updated stub: {output_path}")
    
    
    
    
    @staticmethod
    def get_plugin(plugin: str | None) -> GeometryBase:
        """
        Get an geometry plugin
        
        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of GeometryBase
        """
        assert ((isinstance(plugin,str)) or (plugin is None))
        
        if plugin is None:
            return GeometryBase
            
        return _GeometryPlugins[plugin]
    
    
    @classproperty
    def available_plugins(cls) -> List[str]:
        return list(_GeometryPlugins.keys())
    
from .geomtry_plugins  import *