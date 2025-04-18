import os
import logging
from typing import List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .with_parameter import WithParameter, abstractmethod, Parameters
from ..optimization.parameter import Parameters
from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty


Update_plugin_stub = True

__all__ = ['Coordinate','CoordinateBase']

logger = logging.getLogger("gal3d.shape.coordinate")

_CoordinatePlugins=dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py','.pyi')



class CoordinateBase(WithParameter):

        
    def __init_subclass__(cls, **kwargs):
        
        
        if not super().__init_subclass__():
            logger.info(f"Find CoordinatePlugin: {cls.__name__} but fail to load")
            return

        _CoordinatePlugins[cls.__name__] = cls
        logger.info(f"Find CoordinatePlugin: {cls.__name__} and load successfully")
        if Update_plugin_stub:
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(Coordinate,CoordinateBase,_CoordinatePlugins, output_path)
            logger.info(f"✅ Updated stub: {output_path}")
            
    
    @abstractmethod
    def __call__(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the coordinate function at the given positions.
        """
        pass
    
    @abstractmethod
    def jacobian(self, pos: NDArray[np.float64]) -> tuple:
        """
        Computes the Jacobian of the coordinate function at the given positions for each parameters.
        """
        pass
    
    @abstractmethod
    def inverse(self,pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Inverse transform the given positions using the current translation and rotation parameters.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def quick_call(*args, **kwargs) -> NDArray[np.float64]:
        pass
    
    
    @staticmethod
    @abstractmethod
    def quick_jacobian(*args, **kwargs) -> tuple:
        pass
    
    @staticmethod
    @abstractmethod
    def quick_inverse(*args, **kwargs) -> NDArray[np.float64]:
        pass

    
    
class Coordinate:
    
    @staticmethod
    def _updata_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(Coordinate,CoordinateBase,_CoordinatePlugins, output_path)
        logger.info(f"✅ Updated stub: {output_path}")
    
    
    @staticmethod
    def get_plugin(plugin: str | None) -> CoordinateBase:
        """
        Get an geometry plugin
        
        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of CoordinateBase
        """
        assert ((isinstance(plugin,str)) or (plugin is None))
        
        if plugin is None:
            return CoordinateBase
            
        return _CoordinatePlugins[plugin]
    
    
    @classproperty
    def available_plugins(cls) -> List[str]:
        return list(_CoordinatePlugins.keys())
    
    
    
    
from .coordinate_plugins import *