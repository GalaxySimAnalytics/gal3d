
import os
import logging
from abc import ABC,abstractmethod
from typing import List

from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty
from .. import config


__all__ = ['Optimizer','OptimizerBase']

logger = logging.getLogger("gal3d.optimization.optimizer")

_OptimizerPlugins=dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py','.pyi')

class OptimizerBase(ABC):
        
        
    def __init__(self,algorithm: str, algo_options: dict | None = None):
        
        if not self.has_algorim(algorithm):
            raise ValueError(f"{algorithm} is not a valid algorithm name.\n")
        
        self.algo_name = algorithm

        self.algo_options = algo_options or {}
        
    def __init_subclass__(cls, **kwargs):
        _OptimizerPlugins[cls.__name__] = cls
        logger.info(f"Find OptimizerPlugin: {cls.__name__} and load successfully")
        if config['update_stub']:
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(Optimizer,OptimizerBase,_OptimizerPlugins, output_path)
            logger.info(f"✅ Updated stub: {output_path}")
        
    @abstractmethod
    def fitting(self, fun, x0, bounds, func_args: tuple | None = None, func_kwargs: dict | None = None,**kwargs):
        pass
    

    def set_options(self, **kwargs):
        self.algo_options.update(**kwargs)

    
    def has_algorim(self, algorithm: str ) -> bool:
        if algorithm in self.available_algorithm:
            return True        
        return False
    
    @classproperty
    @abstractmethod
    def available_algorithm(self) -> List[str]:
        pass
    
class Optimizer:
    """ Optimizer """
    
    @staticmethod
    def _updata_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(Optimizer,OptimizerBase,_OptimizerPlugins, output_path)
        logger.info(f"✅ Updated stub: {output_path}")
    
    @staticmethod
    def get_plugin(plugin: str | None) -> OptimizerBase:
        """
        Get an optimizer plugin
        
        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of OptimizerBase
        """
        assert ((isinstance(plugin,str)) or (plugin is None))
        
        if plugin is None:
            return OptimizerBase
            
        return _OptimizerPlugins[plugin] 
        
        
    @classproperty
    def available_plugins(cls) -> List[str]:
        return list(_OptimizerPlugins.keys())
    
from .optimizer_plugins  import *