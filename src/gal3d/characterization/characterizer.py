import os
import logging
from abc import ABC, abstractmethod
from typing import List

from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty
from .. import config
from ..optimization.result import ModelResult


__all__ = ['Characterizer', 'CharacterizerBase']

logger = logging.getLogger("gal3d.characterization.characterizer")

_CharacterizerPlugins = dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py', '.pyi')

class CharacterizerBase(ABC):

    def __init__(self, data: dict | ModelResult):

        self.data = data

    def __init_subclass__(cls, **kwargs):
        
        _CharacterizerPlugins[cls.__name__] = cls
        logger.info(f"Find OptimizerPlugin: {cls.__name__} and load successfully")
        if config['update_stub']:
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(
                Characterizer, CharacterizerBase, _CharacterizerPlugins, output_path
            )
            logger.info(f"✅ Updated stub: {output_path}")

    @abstractmethod
    def measure(self,):
        pass

class Characterizer:
    """Characterizer"""

    @staticmethod
    def _updata_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(Characterizer, CharacterizerBase, _CharacterizerPlugins, output_path)
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> CharacterizerBase:
        """
        Get an Characterizer plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of CharacterizerBase
        """
        assert (isinstance(plugin, str)) or (plugin is None)

        if plugin is None:
            return CharacterizerBase
        
        if not _CharacterizerPlugins:
            Characterizer._load_plugin()

        return _CharacterizerPlugins[plugin]

    @staticmethod
    def _load_plugin():
        import importlib
        importlib.import_module("gal3d.characterization.characterizer_plugins")
        logger.info("Successfully loaded Characterizer plugins")
    
    @classproperty
    def available_plugins(cls) -> List[str]:
        if not _CharacterizerPlugins:
            cls._load_plugin()
        return list(_CharacterizerPlugins.keys())

