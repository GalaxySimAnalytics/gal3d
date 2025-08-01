import logging
import os
from abc import ABC, abstractmethod
from typing import List

from .. import config_parser
from ..optimization.result import ModelResult
from ..util.func_decorator import classproperty
from ..util.func_signature import generate_plugin_stub

__all__ = ['Characterizer', 'CharacterizerBase']

logger = logging.getLogger("gal3d.characterization.characterizer")

_CharacterizerPlugins = dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py', '.pyi')

class CharacterizerBase(ABC):

    def __init__(self, data: dict | ModelResult):
        if not isinstance(data, (dict, ModelResult)):
            raise TypeError(f"Expected 'data' to be of type 'dict' or 'ModelResult', but got {type(data).__name__}")
        self.data = data

    def __init_subclass__(cls, **kwargs):
        
        _CharacterizerPlugins[cls.__name__] = cls
        logger.info(f"CharacterizerPlugin found: {cls.__name__} and loaded successfully")
        if config_parser['general'].getboolean("update_stub"):
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(
                Characterizer, CharacterizerBase, _CharacterizerPlugins, output_path
            )
            logger.info(f"✅ Updated stub: {output_path}")

    @abstractmethod
    def measure(self,):
        pass

class Characterizer:
    """
    Characterizer

    This class serves as the main interface for managing and interacting with 
    Characterizer plugins. It provides methods to retrieve available plugins, 
    load plugins dynamically, and update plugin stubs for type hinting.

    Methods:
        - _update_plugin_stub: Updates the plugin stub file for type hinting.
        - get_plugin: Retrieves a specific Characterizer plugin by name.
        - _load_plugin: Dynamically loads Characterizer plugins.
        - available_plugins: Returns a list of all available plugin names.
    """

    @staticmethod
    def _update_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(Characterizer, CharacterizerBase, _CharacterizerPlugins, output_path)
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> CharacterizerBase:
        """
        Get a Characterizer plugin.

        Parameters:
        plugin: str,
            The name of the plugin. If the plugin is not found in the available plugins, 
            a ValueError will be raised. Use `available_plugins` to see the list of valid plugins.

        Returns:
            An instance of CharacterizerBase corresponding to the specified plugin.
        """
        assert (isinstance(plugin, str)) or (plugin is None)
        
        if not _CharacterizerPlugins:
            Characterizer._load_plugin()
            
        if plugin is not None and plugin not in _CharacterizerPlugins:
            raise ValueError(f"Plugin '{plugin}' not found in available plugins: {list(_CharacterizerPlugins.keys())}")

        if plugin is None:
            return CharacterizerBase

        return _CharacterizerPlugins[plugin]

    @staticmethod
    def _load_plugin():
        import importlib
        try:
            importlib.import_module("gal3d.characterization.characterizer_plugins")
            logger.info("Successfully loaded Characterizer plugins")
        except ImportError as e:
            logger.error(f"Failed to load Characterizer plugins: {e}")
    
    @classproperty
    def available_plugins(cls) -> List[str]:
        if not _CharacterizerPlugins:
            cls._load_plugin()
        return list(_CharacterizerPlugins.keys())

