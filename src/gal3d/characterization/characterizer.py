import logging
import os
from abc import abstractmethod
from typing import List

from ..optimization.result import ModelResult

from gal3d.plugin import PluginBase, PluginManager

__all__ = ['Characterizer', 'CharacterizerBase']

logger = logging.getLogger("gal3d.characterization.characterizer")


class CharacterizerBase(PluginBase):

    def __init__(self, data: dict | ModelResult):
        if not isinstance(data, (dict, ModelResult)):
            raise TypeError(f"Expected 'data' to be of type 'dict' or 'ModelResult', but got {type(data).__name__}")
        self.data = data

    def __init_subclass__(cls, **kwargs):

        super().__init_subclass__(**kwargs)
        CharacterizerManager.register(cls)

    @abstractmethod
    def measure(self,):
        pass

class CharacterizerManager(PluginManager[CharacterizerBase]):
    """
    Factory class for accessing registered characterizer plugins.
    """
    
    _plugins = {}
    _plugin_module = "gal3d.characterization.characterizer_plugins"
    _base_class = CharacterizerBase

Characterizer = CharacterizerManager
