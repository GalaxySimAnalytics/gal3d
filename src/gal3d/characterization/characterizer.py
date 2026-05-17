"""
Characterizer base class and factory for measuring properties from data or model results.
"""

import logging
from abc import abstractmethod
from typing import Any

from gal3d.optimization.result import ModelResult
from gal3d.plugin import PluginBase, PluginManager

__all__ = ["Characterizer", "CharacterizerBase"]

logger = logging.getLogger("gal3d.characterization.characterizer")


class CharacterizerBase(PluginBase):
    def __init__(self, data: dict | ModelResult):
        if not isinstance(data, dict | ModelResult):
            raise TypeError(f"Expected 'data' to be of type 'dict' or 'ModelResult', but got {type(data).__name__}")
        self.data = data

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Characterizer.register(cls)

    @abstractmethod
    def measure(self, *args, **kwargs):
        pass


class Characterizer(PluginManager[CharacterizerBase]):
    """
    Factory class for accessing registered characterizer plugins.
    """

    _plugins = {}
    _plugin_module = "gal3d.characterization.characterizer_plugins"
    _base_class = CharacterizerBase
