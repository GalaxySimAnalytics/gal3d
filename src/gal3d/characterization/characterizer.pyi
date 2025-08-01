import typing
from typing import Any, List, Literal, NoReturn, Sequence, Type, Union, overload

import numpy

import gal3d
from gal3d.characterization.characterizer import CharacterizerBase
from gal3d.characterization.characterizer_plugins.galaxy_bar import Bar

class CharacterizerBase:

    def __init__(self, data: dict | gal3d.optimization.result.ModelResult) -> None:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None: ...

    def measure(self) -> None: ...

class Characterizer:

    @staticmethod
    def _updata_plugin_stub() -> None: ...

    @staticmethod
    def _load_plugin() -> None: ...

    @staticmethod
    @overload
    def get_plugin(plugin: None) -> CharacterizerBase:
        """
        Get an Characterizer plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of CharacterizerBase
        """
        ...

    @staticmethod
    @overload
    def get_plugin(plugin: Literal['Bar']) -> Type[Bar]:...
