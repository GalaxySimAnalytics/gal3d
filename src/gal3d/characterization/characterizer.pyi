import abc
from _typeshed import Incomplete
from abc import abstractmethod
from gal3d.optimization.result import ModelResult
from gal3d.plugin import PluginBase, PluginManager
from typing import Literal, overload
from gal3d.characterization.characterizer_plugins.galaxy_bar import Bar
from gal3d.characterization.characterizer_plugins.galaxy_disk import Disk
from gal3d.characterization.characterizer_plugins.segment import Segment

__all__ = ['Characterizer', 'CharacterizerBase']

class CharacterizerBase(PluginBase, metaclass=abc.ABCMeta):
    data: Incomplete
    def __init__(self, data: dict | ModelResult) -> None: ...
    def __init_subclass__(cls, **kwargs) -> None: ...
    @abstractmethod
    def measure(self, *args, **kwargs): ...

class Characterizer(PluginManager[CharacterizerBase]):
    """
    Factory class for accessing registered characterizer plugins.
    """

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["Bar"]) -> type[Bar]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["Disk"]) -> type[Disk]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["Segment"]) -> type[Segment]: ...
