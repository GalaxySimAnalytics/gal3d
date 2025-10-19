from typing import Any, Literal, overload
from .characterizer import Characterizer as _Characterizer
from .characterizer_plugins import Bar, Disk, Segment

class Characterizer(_Characterizer):
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["Bar"]) -> type[Bar]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["Disk"]) -> type[Disk]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["Segment"]) -> type[Segment]: ...
