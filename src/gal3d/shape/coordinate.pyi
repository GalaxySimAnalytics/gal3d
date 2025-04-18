import typing
from typing import overload, Type, Literal, List, NoReturn, Union, Any
import numpy
from gal3d.shape.coordinate import CoordinateBase
from gal3d.shape.coordinate_plugins.euler_shift import EulerShift

class CoordinateBase:

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None: ...

    def __call__(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Evaluates the coordinate function at the given positions.
        """
        ...

    def jacobian(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple:
        """
        Computes the Jacobian of the coordinate function at the given positions for each parameters.
        """
        ...

    def inverse(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Inverse transform the given positions using the current translation and rotation parameters.
        """
        ...

    @staticmethod
    def quick_call(*args, **kwargs) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...

    @staticmethod
    def quick_jacobian(*args, **kwargs) -> tuple: ...

    @staticmethod
    def quick_inverse(*args, **kwargs) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...

class Coordinate:

    @staticmethod
    def _updata_plugin_stub() -> None: ...

    @staticmethod
    @overload
    def get_plugin(plugin: None) -> CoordinateBase:
        """
        Get an geometry plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of CoordinateBase
        """
        ...

    @staticmethod
    @overload
    def get_plugin(plugin: Literal['EulerShift']) -> Type[EulerShift]:...
