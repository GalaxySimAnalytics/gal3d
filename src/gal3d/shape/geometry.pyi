import typing
from typing import overload, Type, Literal, List, NoReturn, Union, Any, Sequence
import numpy
from gal3d.shape.geometry import GeometryBase
from gal3d.shape.geomtry_plugins.ellipsoid import Ellipsoid
from gal3d.shape.geomtry_plugins.ellipsoid_s import Ellipsoid_S

class GeometryBase:

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None: ...

    def __call__(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Evaluates the geometry function at the given positions.
        """
        ...

    def jacobian(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple:
        """
        Computes the Jacobian of the geometry function at the given positions for each parameters.
        """
        ...

    def ray_intersect(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple:
        """
        Computes the intersection between the ray from center and the surface of geometry.

        Parameter:
            pos: position

        Return:
            tuple[ pos, distance]
        """
        ...

    def line_intersect(self, pos1: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], pos2: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Computes the intersection between given line segment and the surface of geometry
        """
        ...

    def f_ray_d(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Ray distance in unit of the ray distance of the surface, 1 means on the surface
        """
        ...

    def ray_point(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple: ...

    def ray_dist(self, pos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple: ...

    @staticmethod
    def quick_call(*args, **kwargs) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Quick version of call, with given parameters, useful in error function
        """
        ...

    @staticmethod
    def quick_f_ray_d(*args, **kwargs) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Quickly evaluates the distance fraction of the geometry function with given parameters and positions, useful in error function
        """
        ...

    @staticmethod
    def quick_ray_dist(*args, **kwargs) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Quickly computes the distance between points and ray points on the surface of the geometry, useful in error function
        """
        ...

    @staticmethod
    def quick_line_intersect(*args, **kwargs) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Quickly computes the intersection between given line segment and the geometry
        """
        ...

    @staticmethod
    def quick_jacobian(*args, **kwargs) -> tuple:
        """
        Quickly computes the Jacobian of the geometry function at the given positions for each parameters.
        """
        ...

class Geometry:

    @staticmethod
    def _updata_plugin_stub() -> None: ...

    @staticmethod
    def _load_plugin() -> None: ...

    @staticmethod
    @overload
    def get_plugin(plugin: None) -> GeometryBase:
        """
        Get an geometry plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of GeometryBase
        """
        ...

    @staticmethod
    @overload
    def get_plugin(plugin: Literal['Ellipsoid']) -> Type[Ellipsoid]:...

    @staticmethod
    @overload
    def get_plugin(plugin: Literal['Ellipsoid_S']) -> Type[Ellipsoid_S]:...
