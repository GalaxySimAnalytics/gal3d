import typing
from typing import overload, Type, Literal, List, NoReturn, Union, Any, Sequence
import numpy
from gal3d.visualization.model_projector import ModelProjectorBase
from gal3d.visualization.model_projector_plugins.projector_line_integration import ProjectorLineIntegration
from gal3d.visualization.model_projector_plugins.projector_sph_grid import ProjectorSphGrid

class ModelProjectorBase:

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None: ...

    def __init__(self, cache_len: int = 100) -> None:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def ImageCache(func) -> None: ...

    def image(self, x_range, y_range, nbins: int = 100, z_range: Sequence = (-20, 20), rotation: None | numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = None, **kwargs) -> None: ...

    def _image(self, x_range, y_range, nbins: int = 100, z_range: Sequence = (-20, 20), rotation: None | numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = None, **kwargs) -> None: ...

    def image_xz(self, x_range, y_range, nbins: int = 100, z_range: Sequence = (-20, 20)) -> None: ...

    def image_yz(self, x_range, y_range, nbins: int = 100, z_range: Sequence = (-20, 20)) -> None: ...

class ModelProjector:

    @staticmethod
    def _updata_plugin_stub() -> None: ...

    @staticmethod
    def _load_plugin() -> None: ...

    @staticmethod
    @overload
    def get_plugin(plugin: None) -> ModelProjectorBase:
        """
        Get an geometry plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of ModelProjectorBase
        """
        ...

    @staticmethod
    @overload
    def get_plugin(plugin: Literal['ProjectorLineIntegration']) -> Type[ProjectorLineIntegration]:...

    @staticmethod
    @overload
    def get_plugin(plugin: Literal['ProjectorSphGrid']) -> Type[ProjectorSphGrid]:...
