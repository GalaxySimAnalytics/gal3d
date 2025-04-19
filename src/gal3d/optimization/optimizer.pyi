import typing
from typing import overload, Type, Literal, List, NoReturn, Union, Any, Sequence
import numpy
from gal3d.optimization.optimizer import OptimizerBase
from gal3d.optimization.optimizer_plugins.optimize_nlopt import OptimizerNLopt
from gal3d.optimization.optimizer_plugins.optimize_optimagic import OptimizerOptimagic
from gal3d.optimization.optimizer_plugins.optimize_scipy import OptimizerScipy

class OptimizerBase:
    def __init__(self, algorithm: str, algo_options: dict | None = None) -> None:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None: ...
    def fitting(
        self,
        fun,
        x0,
        bounds,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        **kwargs,
    ) -> None: ...
    def set_options(self, **kwargs) -> None: ...
    def has_algorim(self, algorithm: str) -> bool: ...

class Optimizer:
    @staticmethod
    def _updata_plugin_stub() -> None: ...
    @staticmethod
    @overload
    def get_plugin(plugin: None) -> OptimizerBase:
        """
        Get an optimizer plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of OptimizerBase
        """
        ...

    @staticmethod
    @overload
    def get_plugin(plugin: Literal['OptimizerNLopt']) -> Type[OptimizerNLopt]: ...
    @staticmethod
    @overload
    def get_plugin(
        plugin: Literal['OptimizerOptimagic'],
    ) -> Type[OptimizerOptimagic]: ...
    @staticmethod
    @overload
    def get_plugin(plugin: Literal['OptimizerScipy']) -> Type[OptimizerScipy]: ...
