import typing
from typing import overload, Type, Literal, List, NoReturn, Union, Any, Sequence
import numpy
from gal3d.point.density_estimator import DensityEstimatorBase
from gal3d.point.density_estimator_plugins.estimator_knn import DensityEstimatorKNN

class DensityEstimatorBase:
    def __init__(
        self, pos, mass, parameter_mode: str = 'Density', kernel: None = None
    ) -> None:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None: ...
    def _shape_check(self, pos) -> None:
        """
        Ensure the input positions have the correct shape (n, 3).

        Parameters:
            pos: ndarray
                The input positions to be checked and reshaped if necessary.

        Returns:
            pos: ndarray, shape(n,3)
                The reshaped positions.
        """
        ...

    def get_parameter(self, target_pos, **kwargs) -> None:
        """
        Estimate the parameter value at the target positions.

        Parameters:
            target_pos: ndarray, shape(m,3)
                The target positions (x, y, z) where the parameter values are to be estimated.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree query method.

        Returns:
            results: array, shape(m,)
                The estimated parameter values at the target positions.
        """
        ...

    def get_gradient(self, target_pos, **kwargs) -> None:
        """
        Estimate the gradient of the parameter at the target positions.

        Parameters:
            target_pos: ndarray, shape(m,3)
                The target positions (x, y, z) where the gradient is to be estimated.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree query method.

        Returns:
            gradient: tuple of tuples
                A tuple containing two tuples:
                - The first tuple contains the upward gradient magnitude and direction.
                - The second tuple contains the downward gradient magnitude and direction.
        """
        ...

class DensityEstimator:
    @staticmethod
    def _updata_plugin_stub() -> None: ...
    @staticmethod
    @overload
    def get_plugin(plugin: None) -> DensityEstimatorBase:
        """
        Get an DensityEstimator plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of DensityEstimatorBase
        """
        ...

    @staticmethod
    @overload
    def get_plugin(
        plugin: Literal['DensityEstimatorKNN'],
    ) -> Type[DensityEstimatorKNN]: ...
