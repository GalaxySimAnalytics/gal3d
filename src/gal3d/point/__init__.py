
"""
Module for computing centers, inertia tensors, principal axes, and density estimations using Numba-accelerated routines.

This module provides several functions for working with particles' positions and masses,
including calculating the geometric center, mass-weighted center of mass, the moment of inertia tensor,
the principal axes of particle distributions, and density estimators.
"""
# kernel #TODO

from .global_calculator import GlobalCalculator
from .density_estimator import DensityEstimator, DensityEstimatorBase
from ..util.func_decorator import classproperty


class Particles(GlobalCalculator):
    """
    A particle container with density estimation and parameter gradient capabilities.
    """

    def __init__(
        self,
        pos,
        mass,
        parameter_mode: str = 'Density',
        density_estimator: str | DensityEstimatorBase = 'DensityEstimatorKNN',
        estimator_kwargs: dict | None = None,
    ):
        """
        Initialize the Particles class with particle positions, weights, and other parameters.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            The positions of N particles in 3D Cartesian coordinates (x, y, z).
        mass : array_like, shape (N,)
            The properties of N particles, such as mass.
        parameter_mode : str, optional
            {'Density', 'Mean'}, determines how to calculate the target parameter. Default is 'Density'.
        density_estimator:

        estimator_kwargs : dict, optional
            Additional keyword arguments passed to `density_estimator`.
        """

        GlobalCalculator.__init__(self, pos, mass)
        estimator_kwargs = {} if estimator_kwargs is None else estimator_kwargs

        if isinstance(density_estimator, str):
            self.estimator = DensityEstimator.get_plugin(density_estimator)(
                self.pos, self.mass, parameter_mode, **estimator_kwargs
            )
        elif issubclass(density_estimator, DensityEstimatorBase):
            self.estimator = density_estimator(
                self.pos, self.mass, parameter_mode, **estimator_kwargs
            )
        elif isinstance(density_estimator,DensityEstimatorBase):
            self.estimator = density_estimator
        else:
            raise TypeError(
    f"density_estimator must be either a string (plugin name), "
    f"a subclass of DensityEstimatorBase, or an instance of it. Got {type(density_estimator)} instead.")

    @property
    def parameter(self):
        """Cached property that returns the parameter values at the input positions."""
        return self.estimator.parameter

    @property
    def gradient(self):
        """Cached property that returns the gradient of the parameter at the input positions."""
        return self.estimator.gradient

    def get_parameter(self, target_pos, **kwargs):
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
        return self.estimator.get_parameter(target_pos, **kwargs)

    def get_gradient(self, target_pos, **kwargs):
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
        return self.estimator.get_gradient(target_pos, **kwargs)
    @classproperty
    def available_estimator(cls):
        """
        Returns a list of available density estimator plugin names.
        """
        return DensityEstimator.available_plugins