
"""
Module for computing centers, inertia tensors, principal axes, and density estimations using density estimators.

This module provides several functions for working with particles' positions and masses,
including calculating the geometric center, mass-weighted center of mass, the moment of inertia tensor,
the principal axes of particle distributions, and density estimators.

Usage examples
--------------
>>> particles = Particles(pos, mass)
>>> particles.parameter

"""
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .density_estimator import DensityEstimator, DensityEstimatorBase
from .global_calculator import GlobalCalculator

if TYPE_CHECKING:
    from gal3d.optimization.result import ModelResult

class Particles(GlobalCalculator):
    """
    A particle container with density estimation and parameter gradient capabilities.
    """

    def __init__(
        self,
        pos: ArrayLike,
        mass: ArrayLike,
        rmax: float | None = None,
        recenter: bool = True,
        density_estimator: str | DensityEstimatorBase | type[DensityEstimatorBase] = "DensityEstimatorSPH",
        estimator_kwargs: dict | None = None,
    ):
        """
        Initialize the Particles class with particles' positions, weights, and other parameters.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            The positions of N particles in 3D Cartesian coordinates (x, y, z).
        mass : array_like, shape (N,)
            The properties of N particles, such as mass.
        rmax : float, optional
            Maximum radius to include particles. If None, include all particles. Default is None.
        recenter : bool, optional
            Whether to recenter the positions using the shrink-sphere method. Default is True.
        density_estimator : str | DensityEstimatorBase | type[DensityEstimatorBase], optional
            - A plugin name registered in DensityEstimator (e.g., 'DensityEstimatorSPH'), or
            - An instance of DensityEstimatorBase, or
            - A subclass of DensityEstimatorBase to be constructed.
        estimator_kwargs : dict, optional
            Additional keyword arguments passed to `density_estimator`.
        """

        GlobalCalculator.__init__(self, pos, mass, recenter)
        if rmax is not None:
            if rmax <= 0:
                raise ValueError(f"rmax must be positive; got {rmax}")
            sel = (self.r<rmax)
            self.pos = self.pos[sel]
            self.mass = self.mass[sel]
            self.r = self.r[sel]
        estimator_kwargs = {} if estimator_kwargs is None else estimator_kwargs

        if isinstance(density_estimator, str):
            self.estimator = DensityEstimator.get_plugin(density_estimator)(
                self.pos, self.mass, **estimator_kwargs
            )
        elif isinstance(density_estimator, type) and issubclass(density_estimator, DensityEstimatorBase):
            self.estimator = density_estimator(
                self.pos, self.mass, **estimator_kwargs
            )
        elif isinstance(density_estimator,DensityEstimatorBase):
            self.estimator = density_estimator
        else:
            raise TypeError(
                f"density_estimator must be either a string (plugin name), "
                f"a subclass of DensityEstimatorBase, or an instance of it. Got {type(density_estimator)} instead."
            )

    def __del__(self):
        """
        Clean up estimator and call parent class cleanup.
        """
        for attr in ["_parameter", "_gradient", "estimator"]:
            if hasattr(self, attr):
                setattr(self, attr, None)
                delattr(self, attr)

        # Call parent class __del__ method
        try:
            super().__del__()
        except (AttributeError, TypeError):
            pass  # Handle case where parent doesn't have __del__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    @property
    def parameter(self):
        """Cached property that returns the parameter values at the input positions."""
        return self.estimator.parameter

    @property
    def hsm(self):
        """Cached property that returns the half-smooth length at the input positions."""
        return self.estimator.hsm

    @property
    def gradient(self):
        """Cached property that returns the gradient of the parameter at the input positions."""
        return self.estimator.gradient

    def get_parameter(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
        """
        Estimate the parameter value at the target positions.

        Parameters
        ----------
        target_pos : array_like, shape (M, 3)
            Target positions (x, y, z) where parameter values are estimated.
        **kwargs : dict
            Additional keyword arguments passed to the KDTree query.

        Returns
        -------
        np.ndarray, shape (M,)
            Estimated parameter values at target positions.
        """
        return self.estimator.get_parameter(target_pos, **kwargs)

    def get_gradient(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
        """
        Estimate the gradient of the parameter at the target positions.

        Parameters
        ----------
        target_pos : array_like, shape (M, 3)
            Target positions (x, y, z) where the gradient is estimated.
        **kwargs : dict
            Additional keyword arguments passed to the KDTree query.

        Returns:
        gradient: np.ndarray, shape (M, 3)
                Estimated gradient vectors at target positions.
        """
        return self.estimator.get_gradient(target_pos, **kwargs)


    def estimate_spatial_resolution(self) -> float:
        """
        Estimate a spatial resolution scale from the half-smoothing length (hsm).
        Uses a 3-sigma clipped mean and scales by 0.55.

        Returns
        -------
        float
            Estimated spatial resolution (> 0).
        """
        hsm = self.hsm
        d_in = np.median(hsm) - 3 * np.std(hsm)
        d_ou = np.median(hsm) + 3 * np.std(hsm)
        res_r = np.mean(hsm[(hsm > d_in) & (hsm < d_ou)]) * 0.55
        return res_r

    def estimate_mass_resolution(self) -> float:
        """
        Estimate the mass resolution as the mean particle mass.

        Returns
        -------
        float
            Estimated mass resolution.
        """
        return np.mean(self.mass)

    def abc_shape_profile(
        self,
        nbins: int = 100,
        rmin: float | None = None,
        rmax: float | None = None,
        bins: Literal["equal", "log", "lin"] = "equal",
        max_iterations: int = 10,
        tol: float = 1e-3,
    ) -> "ModelResult":
        """
        Fit ellipsoidal shape using an iterative reduced inertia tensor method.

        Parameters
        ----------
        nbins : int, optional
            Number of radial bins (default 100).
        rmin : float, optional
            Minimum radius for binning. If None, set to rmax / 1E3.
        rmax : float, optional
            Maximum radius for binning. If None, set to maximum particle radius.
        bins : {'equal', 'log', 'lin'}, optional
            Binning method for radial shells (default 'equal').
        max_iterations : int, optional
            Maximum iterations per shell (default 10).
        tol : float, optional
            Convergence tolerance (default 1e-3).

        Returns
        -------
        ModelResult
            Summed model result over all radial bins, or EmptyModelResult if no valid results.
        """
        from gal3d.model_workflow.fit_workflow_plugins import IterateEllipsoidWorkflow
        fit = IterateEllipsoidWorkflow()
        return fit(self,nbins=nbins,rmin=rmin,rmax=rmax,bins=bins,max_iterations=max_iterations,tol=tol)


    @classmethod
    def available_estimators(cls) -> list[str]:
        """
        Returns a list of available density estimator plugin names.
        """
        return DensityEstimator.available_plugins()
