import logging
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from gal3d.configuration import config
from gal3d.point.density_estimator import DensityEstimatorBase
from gal3d.util.func_signature import func_optional_key, update_dict_value

from .compute_pa_cy import sph_density, sph_gradient

logger = logging.getLogger("gal3d.particle.density_estimator.DensityEstimatorSPH")


__all__ = ["DensityEstimatorSPH"]


class DensityEstimatorSPH(DensityEstimatorBase):
    """Estimate the parameter value at any position by kd-tree.

    Attributes
    ----------
    pos: ndarray, shape(n,3)
        The coordinates (x, y, z) of the n data points, sorted by their distance from the origin.
    mass: array, shape(n,)
        The property of the n points, sorted by their distance from the origin.
    r: array, shape(n,)
        The radial distance of each point from the origin.
    tree: scipy.spatial.KDTree
        A KDTree object constructed from the input positions.

    Methods
    -------
    get_parameter(target_pos)
        Estimate the parameter value at the target positions.
    get_gradient(target_pos)
        Estimate the gradient of the parameter at the target positions.
    """

    def __init__(
        self, pos: ArrayLike, mass: np.ndarray, k_nearest: int | None = None, r_cut: float | None = None, **kwargs: Any
    ):
        """
        Parameters
        ----------
        pos: ndarray, shape(n,3)
            The coordinates (x, y, z) of the n data points.
        mass: array, shape(n,)
            The property of the n points, such as mass.
        k_nearest: int, optional
            The number of nearest points used to estimate the target parameter.
        r_cut: float, optional
            The maximum distance to consider for neighbors. If None, no distance cutoff is applied.
        **kwargs: dict, optional
            Additional keyword arguments passed to the KDTree constructor and query methods.
        """
        super().__init__(pos, mass)

        self.__generate_kd_options(k_nearest, r_cut, **kwargs)

        self.tree = KDTree(self.pos, **self._tree_build_options)

    @cached_property
    def parameter(self) -> np.ndarray:
        """Cached property that returns the parameter values at the input positions, and caches hsm."""
        target_pos = self.pos
        query_options = self._tree_query_options
        n_d, n_index = self.tree.query(target_pos, **query_options)
        return self._cal_density(n_d, n_index, **query_options)

    @cached_property
    def gradient(self) -> np.ndarray:
        """Cached property that returns the gradient of the parameter at the input positions."""
        return self.get_gradient(self.pos)

    @cached_property
    def hsm(self) -> np.ndarray:
        """Cached property that returns the half-smooth length at the input positions."""
        return self.get_hsm(self.pos)

    def get_hsm(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
        """
        Estimate the half-smooth length at the target positions.

        Parameters
        ----------
        target_pos: ndarray, shape(m,3)
            The target positions (x, y, z) where the half-smooth length is to be estimated.
        **kwargs: dict, optional
            Additional keyword arguments passed to the KDTree query method.

        Returns
        -------
        results: array, shape(m,)
            The estimated half-smooth lengths at the target positions.
        """
        target_pos = self.to_3d_array(target_pos)

        query_options = update_dict_value(self._tree_query_options, kwargs)

        n_d, n_index = self.tree.query(target_pos, **query_options)

        return n_d[:, -1]

    def get_parameter(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
        """
        Estimate the parameter value at the target positions.

        Parameters
        ----------
        target_pos : ndarray of shape (m, 3)
            The target positions (x, y, z) where the parameter values are to be estimated.
        **kwargs
            Additional keyword arguments passed to the KDTree query method.

        Returns
        -------
        results : ndarray of shape (m,)
            The estimated parameter values at the target positions.
        """
        target_pos = self.to_3d_array(target_pos)

        query_options = update_dict_value(self._tree_query_options, kwargs)

        n_d, n_index = self.tree.query(target_pos, **query_options)

        return self._cal_density(n_d, n_index, **query_options)

    def get_gradient(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
        """
        Estimate the gradient of the parameter at the target positions.

        Parameters
        ----------
        target_pos: ndarray, shape(m,3)
            The target positions (x, y, z) where the gradient is to be estimated.
        **kwargs: dict, optional
            Additional keyword arguments passed to the KDTree query method.

        Returns
        -------
        gradient: array, shape(m, 3)
            The estimated gradients at the target positions.
        """
        target_pos = self.to_3d_array(target_pos)
        query_options = update_dict_value(self._tree_query_options, kwargs)

        n_d, n_index = self.tree.query(target_pos, **query_options)

        return self._cal_gradient(target_pos, n_d, n_index)

    def _cal_gradient(self, target_pos: np.ndarray, n_d: np.ndarray, n_index: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Calculate the gradient based on the nearest neighbors.

        Parameters
        ----------
        n_d: ndarray, shape(m, num_near)
            The distances to the nearest neighbors for each target position.
        n_index: ndarray, shape(m, num_near)
            The indices of the nearest neighbors for each target position.

        Returns
        -------
        gradient: array, shape(m, 3)
            The estimated gradients at the target positions.
        """
        # Placeholder implementation
        return sph_gradient(
            n_d.astype(np.float64),
            n_index.astype(np.int32),
            self.mass.astype(np.float64),
            self.pos.astype(np.float64),
            self.hsm.astype(np.float64),
            target_pos.astype(np.float64),
        )

    def _cal_density(self, n_d: np.ndarray, n_index: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Calculate the parameter value based on the nearest neighbors.

        Parameters
        ----------
        n_d: ndarray, shape(m, num_near)
            The distances to the nearest neighbors for each target position.
        n_index: ndarray, shape(m, num_near)
            The indices of the nearest neighbors for each target position.

        Returns
        -------
        fit_pa: array, shape(m,)
            The estimated parameter values based on the nearest neighbors.
        """
        return sph_density(
            n_d.astype(np.float64), n_index.astype(np.int32), self.mass.astype(np.float64), self.hsm.astype(np.float64)
        )

    def __generate_kd_options(self, k_nearest: int | None = None, r_cut: None | float = None, **kwargs: Any) -> None:
        """
        Generate options for KDTree construction and query.

        Parameters
        ----------
        k_nearest: int, optional
            The number of nearest neighbors to consider.
        r_cut: float, optional
            The maximum distance to consider for neighbors.
        **kwargs: dict, optional
            Additional keyword arguments passed to the KDTree constructor and query methods.
        """

        self._tree_build_options = func_optional_key(KDTree)
        self._tree_query_options = func_optional_key(KDTree.query)

        self._tree_build_options["leafsize"] = config.densityknn.leafsize
        self._tree_query_options["workers"] = config.densityknn.workers
        self._tree_query_options["k"] = k_nearest if k_nearest is not None else config.densityknn.k_neighbors
        if r_cut:
            self._tree_query_options["distance_upper_bound"] = r_cut

        logger.debug("cpu nums: %d", self._tree_query_options["workers"])

        self._tree_build_options = update_dict_value(self._tree_build_options, kwargs)

        self._tree_query_options = update_dict_value(self._tree_query_options, kwargs)

        changed_keys = ["leafsize"] + list(kwargs.keys())
        changed_options = {k: self._tree_build_options[k] for k in changed_keys if k in self._tree_build_options}
        changed_options["workers"] = self._tree_query_options["workers"]
        logger.debug("Build KDTree with options: %s", changed_options)
