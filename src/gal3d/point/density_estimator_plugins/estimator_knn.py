from functools import cached_property
import logging
import os

import numpy as np
from scipy.spatial import KDTree

from .compute_pa_cy import cal_pa
from ..density_estimator import DensityEstimatorBase
from ...util.array_operate import unit_vector3d
from ...util.func_signature import func_optional_key, update_dict_value


logger = logging.getLogger('gal3d.particle.density_estimator.DensityEstimatorKNN')


__all__ = ['DensityEstimatorKNN']


class DensityEstimatorKNN(DensityEstimatorBase):
    '''Estimate the parameter value at any position by kd-tree'''

    def __init__(
        self,
        pos,
        mass,
        parameter_mode: str = 'Density',
        kernel=None,
        k_nearest: int = 32,
        r_cut: float | None = None,
        **kwargs,
    ):
        '''
        Parameters:
            pos: ndarray, shape(n,3)
                The coordinates (x, y, z) of the n data points.
            mass: array, shape(n,)
                The property of the n points (e.g., mass, luminosity, etc.).
            parameter_mode: str, optional
                The mode of parameter estimation.
                - If 'Density' (default), the function estimates the density.
                  For example, if the input `mass` is mass, the function returns density.
                - If 'Mean', the function returns the average value of the `mass` property.
            k_nearest: int, default 32
                The number of nearest points used to estimate the target parameter.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree constructor and query methods.

        Attributes:
            pos: ndarray, shape(n,3)
                The coordinates (x, y, z) of the n data points, sorted by their distance from the origin.
            mass: array, shape(n,)
                The property of the n points, sorted by their distance from the origin.
            r: array, shape(n,)
                The radial distance of each point from the origin.
            tree: scipy.spatial.KDTree
                A KDTree object constructed from the input positions.
            pa_mode: str
                The parameter estimation mode ('Density' or 'Mean').

        Methods:
            get_parameter(target_pos)
                Estimate the parameter value at the target positions.
            get_gradient(target_pos)
                Estimate the gradient of the parameter at the target positions.
        '''
        super().__init__(pos, mass, parameter_mode, kernel)

        self.__generate_kd_options(k_nearest, r_cut, **kwargs)

        logger.info(f"Build KDtree with options {self._tree_build_options}")
        self.tree = KDTree(self.pos, **self._tree_build_options)

    @cached_property
    def parameter(self):
        '''Cached property that returns the parameter values at the input positions.'''
        return self.get_parameter(self.pos)

    @cached_property
    def gradient(self):
        '''Cached property that returns the gradient of the parameter at the input positions.'''
        return self.get_gradient(self.pos)
    
    @cached_property
    def hsm(self):
        """Cached property that returns the half-smooth length at the input positions."""

        return self.get_hsm(self.pos)

    def get_hsm(self, target_pos, **kwargs):
        '''
        Estimate the half-smooth length at the target positions.

        Parameters:
            target_pos: ndarray, shape(m,3)
                The target positions (x, y, z) where the half-smooth length is to be estimated.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree query method.

        Returns:
            results: array, shape(m,)
                The estimated half-smooth lengths at the target positions.
        '''
        target_pos = self._shape_check(target_pos)

        query_options = update_dict_value(self._tree_query_options, kwargs)

        n_d, n_index = self.tree.query(target_pos, **query_options)

        return n_d[:,-1]/2

    def get_parameter(self, target_pos, **kwargs):
        '''
        Estimate the parameter value at the target positions.

        Parameters:
            target_pos: ndarray, shape(m,3)
                The target positions (x, y, z) where the parameter values are to be estimated.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree query method.

        Returns:
            results: array, shape(m,)
                The estimated parameter values at the target positions.
        '''
        target_pos = self._shape_check(target_pos)
                    
        query_options = update_dict_value(self._tree_query_options, kwargs)
        
        n_d, n_index = self.tree.query(target_pos, **query_options)
        
        return self._cal_pa(n_d, n_index, **query_options)

    def get_gradient(self, target_pos, **kwargs):
        '''
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
        '''
        target_pos = self._shape_check(target_pos)
        query_options = update_dict_value(self._tree_query_options, kwargs)

        n_d, n_index = self.tree.query(target_pos, **query_options)

        target_pa = self._cal_pa(n_d, n_index)

        parameter = self.parameter[n_index]

        up_grad_ind = np.argmax(parameter, axis=1, keepdims=True)
        lo_grad_ind = np.argmin(parameter, axis=1, keepdims=True)

        up_grad_index = np.take_along_axis(n_index, up_grad_ind, axis=1)[:, 0]
        lo_grad_index = np.take_along_axis(n_index, lo_grad_ind, axis=1)[:, 0]

        up_dist = np.clip(
            np.take_along_axis(n_d, up_grad_ind, axis=1)[:, 0], a_min=1e-8, a_max=None
        )
        lo_dist = np.clip(
            np.take_along_axis(n_d, lo_grad_ind, axis=1)[:, 0], a_min=1e-8, a_max=None
        )

        up_pa = np.take_along_axis(parameter, up_grad_ind, axis=1)[:, 0] - target_pa
        lo_pa = np.take_along_axis(parameter, lo_grad_ind, axis=1)[:, 0] - target_pa

        up_vect = self.pos[up_grad_index] - target_pos
        lo_vect = self.pos[lo_grad_index] - target_pos

        up_gradient = up_pa / up_dist
        lo_gradient = lo_pa / lo_dist

        return (
            (up_gradient, unit_vector3d(up_vect)),
            (lo_gradient, unit_vector3d(lo_vect)),
        )

    def _cal_pa(self, n_d, n_index, **kwargs):
        '''
        Calculate the parameter value based on the nearest neighbors.

        Parameters:
            n_d: ndarray, shape(m, num_near)
                The distances to the nearest neighbors for each target position.
            n_index: ndarray, shape(m, num_near)
                The indices of the nearest neighbors for each target position.

        Returns:
            fit_pa: array, shape(m,)
                The estimated parameter values based on the nearest neighbors.
        '''
        return cal_pa(n_d.astype(np.float64),
            n_index.astype(np.int32),
            self.mass.astype(np.float64), self.pa_mode, kwargs.get('distance_upper_bound', np.inf))

    def __generate_kd_options(self, k_nearest, r_cut, **kwargs):
        '''
        Generate options for KDTree construction and query.

        Parameters:
            k_nearest: int
                The number of nearest neighbors to consider.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree constructor and query methods.
        '''

        self._tree_build_options = func_optional_key(KDTree)
        self._tree_query_options = func_optional_key(KDTree.query)

        self._tree_build_options['leafsize'] = k_nearest
        self._tree_query_options['workers'] = os.cpu_count()
        self._tree_query_options['k'] = k_nearest
        if r_cut:
            self._tree_query_options['distance_upper_bound'] = r_cut

        logger.info(f"cpu nums: {self._tree_query_options['workers']}")

        self._tree_build_options = update_dict_value(self._tree_build_options, kwargs)

        self._tree_query_options = update_dict_value(self._tree_query_options, kwargs)
