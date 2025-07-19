import logging
from collections.abc import Iterable
from typing import Dict, Callable

import numpy as np


from .spherical_vector import SphVector
from .ray import MonotonRay
from ...point import Particles
from ...util.func_decorator import timer

logger = logging.getLogger("gal3d.preprocessing.spherical_field.field")


timing = timer(logger)


class SphField:

    _bound_method: Dict[str, Callable] = {}
    _step_method: Dict[str, Callable] = {}
    _iso_method: Dict[str, Callable] = {}

    def __init__(
        self, particles: Particles, num_ray=1024, ray_method='fibonacci', **kwargs
    ):
        '''
        Initialize the Field class for calculating parameters at various distances in a 3D galaxy model.

        Parameters
        ----------
        particles : Particles
            An instance of the Particles class used to compute parameters at specific positions.

        num_ray : int, optional
            The number of rays to generate. Default is 1024.

        ray_method : str, optional
            {'fibonacci', 'muller'}, the method used to generate unit ray vectors. Default is 'fibonacci'.
        '''

        self._build_ray_vector(particles, num_ray, ray_method)

    @timing
    def _build_ray_vector(self, particles, num_ray, ray_method):

        self.rays = SphVector(num_ray, ray_method)
        self.particles = particles
        self.rays_index = self.rays.assign_points(particles.pos)
        self.rays_points_num = np.bincount(self.rays_index)

        max_num_dex = np.argmax(self.rays_points_num)
        logger.info(
            f"Ray {max_num_dex} has the maximum particle count of {self.rays_points_num[max_num_dex]}. "
        )

        min_num_dex = np.argmin(self.rays_points_num)
        logger.info(
            f"Ray {min_num_dex} has the minimum particle count of {self.rays_points_num[min_num_dex]}. "
        )
        if self.rays_points_num[min_num_dex] < 3:
            logger.error(f"It should be > 2, so please make the ray num smaller. ")

        ind = [[] for _ in range(num_ray)]
        for i, j in enumerate(self.rays_index):
            ind[j].append(i)

        self.points_index = [np.array(i) for i in ind]
        return self

    @timing
    def build_field_boundary(
        self, inner=0.5, outer=95, inner_mode='dist', outer_mode='pct'
    ):
        """
        inner : float, optional
            The inner boundary value. Default is 0.5.
        outer : float, optional
            The outer boundary value. Default is 95.

        inner_mode : str, optional
            The mode for calculating the inner boundary. Options are 'dist', 'pct', or 'value'. Default is 'dist'.
        outer_mode : str, optional
            The mode for calculating the outer boundary. Options are 'dist', 'pct', or 'value'. Default is 'pct'.
        """

        self.inner_r = self._bound_method[inner_mode](self, inner, mode='min')
        r_in_min = np.min(self.inner_r)
        r_in_max = np.max(self.inner_r)
        logger.info(f"Field inner boundaries range from {r_in_min:.2f} to {r_in_max:.2f}")
        if r_in_min/r_in_max < 0.09:
            logger.warning("The axial ratio of the inner boundary shape is quite extreme. Consider limiting the particles or refining the boundary.")

        self.outer_r = self._bound_method[outer_mode](self, outer, mode='max')
        r_ou_min = np.min(self.outer_r)
        r_ou_max = np.max(self.outer_r)
        logger.info(f"Field outer boundaries range from {r_ou_min:.2f} to {r_ou_max:.2f}")
        if r_ou_min/r_ou_max < 0.09:
            logger.warning("The axial ratio of the outer boundary shape is quite extreme. Consider limiting the particles or refining the boundary.")

        self.rays_vect = self.rays.pos
        self.check_boundary()
        return self

    @timing
    def build_profile_sample(
        self,
        num_p: int = 500,
        step_mode: str = 'log',
    ):
        '''
        Build a sample of points along the rays for parameter calculation.

        Parameters
        ----------
        base : Local_est
            An instance of the Local_est class used to compute parameters.
        num_p : int, optional
            The number of points to sample along each ray. Default is 500.
        step_mode : str, optional
            The mode for spacing the points. Options are 'lin' for linear spacing or 'log' for logarithmic spacing. Default is 'log'.
        '''

        self.points_r = self._step_method[step_mode](self, num_p)
        self.points_pos = np.einsum('ij,ik->ijk', self.points_r, self.rays.pos)

        points_que = self.points_pos.reshape(
            self.points_pos.shape[0] * self.points_pos.shape[1], 3
        )
        self.points_parameter = self.particles.get_parameter(points_que).reshape(
            self.points_r.shape
        )
        return self

    @timing
    def build_profile_interpolator(
        self, interpolator_method='LU', f_de=True, interpolator_kwargs=dict(), **kwargs
    ):
        '''
        Build interpolators for the sampled points.

        Parameters
        ----------
        interpolator_method : str, optional
            The method used for interpolation. Default is 'LU'.
        f_de : bool, optional
            Whether to use density estimation. Default is True.
        interpolator_kwargs : dict, optional
            Additional keyword arguments for the interpolator. Default is an empty dictionary.
        **kwargs : dict
            Additional keyword arguments.
        '''
        self.rays_func = [
            MonotonRay(
                self.points_r[i],
                self.points_parameter[i],
                f_de=f_de,
                interpolator_method=interpolator_method,
                interpolator_kwargs=interpolator_kwargs,
                **kwargs,
            )
            for i in range(len(self.points_parameter))
        ]

        return self

    @timing
    def build_isodensity_profile(
        self,
        method: str = 'pair',
        from_rays_func: bool = False,
        res_b: float = 0.2,
        res_c: float = 0.1,
        **kwargs,
    ):
        '''
        Build isodensity profiles for the galaxy model.

        Parameters
        ----------
        Base : Galaxy3D
            An instance of the Galaxy3D class used to compute parameters.
        method : str, optional
            The method used for building isoprofiles. Options are 'moi' or 'pair'. Default is 'pair'.
        from_rays_func : bool, optional
            Whether to use the ray functions for building isoprofiles. Default is False.
        res_b : float, optional
            Resolution parameter for the isoprofile. Default is 0.2.
        res_c : float, optional
            Resolution parameter for the isoprofile. Default is 0.1.
        **kwargs : dict
            Additional keyword arguments.
        '''

        self.set_isodensity_sphere(from_rays_func=from_rays_func, **kwargs)

        self.iso_pro_parameter = self._iso_method[method](
            self.rays_vect, self.iso_parameters, res_b, res_c
        )
        interpolator_method = kwargs.get('interpolator_method', 'LU')

        self.iso_pro_func = MonotonRay(
            self.iso_pro_r,
            self.iso_pro_parameter,
            interpolator_method=interpolator_method,
        )

        return self

    @timing
    def set_isodensity_sphere(self, from_rays_func=False, **kwargs):
        '''
        Set the isosphere for the galaxy model.

        Parameters
        ----------
        from_rays_func : bool, optional
            Whether to use the ray functions for setting the isosphere. Default is False.
        **kwargs : dict
            Additional keyword arguments.
        '''
        num_p = kwargs.get('num_p', self.points_r.shape[1])

        if from_rays_func:
            self.iso_pro_r = np.geomspace(
                np.max(self.inner_r), np.min(self.outer_r), num_p
            )
            self.iso_points = np.einsum('ij,k->ikj', self.rays_vect, self.iso_pro_r)
            self.iso_parameters = np.array(
                [self.rays_func[i](self.iso_pro_r) for i in range(len(self.rays_func))]
            )

        else:
            self.iso_pro_r = np.geomspace(
                np.percentile(self.inner_r, 50), np.percentile(self.outer_r, 50), num_p
            )
            self.iso_points = np.einsum('ij,k->ikj', self.rays_vect, self.iso_pro_r)

            self.iso_parameters = self.particles.get_parameter(
                self.iso_points.reshape(
                    self.iso_points.shape[0] * self.iso_points.shape[1], 3
                )
            ).reshape((self.iso_points.shape[0], self.iso_points.shape[1]))

        return self

    def check_boundary(self):
        '''
        Check if the outer boundaries are greater than the inner boundaries.

        Raises
        ------
        ValueError
            If any outer boundary is not greater than the corresponding inner boundary.
        '''
        if not all(self.outer_r > self.inner_r):
            ind = np.arange(len(self.outer_r))
            ind = ind[(self.outer_r < self.inner_r)]
            logger.error(
                f'The outer boundaries need to be greater than the inner boundaries. Check Ray {ind}'
            )
            raise ValueError(
                'The outer boundaries need to be greater than the inner boundaries'
            )
        return

    def pos_ray_n(self, n: int) -> np.ndarray:
        '''
        Retrieve the positions of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The coordinates of the points closest to the nth ray.
        '''
        return self.particles.pos[self.points_index[n]]

    def r_ray_n(self, n: int) -> np.ndarray:
        '''
        Retrieve the radii of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The radii of the points closest to the nth ray.
        '''
        return self.particles.r[self.points_index[n]]

    def mass_ray_n(self, n: int) -> np.ndarray:
        '''
        Retrieve the mass of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The mass of the points closest to the nth ray.
        '''
        return self.particles.mass[self.points_index[n]]

    def parameter_ray_n(self, n: int) -> np.ndarray:
        '''
        Retrieve the parameters of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The parameters of the points closest to the nth ray.
        '''
        return self.particles.parameter[self.points_index[n]]

    def gradient_ray_n(self, n: int) -> tuple:
        '''
        Retrieve the gradients of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        tuple
            The gradients of the points closest to the nth ray.
        '''
        return (
            (
                self.particles.gradient[0][0][self.points_index[n]],
                self.particles.gradient[0][1][self.points_index[n]],
            ),
            (
                self.particles.gradient[1][0][self.points_index[n]],
                self.particles.gradient[1][1][self.points_index[n]],
            ),
        )

    def generate(self, r: Iterable | float, for_fit=False, **kwargs) -> dict:
        '''
        Generate the equivalent surface for a given radius.

        Parameters
        ----------
        r : Iterable | float
            The radius or radii for which to generate the equivalent surface.

        Returns
        -------
        dict
            A dictionary containing the positions, parameters, and radii of the equivalent surface.
        '''

        level = kwargs.get('level', (0, 0))

        ftarget = self.query_iso_f(r, which=level[0])

        rtarget = self.query_rays_r(
            ftarget, which=level[1]
        )  # shape (len(ftarget),num_rays)

        if isinstance(r, Iterable):
            target_pos = np.einsum(
                'ji,ik->jik', rtarget, self.rays_vect
            )  # (len(ftarget),num_rays) * (num_rays , 3)
        else:
            target_pos = np.einsum('i,ik->ik', rtarget, self.rays_vect)

        Eq_surface = {}
        Eq_surface['pos'] = target_pos
        Eq_surface['parameter'] = ftarget
        Eq_surface['r'] = rtarget
        if for_fit:
            Eq_surface['pos'] = Eq_surface['pos'][~np.isnan(Eq_surface['r'])]
            Eq_surface['r'] = Eq_surface['r'][~np.isnan(Eq_surface['r'])]
            Eq_surface['r'] = Eq_surface['r'] / np.sqrt(
                np.sum(Eq_surface['r'] ** 2) / len(Eq_surface['r'])
            )  #  normalization as this used for calculate error
            Eq_surface['info'] = {'parameter': Eq_surface['parameter']}

        return Eq_surface

    def generate_by_f(self, f: Iterable | float, **kwargs) -> dict:
        '''
        Generate the equivalent surface for a given parameter value.

        Parameters
        ----------
        f : Iterable | float
            The parameter value or values for which to generate the equivalent surface.
        level : tuple, optional
            The level of the equivalent surface. Default is (0, 0).

        Returns
        -------
        dict
            A dictionary containing the positions, radii, and isoradii of the equivalent surface.
        '''
        level = kwargs.get('level', (0, 0))

        rtarget = self.query_rays_r(f, which=level[1])

        if isinstance(f, Iterable):
            target_pos = np.einsum('ji,ik->jik', rtarget, self.rays_vect)
        else:
            target_pos = np.einsum('i,ik->ik', rtarget, self.rays_vect)

        iso_r = self.query_iso_r(f, which=level[0])

        Eq_surface = {}
        Eq_surface['pos'] = target_pos
        Eq_surface['r'] = rtarget
        Eq_surface['iso_r'] = iso_r
        return Eq_surface

    def query_rays_f(self, r, which=0):
        '''
        Query the parameter value at a given radius along the rays.

        Parameters
        ----------
        r : float
            The radius at which to query the parameter value.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        np.ndarray
            The parameter values at the given radius.
        '''
        if which > 0:
            return np.array([i.upper(r, inv=False) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(r, inv=False) for i in self.rays_func])

        return np.array([i(r, inv=False) for i in self.rays_func]).T

    def query_rays_r(self, f, which=0):
        '''
        Query the radius for a given parameter value along the rays.

        Parameters
        ----------
        f : float
            The parameter value for which to query the radius.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        np.ndarray
            The radii corresponding to the given parameter value.
        '''
        if which > 0:
            return np.array([i.upper(f, inv=True) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(f, inv=True) for i in self.rays_func])

        return np.array([i(f, inv=True) for i in self.rays_func]).T

    def query_iso_f(self, r, which=0):
        '''
        Query the parameter value at a given radius for the isoprofile.

        Parameters
        ----------
        r : float
            The radius at which to query the parameter value.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        float
            The parameter value at the given radius.
        '''
        if which > 0:
            return self.iso_pro_func.upper(r, inv=False)
        if which < 0:
            return self.iso_pro_func.lower(r, inv=False)

        return self.iso_pro_func(r, inv=False)

    def query_iso_r(self, f, which=0):
        '''
        Query the radius for a given parameter value for the isoprofile.

        Parameters
        ----------
        f : float
            The parameter value for which to query the radius.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        float
            The radius corresponding to the given parameter value.
        '''
        if which > 0:
            return self.iso_pro_func.upper(f, inv=True)
        if which < 0:
            return self.iso_pro_func.lower(f, inv=True)

        return self.iso_pro_func(f, inv=True)

    @staticmethod
    def boundary_registry(fn: str | Callable) -> Callable:
        """Function decorator to define a new bound method.

        @SphField.boundary_registry
        def bound_dist(cls, value, **kwargs):
            return value*np.ones(cls.rays.num)

        """
        if callable(fn):
            SphField._bound_method[fn.__name__] = fn
            return fn

        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                SphField._bound_method[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator

    @staticmethod
    def step_registry(fn: str | Callable) -> Callable:
        """Function decorator to define a new step method.

        @SphField.step_registry
        def bound_dist(cls, value, **kwargs):
            return value*np.ones(cls.rays.num)

        """
        if callable(fn):
            SphField._step_method[fn.__name__] = fn
            return fn

        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                SphField._step_method[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator

    @staticmethod
    def iso_registry(fn: Callable) -> Callable:

        if callable(fn):
            SphField._iso_method[fn.__name__] = fn
            return fn
        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                SphField._iso_method[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator

from gal3d import config
if config['general']['use_cython']:
    from .util_cy import iso_profile_by_moi, iso_profile_by_pair
else:
    from .util_nb import iso_profile_by_moi, iso_profile_by_pair
    
SphField.iso_registry('moi')(iso_profile_by_moi)
SphField.iso_registry('pair')(iso_profile_by_pair)


@SphField.boundary_registry('dist')
def bound_dist(cls, value, **kwargs):
    '''
    Calculate the boundary based on a fixed distance.

    Parameters
    ----------
    value : float
        The distance value.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The boundary values.
    '''
    return value * np.ones(cls.rays.num)


@SphField.boundary_registry('pct')
def bound_pct(cls, value, **kwargs):
    '''
    Calculate the boundary based on a percentile.

    Parameters
    ----------
    value : float
        The percentile value.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The boundary values.
    '''
    return np.array([np.percentile(cls.r_ray_n(i), value) for i in range(cls.rays.num)])


@SphField.boundary_registry('value')
def bound_value(cls, value, mode='max', **kwargs):
    '''
    Calculate the boundary based on a parameter value.

    Parameters
    ----------
    value : float
        The parameter value.
    mode : str, optional
        The mode for calculating the boundary. Options are 'max' or 'min'. Default is 'max'.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The boundary values.

    Raises
    ------
    ValueError
        If the mode is not 'max' or 'min'.
    '''
    # np.array([np.max(Base.r_ray_n(i)[Base.parameter_ray_n(i)>value]) for i in range(Base.rays.num)])
    if mode == 'max':
        return np.array(
            [
                np.max(cls.r_ray_n(i)[cls.parameter_ray_n(i) > value])
                for i in range(cls.rays.num)
            ]
        )
    if mode == 'min':
        return np.array(
            [
                np.min(cls.r_ray_n(i)[cls.parameter_ray_n(i) < value])
                for i in range(cls.rays.num)
            ]
        )
    raise ValueError(f"{mode} is not a valid value. Only 'max' and 'min' are valid.")


@SphField.step_registry('lin')
def step_lin(cls, num_p):
    '''
    Generate linearly spaced points between the inner and outer boundaries.

    Parameters
    ----------
    num_p : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The generated points.
    '''
    return np.linspace(cls.inner_r, cls.outer_r, num_p).T


@SphField.step_registry('log')
def step_log(cls, num_p):
    '''
    Generate logarithmically spaced points between the inner and outer boundaries.

    Parameters
    ----------
    num_p : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The generated points.
    '''
    return np.geomspace(cls.inner_r, cls.outer_r, num_p).T
