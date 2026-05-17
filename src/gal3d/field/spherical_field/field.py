import logging
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from gal3d.density import DensitySource
from gal3d.point import Particles
from gal3d.util.func_decorator import timer

from .ray import MonotonRay
from .spherical_vector import SphVector

logger = logging.getLogger("gal3d.field.SphField")


timing = timer(logger)


class SphField:
    _bound_method: dict[str, Callable] = {}
    _step_method: dict[str, Callable] = {}
    _iso_method: dict[str, Callable] = {}

    def __init__(
        self, density_source: DensitySource, num_ray: int = 1024, ray_method: str = "fibonacci", **kwargs: Any
    ):
        """
        Initialize the Field class for calculating parameters at various distances in a 3D galaxy model.

        Parameters
        ----------
        density_source : DensitySource
            Any instance of DensitySource, which provides the density estimation at arbitrary positions.(e.g. Particles or TheoreticalDensityDistribution)

        num_ray : int, optional
            The number of rays to generate. Default is 1024.

        ray_method : str, optional
            {'fibonacci', 'muller'}, the method used to generate unit ray vectors. Default is 'fibonacci'.
        """

        self._build_ray_vector(density_source, num_ray, ray_method)

    def __repr__(self):
        info = [f"SphField(num_ray={getattr(self.rays, 'num', 'N/A')})", f"[{repr(self.density_source)}]"]
        if hasattr(self, "inner_r") and hasattr(self, "outer_r"):
            info.append(f"inner_r=[{np.min(self.inner_r):.3g}, {np.max(self.inner_r):.3g}]")
            info.append(f"outer_r=[{np.min(self.outer_r):.3g}, {np.max(self.outer_r):.3g}]")
        return "<" + ", ".join(info) + ">"

    @timing
    def _build_ray_vector(self, density_source: DensitySource, num_ray: int, ray_method: str) -> "SphField":
        self.rays = SphVector(num_ray, ray_method)
        self.density_source = density_source
        return self

    def _require_particles(self) -> Particles:
        """
        Return density_source as Particles, or raise if the source is continuous.
        """
        if not isinstance(self.density_source, Particles):
            raise TypeError(
                "This method requires discrete particle data (Particles with pos/r). "
                "For continuous theoretical models, use boundary mode 'value' or 'dist'."
            )
        return self.density_source

    def _assign_ray_points(self):
        particles = self._require_particles()
        self.rays_index = self.rays.assign_points(particles.pos)
        self.rays_points_num = np.bincount(self.rays_index)

        max_num_dex = np.argmax(self.rays_points_num)
        logger.debug("Ray %d has the maximum particle count of %d.", max_num_dex, self.rays_points_num[max_num_dex])

        min_num_dex = np.argmin(self.rays_points_num)
        logger.debug("Ray %d has the minimum particle count of %d.", min_num_dex, self.rays_points_num[min_num_dex])
        if self.rays_points_num[min_num_dex] < 3:
            logger.error("It should be > 2, so please make the ray num smaller. ")

        ind: list[list[int]] = [[] for _ in range(self.rays.num)]
        for i, j in enumerate(self.rays_index):
            ind[j].append(i)

        self.points_index = [np.array(i) for i in ind]

    @timing
    def build_field_boundary(
        self, inner: float = 0.5, outer: float = 95, inner_mode: str = "dist", outer_mode: str = "pct"
    ) -> "SphField":
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

        self.rays_vect = self.rays.pos

        self.inner_r = self._bound_method[inner_mode](self, inner, mode="min")
        r_in_min = np.min(self.inner_r)
        r_in_max = np.max(self.inner_r)
        logger.debug("Field inner boundaries range from %.2f to %.2f", r_in_min, r_in_max)
        if r_in_min / r_in_max < 0.09:
            logger.warning(
                "The axial ratio of the inner boundary shape is quite extreme. Consider limiting the particles or refining the boundary."
            )

        self.outer_r = self._bound_method[outer_mode](self, outer, mode="max")
        r_ou_min = np.min(self.outer_r)
        r_ou_max = np.max(self.outer_r)
        logger.debug("Field outer boundaries range from %.2f to %.2f", r_ou_min, r_ou_max)
        if r_ou_min / r_ou_max < 0.09:
            logger.warning(
                "The axial ratio of the outer boundary shape is quite extreme. Consider limiting the particles or refining the boundary."
            )

        self.check_boundary()
        return self

    @timing
    def build_profile_sample(self, num_p: int = 500, step_mode: str = "log") -> "SphField":
        """
        Build a sample of points along the rays for parameter calculation.

        Parameters
        ----------
        base : Local_est
            An instance of the Local_est class used to compute parameters.
        num_p : int, optional
            The number of points to sample along each ray. Default is 500.
        step_mode : str, optional
            The mode for spacing the points. Options are 'lin' for linear spacing or 'log' for logarithmic spacing. Default is 'log'.
        """

        self.points_r = self._step_method[step_mode](self, num_p)
        self.points_pos = np.einsum("ij,ik->ijk", self.points_r, self.rays.pos)

        points_que = self.points_pos.reshape(self.points_pos.shape[0] * self.points_pos.shape[1], 3)
        self.points_parameter = self.density_source(points_que).reshape(self.points_r.shape)
        return self

    @timing
    def build_profile_interpolator(
        self,
        interpolator_method: str = "LU",
        is_decreasing: bool = True,
        interpolator_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> "SphField":
        """
        Build interpolators for the sampled points.

        Parameters
        ----------
        interpolator_method : str, optional
            The method used for interpolation. Default is 'LU'.
        is_decreasing : bool, optional
            Whether to use density estimation. Default is True.
        interpolator_kwargs : dict, optional
            Additional keyword arguments for the interpolator. Default is an empty dictionary.
        **kwargs : dict
            Additional keyword arguments.
        """
        if interpolator_kwargs is None:
            interpolator_kwargs = {}
        self.rays_func = [
            MonotonRay(
                self.points_r[i],
                self.points_parameter[i],
                is_decreasing=is_decreasing,
                interpolator_method=interpolator_method,
                interpolator_kwargs=interpolator_kwargs,
                **kwargs,
            )
            for i in range(len(self.points_parameter))
        ]

        return self

    @timing
    def build_isodensity_profile(
        self, method: str = "pair", from_rays_func: bool = False, res_b: float = 0.2, res_c: float = 0.1, **kwargs: Any
    ) -> "SphField":
        """
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
        """

        self.set_isodensity_sphere(from_rays_func=from_rays_func, **kwargs)

        self.iso_pro_parameter = self._iso_method[method](self.rays_vect, self.iso_parameters, res_b, res_c)
        interpolator_method = kwargs.get("interpolator_method", "LU")

        self.iso_pro_func = MonotonRay(self.iso_pro_r, self.iso_pro_parameter, interpolator_method=interpolator_method)

        return self

    @timing
    def set_isodensity_sphere(self, from_rays_func: bool = False, **kwargs: Any) -> "SphField":
        """
        Set the isosphere for the galaxy model.

        Parameters
        ----------
        from_rays_func : bool, optional
            Whether to use the ray functions for setting the isosphere. Default is False.
        **kwargs : dict
            Additional keyword arguments.
        """
        num_p = kwargs.get("num_p", self.points_r.shape[1])

        if from_rays_func:
            self.iso_pro_r = np.geomspace(np.max(self.inner_r), np.min(self.outer_r), num_p)
            self.iso_points = np.einsum("ij,k->ikj", self.rays_vect, self.iso_pro_r)
            self.iso_parameters = np.array([self.rays_func[i](self.iso_pro_r) for i in range(len(self.rays_func))])

        else:
            self.iso_pro_r = np.geomspace(np.percentile(self.inner_r, 50), np.percentile(self.outer_r, 50), num_p)
            self.iso_points = np.einsum("ij,k->ikj", self.rays_vect, self.iso_pro_r)

            self.iso_parameters = self.density_source(
                self.iso_points.reshape(self.iso_points.shape[0] * self.iso_points.shape[1], 3)
            ).reshape((self.iso_points.shape[0], self.iso_points.shape[1]))

        return self

    def check_boundary(self):
        """
        Check if the outer boundaries are greater than the inner boundaries.

        Raises
        ------
        ValueError
            If any outer boundary is not greater than the corresponding inner boundary.
        """
        if not all(self.outer_r > self.inner_r):
            ind = np.arange(len(self.outer_r))
            ind = ind[(self.outer_r < self.inner_r)]
            logger.error("The outer boundaries need to be greater than the inner boundaries. Check Ray %d", ind)
            raise ValueError("The outer boundaries need to be greater than the inner boundaries")

    def pos_ray_n(self, n: int) -> np.ndarray:
        """
        Retrieve the positions of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The coordinates of the points closest to the nth ray.
        """
        if not hasattr(self, "points_index"):
            self._assign_ray_points()
        particles = self._require_particles()
        return particles.pos[self.points_index[n]]

    def r_ray_n(self, n: int) -> np.ndarray:
        """
        Retrieve the radii of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The radii of the points closest to the nth ray.
        """
        if not hasattr(self, "points_index"):
            self._assign_ray_points()
        particles = self._require_particles()
        return particles.r[self.points_index[n]]

    def mass_ray_n(self, n: int) -> np.ndarray:
        """
        Retrieve the mass of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The mass of the points closest to the nth ray.
        """
        if not hasattr(self, "points_index"):
            self._assign_ray_points()
        particles = self._require_particles()
        return particles.mass[self.points_index[n]]

    def parameter_ray_n(self, n: int) -> np.ndarray:
        """
        Retrieve the parameters of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The parameters of the points closest to the nth ray.
        """
        if not hasattr(self, "points_index"):
            self._assign_ray_points()
        particles = self._require_particles()
        return particles.parameter[self.points_index[n]]

    def gradient_ray_n(self, n: int) -> np.ndarray:
        """
        Retrieve the gradients of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        gradient: np.ndarray
            The gradients of the points closest to the nth ray.
        """
        if not hasattr(self, "points_index"):
            self._assign_ray_points()
        particles = self._require_particles()
        return particles.gradient[self.points_index[n]]

    def generate(self, r: Iterable[float] | float, for_fit: bool = False, **kwargs: Any) -> dict:
        """
        Generate the equivalent surface for a given radius.

        Parameters
        ----------
        r : Iterable | float
            The radius or radii for which to generate the equivalent surface.
        for_fit : bool, optional
            Whether the generated surface is for fitting purposes. Default is False.

        Returns
        -------
        dict
            A dictionary containing the positions, parameters, and radii of the equivalent surface.
        """

        level = kwargs.get("level", (0, 0))
        rval: np.ndarray | float
        if isinstance(r, Iterable):
            rval = np.array(r)
        else:
            rval = r

        ftarget = self.query_iso_f(rval, which=level[0])

        rtarget = self.query_rays_r(ftarget, which=level[1])  # shape (len(ftarget),num_rays)

        if isinstance(r, Iterable):
            target_pos = np.einsum("ji,ik->jik", rtarget, self.rays_vect)  # (len(ftarget),num_rays) * (num_rays , 3)
        else:
            target_pos = np.einsum("i,ik->ik", rtarget, self.rays_vect)

        Eq_surface = {}
        Eq_surface["pos"] = target_pos
        Eq_surface["parameter"] = ftarget
        Eq_surface["r"] = rtarget
        if for_fit:
            Eq_surface["pos"] = Eq_surface["pos"][~np.isnan(Eq_surface["r"])]
            Eq_surface["r"] = Eq_surface["r"][~np.isnan(Eq_surface["r"])]
            # Eq_surface["r"] = Eq_surface["r"] / np.sqrt(
            #    np.sum(Eq_surface["r"] ** 2) / len(Eq_surface["r"])
            # )  #  normalization as this used for calculate error
            Eq_surface["info"] = {"parameter": Eq_surface["parameter"]}

        return Eq_surface

    def generate_by_f(self, f: Iterable[float] | float, **kwargs: Any) -> dict:
        """
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
        """
        level = kwargs.get("level", (0, 0))
        fval: np.ndarray | float
        if isinstance(f, Iterable):
            fval = np.array(f)
        else:
            fval = f

        rtarget = self.query_rays_r(fval, which=level[1])

        if isinstance(f, Iterable):
            target_pos = np.einsum("ji,ik->jik", rtarget, self.rays_vect)
        else:
            target_pos = np.einsum("i,ik->ik", rtarget, self.rays_vect)

        iso_r = self.query_iso_r(fval, which=level[0])

        Eq_surface = {}
        Eq_surface["pos"] = target_pos
        Eq_surface["r"] = rtarget
        Eq_surface["iso_r"] = iso_r
        return Eq_surface

    def query_rays_f(self, r: float | np.ndarray, which: int = 0) -> np.ndarray:
        """
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
        """
        if which > 0:
            return np.array([i.upper(r, inv=False) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(r, inv=False) for i in self.rays_func])

        return np.array([i(r, inv=False) for i in self.rays_func]).T

    def query_rays_r(self, f: float | np.ndarray, which: int = 0) -> np.ndarray:
        """
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
        """
        if which > 0:
            return np.array([i.upper(f, inv=True) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(f, inv=True) for i in self.rays_func])

        return np.array([i(f, inv=True) for i in self.rays_func]).T

    def query_iso_f(self, r: float | np.ndarray, which: int = 0) -> np.ndarray | float:
        """
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
        """
        if which > 0:
            return self.iso_pro_func.upper(r, inv=False)
        if which < 0:
            return self.iso_pro_func.lower(r, inv=False)

        return self.iso_pro_func(r, inv=False)

    def query_iso_r(self, f: float | np.ndarray, which: int = 0) -> np.ndarray | float:
        """
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
        """
        if which > 0:
            return self.iso_pro_func.upper(f, inv=True)
        if which < 0:
            return self.iso_pro_func.lower(f, inv=True)

        return self.iso_pro_func(f, inv=True)

    @staticmethod
    def boundary_registry(fn: str | Callable) -> Callable:
        """Function decorator to define a new bound method.

        Examples
        --------
        >>> @SphField.boundary_registry
        >>> def bound_dist(cls, value, **kwargs):
        >>>     return value*np.ones(cls.rays.num)
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

        Examples
        --------
        >>> @SphField.step_registry
        ... def step_lin(cls, num_p, **kwargs):
        ...     return np.linspace(cls.inner_r, cls.outer_r, num_p).T
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
    def iso_registry(fn: str | Callable) -> Callable:
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


from .util_cy import iso_profile_by_moi, iso_profile_by_pair

SphField.iso_registry("moi")(iso_profile_by_moi)
SphField.iso_registry("pair")(iso_profile_by_pair)


@SphField.boundary_registry("dist")
def bound_dist(self: SphField, value: float, **kwargs: Any) -> np.ndarray:
    """
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
    """
    return value * np.ones(self.rays.num)


@SphField.boundary_registry("pct")
def bound_pct(self: SphField, value: float, **kwargs: Any) -> np.ndarray:
    """
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
    """
    return np.array([np.percentile(self.r_ray_n(i), value) for i in range(self.rays.num)])


@SphField.boundary_registry("value")
def bound_value(self: SphField, value: float, **kwargs: Any) -> np.ndarray:
    """
    Finds the boundary radius for each ray where the parameter value crosses the specified value.
    For density-like parameters that decrease with radius, this function will:
    - For inner boundary (mode="min"): Search inward until parameter value exceeds target
    - For outer boundary (mode="max"): Search outward until parameter value falls below target

    Parameters
    ----------
    self : SphField
        The SphField instance containing ray and particle information.
    value : float
        The target parameter value to find along each ray.
    **kwargs : dict
        Additional keyword arguments:
        - mode : str, optional
            "min" for inner boundary (search inward),
            "max" for outer boundary (search outward).
            Default is "max".

    Returns
    -------
    np.ndarray
        Array of radii for each ray where the parameter value crosses the specified boundary.
    """
    rays_vect = self.rays_vect
    mode = kwargs.get("mode", "max")

    iter_max = 100
    step_fraction = 0.2

    # determine search direction
    is_inner = mode == "min"
    r10, r90 = (
        np.percentile(self.density_source.r, [10, 90]) if hasattr(self.density_source, "r") else (1.0, 30.0)
    )  # fallback values if particle radii are not available
    radius_guess = r90 * np.ones(len(self.rays_vect)) if is_inner else r10 * np.ones(len(self.rays_vect))
    eps_radius = r10 / 100 if is_inner else r90 / 100

    active_mask = np.ones_like(radius_guess, dtype=bool)  # Only iterate over unconverged rays

    # Coarse search to bracket the boundary
    for _i in range(iter_max):
        param_val = np.zeros_like(radius_guess)
        param_val[active_mask] = self.density_source(radius_guess[active_mask, None] * rays_vect[active_mask])
        iter_step = step_fraction * radius_guess

        if is_inner:
            condition = param_val > value  # inner boundary: param_val > value
            active_mask[condition] = False  # stop iterating over converged rays
            # otherwise, search inward
            radius_guess[active_mask] -= iter_step[active_mask]
        else:
            condition = param_val < value  # outer boundary: param_val < value
            active_mask[condition] = False  # stop iterating over converged rays
            # otherwise, search outward
            radius_guess[active_mask] += iter_step[active_mask]

        if not np.any(active_mask):
            break

    search_type = "Inner" if is_inner else "Outer"
    if _i >= iter_max - 1:
        logger.error("[Coarse Search] %s boundary search did not converge after %d iterations", search_type, iter_max)
    else:
        logger.debug("[Coarse Search] %s boundary converged in %d iterations", search_type, _i + 1)

    # Set the upper and lower bounds for bisection search
    radius_upper = radius_guess * (1 + step_fraction) if is_inner else radius_guess.copy()
    radius_lower = radius_guess.copy() if is_inner else radius_guess / (1 + step_fraction)

    active_mask = np.ones_like(radius_guess, dtype=bool)

    # Bisection search for refinement
    for _i in range(iter_max):
        radius_mid = (radius_lower[active_mask] + radius_upper[active_mask]) / 2
        pos_mid = radius_mid[:, None] * rays_vect[active_mask]
        param_val_mid = self.density_source(pos_mid)

        if is_inner:
            # inner boundary: param_val > value, update lower boundary
            mask = param_val_mid > value
            radius_lower[active_mask] = np.where(mask, radius_mid, radius_lower[active_mask])
            radius_upper[active_mask] = np.where(~mask, radius_mid, radius_upper[active_mask])
        else:
            # outer boundary: param_val < value, update upper boundary
            mask = param_val_mid < value
            radius_upper[active_mask] = np.where(mask, radius_mid, radius_upper[active_mask])
            radius_lower[active_mask] = np.where(~mask, radius_mid, radius_lower[active_mask])

        converged = np.abs(radius_lower - radius_upper) < eps_radius
        if np.all(converged):
            break

        active_mask = ~converged

    if _i >= iter_max - 1:
        logger.error("[Bisection Search] %s boundary iteration exceeded maximum limit: %d", search_type, iter_max)
    else:
        logger.debug("[Bisection Search] Found target %s radius in %d iterations", search_type, _i + 1)

    return (radius_lower + radius_upper) / 2


@SphField.step_registry("lin")
def step_lin(self: SphField, num_p: int) -> np.ndarray:
    """
    Generate linearly spaced points between the inner and outer boundaries.

    Parameters
    ----------
    num_p : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The generated points.
    """
    return np.linspace(self.inner_r, self.outer_r, num_p, endpoint=True).T


@SphField.step_registry("log")
def step_log(self: SphField, num_p: int) -> np.ndarray:
    """
    Generate logarithmically spaced points between the inner and outer boundaries.

    Parameters
    ----------
    num_p : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The generated points.
    """
    return np.geomspace(self.inner_r, self.outer_r, num_p).T
