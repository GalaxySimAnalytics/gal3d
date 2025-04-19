from typing import Callable, Self, Tuple, Dict
from types import MethodType

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Geometry, GeometryBase, Parameters
from .coordinate import CoordinateBase, Coordinate
from .minimize_func import MinimizeFunc
from ..util.func_signature import func_required_key
from ..field.spherical_field.spherical_vector import fibonacci_sampling


class Structure3D:

    _compute_error_method: Dict[str, Callable] = {}

    def __init__(
        self,
        coordinate: CoordinateBase | str,
        geometry: GeometryBase | str,
        error_func: Callable | str,
        error_method: Callable | str,
    ):

        self._set_geometry(geometry)
        self._set_coordinate(coordinate)
        self._set_error_func(error_func)
        self._set_error_method(error_method)

        self.parameters = (
            self._coordinate.get_parameters() + self._geometry.get_parameters()
        )

    def init_parameters(self, *args, **kwargs) -> Parameters:
        if args:
            params = dict(zip(self.parameters.keys(), *args))
            return self._coordinate.init_parameters(
                **params
            ) + self._geometry.init_parameters(**params)
        if kwargs:
            return self._coordinate.init_parameters(
                **kwargs
            ) + self._geometry.init_parameters(**kwargs)

        return self._coordinate.get_parameters() + self._geometry.get_parameters()

    def set_parameters(self, *args, **kwargs):
        '''
        Sets the parameters for the structure.

        Parameters
        ----------
        *args : tuple
            Positional arguments to set parameters.
        **kwargs : dict
            Keyword arguments to set parameters.
        '''
        self.parameters = self.parameters + self.init_parameters(*args, **kwargs)

    def from_parameters(self, *args, **kwargs) -> Self:
        '''
        Creates a new Structure3D instance with the given parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments to initialize parameters.
        **kwargs : dict
            Keyword arguments to initialize parameters.

        Returns
        -------
        Structure3D
            A new Structure3D instance with the given parameters.
        '''
        ret = Structure3D(
            coordinate=self._coordinate,
            geometry=self._geometry,
            error_func=self._error_func,
            error_method=self._error_method,
        )
        ret.set_parameters(*args, **kwargs)
        return ret

    def _set_geometry(self, geometry: GeometryBase | str):

        assert (
            isinstance(geometry, GeometryBase)
            or isinstance(geometry, str)
            or issubclass(geometry, GeometryBase)
        )
        self._geometry = (
            Geometry.get_plugin(geometry) if isinstance(geometry, str) else geometry
        )
        self._geometry_name = self._geometry.__name__

    def _set_coordinate(self, coordinate: CoordinateBase | str):

        assert (
            isinstance(coordinate, CoordinateBase)
            or isinstance(coordinate, str)
            or issubclass(coordinate, CoordinateBase)
        )
        self._coordinate = (
            Coordinate.get_plugin(coordinate)
            if isinstance(coordinate, str)
            else coordinate
        )
        self._coordinate_name = self._coordinate.__name__
        self.__coor_pa_num = len(self._coordinate.PN)

    def _set_error_func(self, error_func: Callable | str):

        assert callable(error_func) or isinstance(error_func, str)
        self._error_func = (
            MinimizeFunc.minimize_fn[error_func]
            if isinstance(error_func, str)
            else error_func
        )
        self._error_func_name = self._error_func.__name__

        self._error_params = list(func_required_key(self._error_func).keys())
        self._error_params = list(
            filter(
                lambda x: ('_call' not in x) and ('_call' not in x), self._error_params
            )
        )

    def _set_error_method(self, error_method: Callable | str):

        assert callable(error_method) or isinstance(error_method, str)

        self._error_method = (
            MethodType(self._compute_error_method[error_method], self)
            if isinstance(error_method, str)
            else error_method
        )
        self._error_method_name = self._error_method.__name__

    def __repr__(self):
        '''
        Returns a string representation of the Structure_3D object.

        Returns
        -------
        str
            String representation of the object.
        '''
        coor_repr = repr(self._coordinate(**self.parameters))
        geometry_repr = repr(self._geometry(**self.parameters))
        lin1 = [f"<{self.__class__.__name__}|: ", '\n']
        lin2 = ["   ", coor_repr, '\n']
        lin3 = ["   ", geometry_repr]
        return ''.join(lin1 + lin2 + lin3)

    def __eq__(self, other):
        '''
        Checks equality with another Structure3D object.

        Parameters
        ----------
        other : Structure3D
            Another Structure3D object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        '''

        if (
            isinstance(other, Structure3D)
            and (self._coordinate_name == other._coordinate_name)
            and (self._geometry_name == other._geometry_name)
            and (self._error_func_name == other._error_func_name)
            and (self._error_method_name == other._error_method_name)
        ):
            return True

        return False

    def __call__(self, pos, **kwargs):
        '''
        Evaluates the structure at the given position.

        Parameters
        ----------
        pos : array_like
            The position at which to evaluate the structure.
        **kwargs : dict
            Additional keyword arguments for the evaluation.

        Returns
        -------
        array_like
            The evaluated structure at the given position.
        '''
        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_normal(**kwargs)

        return self._geometry(**geoty_pa)(self._coordinate(**coord_pa)(pos))

    def f_ray_d(self, pos: ArrayLike, **kwargs) -> np.ndarray:

        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_normal(**kwargs)

        return self._geometry(**geoty_pa).f_ray_d(self._coordinate(**coord_pa)(pos))

    def ray_intersect(self, pos, **kwargs) -> tuple:
        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_normal(**kwargs)

        return self._geometry(**geoty_pa).ray_intersect(
            self._coordinate(**coord_pa)(pos)
        )

    def line_intersect(self, pos1: ArrayLike, pos2: ArrayLike, **kwargs) -> np.ndarray:

        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)

        coord_pa, geoty_pa = self._generate_normal(**kwargs)

        return self._geometry(**geoty_pa).line_intersect(
            pos1=self._coordinate(**coord_pa)(pos1),
            pos2=self._coordinate(**coord_pa)(pos2),
        )

    def quick_call(self, *args, pos, **kwargs) -> np.ndarray:
        '''
        Quick evaluation of the structure at the given position.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the quick evaluation.
        pos : array_like
            The position at which to evaluate the structure.
        **kwargs : dict
            Additional keyword arguments for the evaluation.

        Returns
        -------
        array_like
            The quick evaluated structure at the given position.

        Raises
        ------
        KeyError
            If parameters are not provided.
        '''
        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_quick(*args, **kwargs)

        return self._geometry.quick_call(
            **geoty_pa, pos=self._coordinate.quick_call(**coord_pa, pos=pos)
        )

    def quick_f_ray_d(self, *args, pos, **kwargs) -> np.ndarray:

        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_quick(*args, **kwargs)

        return self._geometry.quick_f_ray_d(
            **geoty_pa, pos=self._coordinate.quick_call(**coord_pa, pos=pos)
        )

    def quick_ray_dist(self, *args, pos, **kwargs) -> np.ndarray:

        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_quick(*args, **kwargs)

        return self._geometry.quick_ray_dist(
            **geoty_pa, pos=self._coordinate.quick_call(**coord_pa, pos=pos)
        )

    def quick_line_intersect(self, *args, pos1, pos2, **kwargs) -> np.ndarray:

        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)

        coord_pa, geoty_pa = self._generate_quick(**kwargs)

        return self._geometry.quick_line_intersect(
            **geoty_pa,
            pos1=self._coordinate(**coord_pa)(pos1),
            pos2=self._coordinate(**coord_pa)(pos2),
        )

    def _generate_normal(self, **kwargs) -> Tuple[dict, dict]:
        coord_pa = (
            self._coordinate.init_parameters(**kwargs) if kwargs else self.parameters
        )
        geoty_pa = (
            self._geometry.init_parameters(**kwargs) if kwargs else self.parameters
        )

        return coord_pa, geoty_pa

    def _generate_quick(self, *args, **kwargs) -> Tuple[dict, dict]:
        if args:
            args = args[0] if len(args) == 1 else args
            coord_pa = dict(zip(self._coordinate.PN, args[: self.__coor_pa_num]))
            geoty_pa = dict(zip(self._geometry.PN, args[self.__coor_pa_num :]))
            return coord_pa, geoty_pa

        if kwargs:
            try:
                coord_pa = {i: kwargs[i] for i in self._coordinate.PN}
                geoty_pa = {i: kwargs[i] for i in self._geometry.PN}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                geoty_parameters = self._geometry.init_parameters(**kwargs)
                coord_pa = {i: coord_parameters[i] for i in self._coordinate.PN}
                geoty_pa = {i: geoty_parameters[i] for i in self._geometry.PN}
            return coord_pa, geoty_pa

        coord_pa = {i: self.parameters[i] for i in self._coordinate.PN}
        geoty_pa = {i: self.parameters[i] for i in self._geometry.PN}
        return coord_pa, geoty_pa

    def generate_points(self, random_np: int = 1024):
        '''
        Generates random points on the surface of the structure.

        Parameters
        ----------
        random_np : int, optional
            Number of random points to generate (default is 1024).

        Returns
        -------
        array_like
            Generated points on the surface of the structure.
        '''
        cpos, spos = fibonacci_sampling(random_np)

        coord_pa, geoty_pa = self._generate_quick()

        points = self._geometry(**geoty_pa).ray_point(cpos)

        return self._coordinate(**coord_pa).inverse(points)

    @staticmethod
    def compute_method_registry(fn: str | Callable) -> Callable:

        if callable(fn):
            Structure3D._compute_error_method[fn.__name__] = fn
            return fn
        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                Structure3D._compute_error_method[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator


@Structure3D.compute_method_registry
def isodensity_fcall(self: Structure3D, params: tuple | ArrayLike, **kwargs) -> float:

    f_call = self.quick_call(*params, pos=kwargs['pos']) - 1.0
    error_pa = {i: kwargs[i] for i in self._error_params}

    return self._error_func(f_call, **error_pa)


@Structure3D.compute_method_registry
def isodensity_dcall(
    self: Structure3D, params: tuple | ArrayLike, *args, **kwargs
) -> float:

    f_call = self.quick_f_ray_d(*params, pos=kwargs['pos']) - 1.0
    error_pa = {i: kwargs[i] for i in self._error_params}

    return self._error_func(f_call, **error_pa)


@Structure3D.compute_method_registry
def isodensity_dist(self: Structure3D, params: tuple | ArrayLike, **kwargs) -> float:

    f_call = self.quick_ray_dist(*params, pos=kwargs['pos'])
    error_pa = {i: kwargs[i] for i in self._error_params}

    return self._error_func(f_call, **error_pa)


@Structure3D.compute_method_registry
def shell_padev(self: Structure3D, params: tuple | ArrayLike, *args, **kwargs) -> float:
    """
    Calculate the error using the shell error function.

    Parameters
    ----------
    params : list or dict
        The parameters to be used in the error calculation.
    kwargs : dict
        Additional keyword arguments required by the error function.

    Returns
    -------
    float
        The calculated error value.
    """
    if not isinstance(params, dict):
        params = dict(zip(self.parameters.keys(), params))
    params1 = params.copy()
    params1['a'] = 0.98 * params1['a']
    params2 = params.copy()
    params2['a'] = 1.02 * params2['a']
    f_call1 = self.quick_call(pos=kwargs['pos'], **params1) - 1
    f_call2 = self.quick_call(pos=kwargs['pos'], **params2) - 1
    error_pa = {i: kwargs[i] for i in self._error_params}
    return self._error_func(f_call1=f_call1, f_call2=f_call2, **error_pa)


@Structure3D.compute_method_registry
def grid_padev(self: Structure3D, params: tuple | ArrayLike, *args, **kwargs) -> float:
    """
    Calculate the error using the grid error function.

    Parameters
    ----------
    params : list or dict
        The parameters to be used in the error calculation.
    kwargs : dict
        Additional keyword arguments required by the error function.

    Returns
    -------
    float
        The calculated error value.
    """
    if not isinstance(params, dict):
        params = dict(zip(self.parameters.keys(), params))
    params1 = params.copy()
    params1['a'] = 0.9 * params1['a']
    params2 = params.copy()
    params2['a'] = 1.1 * params2['a']
    f_call1 = self.quick_call(pos=kwargs['pos'], **params1) - 1
    f_call2 = self.quick_call(pos=kwargs['pos'], **params2) - 1
    error_pa = {i: kwargs[i] for i in self._error_params}

    return self._error_func(f_call1=f_call1, f_call2=f_call2, **error_pa)
