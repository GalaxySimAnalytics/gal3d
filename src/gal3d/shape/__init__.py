import logging
from types import MethodType
from typing import Callable, Dict, Self, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ..field.spherical_field.spherical_vector import fibonacci_sampling
from ..util.func_signature import func_required_key
from .coordinate import Coordinate, CoordinateBase
from .geometry import Geometry, GeometryBase, Parameters
from .minimize_func import MinimizeFunc

logger = logging.getLogger("gal3d.shape")

class Structure3D:
    """
    A 3D structure composed of a coordinate transformation and a geometry definition.
    
    This class combines a coordinate transformation (via `CoordinateBase`) and a 
    shape definition (via `GeometryBase`). It also provides an error function from 
    `MinimizeFunc` and an associated error evaluation method.
    """

    _compute_error_method: Dict[str, Callable] = {}

    def __init__(
        self,
        coordinate: CoordinateBase | str,
        geometry: GeometryBase | str,
        error_func: Callable | str,
        error_method: Callable | str,
    ):
        """
        Initialize a Structure3D instance.

        Parameters
        ----------
        coordinate : CoordinateBase or str
            Coordinate transformation class or registered plugin name.
        geometry : GeometryBase or str
            Geometry class or registered plugin name.
        error_func : Callable or str
            Error function or its name in `MinimizeFunc`.
        error_method : Callable or str
            Method for computing the error or its name in `_compute_error_method`.
        """

        self._set_geometry(geometry)
        self._set_coordinate(coordinate)
        self._set_error_func(error_func)
        self._set_error_method(error_method)

        self.parameters = (
            self._coordinate.get_parameters() + self._geometry.get_parameters()
        )

    def init_parameters(self, *args, **kwargs) -> Parameters:
        """
        Initialize parameters for coordinate and geometry.

        Parameters
        ----------
        *args : tuple
            Positional values to initialize parameters.
        **kwargs : dict
            Keyword arguments to initialize parameters.

        Returns
        -------
        Parameters
            Combined parameters from coordinate and geometry.
        """
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
        """
        Set the structure's parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments for parameters.
        **kwargs : dict
            Keyword arguments for parameters.
        """
        self.parameters = self.parameters + self.init_parameters(*args, **kwargs)

    def from_parameters(self, *args, **kwargs) -> Self:
        """
        Create a new Structure3D with given parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments for initialization.
        **kwargs : dict
            Keyword arguments for initialization.

        Returns
        -------
        Structure3D
            A new instance with specified parameters.
        """
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
        """
        Return string representation of Structure3D.

        Returns
        -------
        str
            Human-readable string of the coordinate and geometry.
        """
        coor_repr = repr(self._coordinate(**self.parameters))
        geometry_repr = repr(self._geometry(**self.parameters))
        lin1 = [f"<{self.__class__.__name__}|: ", '\n']
        lin2 = ["   ", coor_repr, '\n']
        lin3 = ["   ", geometry_repr]
        return ''.join(lin1 + lin2 + lin3)

    def __eq__(self, other):
        """
        Check equality with another Structure3D.

        Parameters
        ----------
        other : Structure3D
            Object to compare against.

        Returns
        -------
        bool
            Whether the two objects are considered equal.
        """

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
        """
        Evaluate the structure at a given position.

        Parameters
        ----------
        pos : array_like
            Positions to evaluate.
        **kwargs : dict
            Additional keyword arguments for parameter override.

        Returns
        -------
        np.ndarray
            Evaluated structure value.
        """
        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_normal(**kwargs)

        return self._geometry(**geoty_pa)(self._coordinate(**coord_pa)(pos))

    def f_ray_d(self, pos: ArrayLike, **kwargs) -> np.ndarray:
        """
        Compute the normalized ray distance; 1 indicates the surface.

        Parameters
        ----------
        pos : array_like
            Ray origin or position.
        **kwargs : dict
            Parameters to override.

        Returns
        -------
        np.ndarray
            normalized ray distance.
        """

        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_normal(**kwargs)

        return self._geometry(**geoty_pa).f_ray_d(self._coordinate(**coord_pa)(pos))

    def ray_intersect(self, pos, **kwargs) -> tuple:
        """
        Compute ray intersection with the structure.

        Parameters
        ----------
        pos : array_like
            Ray position.
        **kwargs : dict
            Optional parameters.

        Returns
        -------
        tuple
            Intersection result.
        """
        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_normal(**kwargs)
        
        pos_in,dist = self._geometry(**geoty_pa).ray_intersect(
            self._coordinate(**coord_pa)(pos)
        )
        pos_in = self._coordinate(**coord_pa).inverse(pos_in)

        return pos_in,dist

    def line_intersect(self, pos1: ArrayLike, pos2: ArrayLike, **kwargs) -> np.ndarray:
        """
        Compute intersection of a line segment with the structure.

        Parameters
        ----------
        pos1 : array_like
            Starting point of the line.
        pos2 : array_like
            Ending point of the line.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        np.ndarray
            Intersection points or distances.
        """

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
        """
        Quick the normalized ray distance evaluation.

        Same as `f_ray_d` but accepts direct parameter values.

        Returns
        -------
        np.ndarray
            Result of quick directional evaluation.
        """
        # Convert to numpy array efficiently
        if not isinstance(pos, np.ndarray):
            pos = np.asarray(pos)
            
        # Get parameters
        coord_pa, geoty_pa = self._generate_quick(*args, **kwargs)
        
        # First apply coordinate transformation
        transformed_pos = self._coordinate.quick_call(**coord_pa, pos=pos)

        # Then compute ray distance with transformed positions
        return self._geometry.quick_f_ray_d(**geoty_pa, pos=transformed_pos)

    def quick_ray_dist(self, *args, pos, **kwargs) -> np.ndarray:
        """
        Quick ray distance evaluation.

        Returns
        -------
        np.ndarray
            Ray distances.
        """

        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._generate_quick(*args, **kwargs)

        return self._geometry.quick_ray_dist(
            **geoty_pa, pos=self._coordinate.quick_call(**coord_pa, pos=pos)
        )

    def quick_line_intersect(self, *args, pos1, pos2, **kwargs) -> np.ndarray:
        """
        Quickly compute intersection of a line with the structure.

        Returns
        -------
        np.ndarray
            Intersection points or indicators.
        """

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
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                args = args[0]
                
            # Use pre-allocated dictionaries
            coor_params = self.__coor_pa_num
            coord_pa = {}
            geoty_pa = {}
            
            # Manual assignment is faster than dict comprehension
            for i in range(coor_params):
                coord_pa[self._coordinate.PN[i]] = args[i]
                
            for i in range(coor_params, len(args)):
                geoty_pa[self._geometry.PN[i-coor_params]] = args[i]
                
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
        """
        Generate uniformly distributed points on the structure.

        Parameters
        ----------
        random_np : int, optional
            Number of points to generate (default is 1024).

        Returns
        -------
        np.ndarray
            Points on the surface.
        """
        cpos, spos = fibonacci_sampling(random_np)

        coord_pa, geoty_pa = self._generate_quick()

        points = self._geometry(**geoty_pa).ray_point(cpos)

        return self._coordinate(**coord_pa).inverse(points)
    
    def generate_edge2D(self, n_angle_bins = 130, n_r_bins = 400,r_min=0.2,r_max=3,z_l=1.5,rotation=np.eye(3)):
        """
        Generate the 2D boundary of a shape's projection.

        Parameters
        ----------
        n_angle_bins : int, optional
            Number of bins for angles (default is 130).
        n_r_bins : int, optional
            Number of bins for radius (default is 400).
        r_min : float, optional
            Minimum radius value (default is 0.2).
        r_max : float, optional
            Maximum radius value (default is 3).
        z_l : float, optional
            The z-coordinate limit for the projection (default is 1.5).
        rotation : ndarray, shape (3, 3), optional
            A 3x3 rotation matrix for rotating the shape (default is the identity matrix).

        Returns
        -------
        X : ndarray
            X coordinates of the 2D boundary.
        Y : ndarray
            Y coordinates of the 2D boundary.

        Notes
        -----
        For an edge view along the x-z plane, use:
            rotation = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0.0]]).T

        Visualize example:
            plt.plot(X, Y)
        """
        ang_bins = np.linspace(0,2*np.pi,n_angle_bins)
        r_bins = np.linspace(r_min,r_max,n_r_bins)

        x = np.sin(ang_bins)
        y = np.cos(ang_bins)

        z0 = np.ones(len(x))*(-z_l)
        z1 = np.ones(len(x))*(z_l)

        pos0 = np.array([x,y,z0]).T
        pos1 = np.array([x,y,z1]).T

        pos0_all = np.einsum('ij,k->ikj',pos0,r_bins).reshape(n_angle_bins*n_r_bins,3)

        pos0_all[:,2] = -z_l
        pos1_all = np.einsum('ij,k->ikj',pos1,r_bins).reshape(n_angle_bins*n_r_bins,3)
        pos1_all[:,2] = z_l
        
        pos0_all = np.matmul(pos0_all,rotation.T)
        pos1_all = np.matmul(pos1_all,rotation.T)

        lineinter = self.line_intersect(pos0_all,pos1_all)[:,0]
        lineinter = lineinter.reshape(n_angle_bins,n_r_bins)
        R_all = []
        for i in range(n_angle_bins):
            R_all.append(r_bins[lineinter[i]>0][-1])
            
        R_all = np.asarray(R_all)

        X = R_all*x
        Y = R_all*y
        return X,Y
    
    def generate_edge3D(self, n_phi_bins: int = 120, n_theta_bins: int = 60):
        """
        Generate the 3D boundary of a shape.

        Parameters
        ----------
        n_phi_bins : int, optional
            Number of bins for the azimuthal angle (0~2pi) (default is 120).
        n_theta_bins : int, optional
            Number of bins for the polar angle (0~pi) (default is 60).

        Returns
        -------
        X : ndarray
            X coordinates of the 3D boundary.
        Y : ndarray
            Y coordinates of the 3D boundary.
        Z : ndarray
            Z coordinates of the 3D boundary.

        Visualize example:
            fig = plt.figure(dpi=150, figsize=plt.figaspect(1))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='grey', linewidth=0.1, edgecolor='k', alpha=0.2)
        """
        u = np.linspace(0, 2 * np.pi, n_phi_bins)
        v = np.linspace(0, np.pi, n_theta_bins)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        pos = np.array([x.reshape(1,-1)[0],y.reshape(1,-1)[0],z.reshape(1,-1)[0]]).T
        pos=self._coordinate(**self.parameters).inverse(pos)

        pos_plot,_ =self.ray_intersect(pos)

        X = pos_plot.reshape(n_phi_bins,n_theta_bins,3)[:,:,0]
        Y = pos_plot.reshape(n_phi_bins,n_theta_bins,3)[:,:,1]
        Z = pos_plot.reshape(n_phi_bins,n_theta_bins,3)[:,:,2]
        return X,Y,Z

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
    pos = kwargs['pos']
    f_call = self.quick_f_ray_d(*params, pos=pos) - 1.0
    error_pa = {i: kwargs[i] for i in self._error_params}

    return self._error_func(f_call, **error_pa)


@Structure3D.compute_method_registry
def isodensity_dist(self: Structure3D, params: tuple | ArrayLike, **kwargs) -> float:

    f_call = self.quick_ray_dist(*params, pos=kwargs['pos'])
    error_pa = {i: kwargs[i] for i in self._error_params}

    return self._error_func(f_call, **error_pa)

@Structure3D.compute_method_registry
def validate_fitting_data(self: Structure3D, params: tuple | ArrayLike, **kwargs) -> float:
    """
    Validate data before computing error to catch issues early.
    
    This error computation method provides additional validation before calling
    the standard error computation methods. It's helpful for debugging and testing.
    
    Parameters
    ----------
    self : Structure3D
        The structure instance.
    params : tuple or array_like
        Parameter values for the structure.
    **kwargs : dict
        Additional arguments passed to the error function.
        Must contain 'pos' key with position data.
        
    Returns
    -------
    float
        The computed error value.
        
    Raises
    ------
    ValueError
        If validation fails or required data is missing.
    """
    # Check that required position data exists
    if 'pos' not in kwargs:
        error_msg = "Missing required position data ('pos' key)"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    pos_data = kwargs['pos']
    # Validate position data
    if not isinstance(pos_data, np.ndarray):
        error_msg = f"Position data must be a numpy array, got {type(pos_data)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if len(pos_data.shape) != 2 or pos_data.shape[1] != 3:
        error_msg = f"Position data must have shape (N, 3), got {pos_data.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if np.isnan(pos_data).any():
        error_msg = "Position data contains NaN values"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # After validation, compute f_call using standard method
    try:
        f_call = self.quick_call(*params, pos=pos_data) - 1.0
        error_pa = {i: kwargs[i] for i in self._error_params}
        
        # Log some statistics about the error values before returning
        logger.debug(f"Error computation on {len(f_call)} points: "
                    f"mean={np.mean(f_call**2):.6f}, "
                    f"min={np.min(f_call**2):.6f}, "
                    f"max={np.max(f_call**2):.6f}")
        
        return self._error_func(f_call, **error_pa)
    except Exception as e:
        logger.error(f"Error during validation fitting: {e}", exc_info=True)
        raise
