"""
Module for defining and manipulating 3D shapes.

"""
import logging
from collections.abc import Callable, Sequence
from types import MethodType
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gal3d.field.spherical_field.spherical_vector import fibonacci_sampling
from gal3d.optimization.optimizer import Optimizer, OptimizerBase
from gal3d.util.func_signature import func_required_key

from .coordinate import Coordinate, CoordinateBase
from .fns_cy import register_all
from .geometry import Geometry, GeometryBase
from .minimize_func import MinimizeFunc

if TYPE_CHECKING:
    from gal3d.optimization.parameter import Parameters
    from gal3d.optimization.result import ModelResult

logger = logging.getLogger("gal3d.shape")

class StructureCore:
    """Base class for managing coordinate transformations, geometry, and parameters of a 3D structure."""

    def __init__(
        self,
        coordinate: CoordinateBase | type[CoordinateBase] | str,
        geometry: GeometryBase | type[GeometryBase] | str,
    ):
        """
        Initialize structure components.

        Parameters
        ----------
        coordinate : CoordinateBase or str
            Coordinate transformation class or registered plugin name.
        geometry : GeometryBase or str
            Geometry class or registered plugin name.
        """
        self._set_geometry(geometry)
        self._set_coordinate(coordinate)

        self.parameters: Parameters = (
            self._coordinate.default_parameters() + self._geometry.default_parameters()
        )

    def _set_geometry(self, geometry: GeometryBase | type[GeometryBase] | str) -> None:
        if isinstance(geometry, str):
            self._geometry = Geometry.get_plugin(geometry)
        elif isinstance(geometry, type) and issubclass(geometry, GeometryBase):
            self._geometry = geometry
        elif isinstance(geometry, GeometryBase):
            self._geometry = type(geometry)
        else:
            raise TypeError("geometry must be a class, instance, or plugin name")
        self._geometry_name = self._geometry.__name__

    def _set_coordinate(self, coordinate: CoordinateBase | type[CoordinateBase] | str) -> None:
        if isinstance(coordinate, str):
            self._coordinate = Coordinate.get_plugin(coordinate)
        elif isinstance(coordinate, type) and issubclass(coordinate, CoordinateBase):
            self._coordinate = coordinate
        elif isinstance(coordinate, CoordinateBase):
            self._coordinate = type(coordinate)
        else:
            raise TypeError("coordinate must be a class, instance, or plugin name")
        self._coordinate_name = self._coordinate.__name__
        self.__coor_pa_num = len(self._coordinate.PN)

    def derived_param_funcs(self) -> dict[str, Callable]:
        """Get derived parameter functions."""
        derived_funcs = {}
        derived_funcs.update(self._coordinate.derived_param_funcs())
        derived_funcs.update(self._geometry.derived_param_funcs())
        return derived_funcs

    def copy(self) -> "StructureCore":
        """
        Create a deep copy of the StructureCore instance.

        Returns
        -------
        StructureCore
            A new StructureCore instance with the same coordinate, geometry, and parameters.
        """
        new_instance = StructureCore(
            coordinate=self._coordinate,
            geometry=self._geometry
        )
        new_instance.parameters = self.parameters.copy()
        return new_instance

    def __repr__(self):
        """
        Return string representation of StructureCore.

        Returns
        -------
        str
            Human-readable string of the coordinate and geometry.
        """
        coor_repr = repr(self._coordinate(**self.parameters))
        geometry_repr = repr(self._geometry(**self.parameters))
        lin1 = [f"<{self.__class__.__name__}|: ", "\n"]
        lin2 = ["   ", coor_repr, "\n"]
        lin3 = ["   ", geometry_repr]
        return "".join(lin1 + lin2 + lin3)

    @property
    def geometry_name(self) -> str:
        """Get the name of the geometry."""
        return self._geometry_name

    @property
    def coordinate_name(self) -> str:
        """Get the name of the coordinate system."""
        return self._coordinate_name

    @classmethod
    def available_options(cls) -> dict[str, list[str]]:
        """
        Get available options for the coordinate and geometry.

        Returns
        -------
        dict
            A dictionary of available options.
        """
        return {
            "coordinate": Coordinate.available_plugins(),
            "geometry": Geometry.available_plugins(),
        }

    def estimate_parameters(self, pos: NDArray[np.float64]) -> "Parameters":
        """
        Estimate parameters for the structure based on the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions to estimate parameters from.

        Returns
        -------
        dict
            A dictionary of estimated parameters.
        """
        param_coor = self._coordinate.estimate_parameters(pos)
        new_pos = self._coordinate(**param_coor)(pos)
        param_geom = self._geometry.estimate_parameters(new_pos)

        return self._coordinate(**param_coor).parameters + self._geometry(**param_geom).parameters

    def create_parameters(self, *args: Any, **kwargs: Any) -> "Parameters":
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
            params = dict(zip(self.parameters.keys(), *args, strict=False))
            return self._coordinate.create_parameters(
                **params
            ) + self._geometry.create_parameters(**params)
        if kwargs:
            return self._coordinate.create_parameters(
                **kwargs
            ) + self._geometry.create_parameters(**kwargs)

        return self._coordinate.default_parameters() + self._geometry.default_parameters()

    def set_parameters(self, *args: Any, **kwargs: Any) -> None:
        """
        Set the structure's parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments for parameters.
        **kwargs : dict
            Keyword arguments for parameters.
        """
        self.parameters = self.parameters + self.create_parameters(*args, **kwargs)

    def clone_with_parameters(self: "StructureCore", *args: Any, **kwargs: Any) -> "StructureCore":
        """
        Create a new StructureCore with given parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments for initialization.
        **kwargs : dict
            Keyword arguments for initialization.

        Returns
        -------
        StructureCore
            A new instance with specified parameters.
        """
        new_instance = self.copy()
        new_instance.set_parameters(*args, **kwargs)
        return new_instance

    def is_equal(self, other: "StructureCore") -> bool:
        """
        Check equality with another StructureCore.

        Parameters
        ----------
        other : StructureCore
            Object to compare against.

        Returns
        -------
        bool
            Whether the two objects are considered equal.
        """
        if (
            isinstance(other, StructureCore)
            and (self._coordinate_name == other._coordinate_name)
            and (self._geometry_name == other._geometry_name)
        ):
            return True

        return False


    def transform_pos(self, pos: NDArray[np.float64]) -> np.ndarray:
        """
        Transform positions using the structure's coordinate system.

        Parameters
        ----------
        pos : array_like
            Positions to transform.

        Returns
        -------
        np.ndarray
            Transformed positions.
        """
        coord_pa, geoty_pa = self._split_parameters()
        return self._coordinate(**coord_pa)(pos)

    def inverse_transform(self, pos: NDArray[np.float64]) -> np.ndarray:
        """
        Inverse transform positions from the structure's coordinate system.

        Parameters
        ----------
        pos : array_like
            Positions to inverse transform.

        Returns
        -------
        np.ndarray
            Inverse transformed positions.
        """
        coord_pa, geoty_pa = self._split_parameters()
        return self._coordinate(**coord_pa).inverse(pos)

    def _split_parameters(self, **kwargs: Any) -> tuple[dict, dict]:

        if kwargs:
            coord_pa = self._coordinate.create_parameters(**kwargs)
            geoty_pa = self._geometry.create_parameters(**kwargs)
        else:
            coord_pa = self.parameters.structure_parameters
            geoty_pa = self.parameters.structure_parameters

        return coord_pa, geoty_pa

    def _split_quick_parameters(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        if args:
            flat_args = args[0] if (len(args) == 1 and isinstance(args[0], list | tuple)) else args

            # Use pre-allocated dictionaries
            coor_num = self.__coor_pa_num
            coord_pa = dict(zip(self._coordinate.PN, flat_args[:coor_num], strict=False))
            geoty_pa = dict(zip(self._geometry.PN, flat_args[coor_num:], strict=False))
            return coord_pa, geoty_pa

        if kwargs:
            try:
                coord_pa = {i: kwargs[i] for i in self._coordinate.PN}
                geoty_pa = {i: kwargs[i] for i in self._geometry.PN}
            except KeyError:
                coord_pa = self._coordinate.create_parameters(**kwargs)
                geoty_pa = self._geometry.create_parameters(**kwargs)
            return coord_pa, geoty_pa

        coord_pa = {i: self.parameters[i] for i in self._coordinate.PN}
        geoty_pa = {i: self.parameters[i] for i in self._geometry.PN}
        return coord_pa, geoty_pa

    def __call__(self, pos: ArrayLike, **kwargs: Any) -> np.ndarray:
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

        coord_pa, geoty_pa = self._split_parameters(**kwargs)

        return self._geometry(**geoty_pa)(self._coordinate(**coord_pa)(pos))

    def f_ray_d(self, pos: ArrayLike, **kwargs: Any) -> np.ndarray:
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

        coord_pa, geoty_pa = self._split_parameters(**kwargs)

        return self._geometry(**geoty_pa).f_ray_d(self._coordinate(**coord_pa)(pos))

    def ray_intersect(self, pos: ArrayLike, **kwargs: Any) -> tuple[np.ndarray,np.ndarray]:
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

        coord_pa, geoty_pa = self._split_parameters(**kwargs)

        pos_in,dist,r = self._geometry(**geoty_pa).ray_intersect(
            self._coordinate(**coord_pa)(pos)
        )
        pos_in = self._coordinate(**coord_pa).inverse(pos_in)

        return pos_in,dist

    def line_intersect(self, pos1: ArrayLike, pos2: ArrayLike, **kwargs: Any) -> np.ndarray:
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

        coord_pa, geoty_pa = self._split_parameters(**kwargs)

        return self._geometry(**geoty_pa).line_intersect(
            pos1=self._coordinate(**coord_pa)(pos1),
            pos2=self._coordinate(**coord_pa)(pos2),
        )

    def quick_call(self, *args: Any, pos: ArrayLike, **kwargs: Any) -> tuple[np.ndarray,np.ndarray]:
        """
        Quick evaluation of the structure at the given position.

        Parameters
        ----------
        *args : tuple
            Flat sequence of parameter values (coordinate first, then geometry), or leave empty to use current parameters.
        pos : array_like
            Positions to evaluate.
        **kwargs : dict
            Parameter overrides by name.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (f, r), where:
            - f: shape evaluation.
            - r: radius after transformation.

        Raises
        ------
        KeyError
            If parameters are not provided.
        """
        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._split_quick_parameters(*args, **kwargs)

        return self._geometry.quick_call(
            **geoty_pa, pos=self._coordinate.quick_call(**coord_pa, pos=pos)
        )

    def quick_f_ray_d(self, *args: Any, pos: ArrayLike, **kwargs: Any) -> tuple[np.ndarray,np.ndarray]:
        """
        Quick the normalized ray distance evaluation.

        Same as `f_ray_d` but accepts direct parameter values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (d, r), where d are normalized distances along rays to the surface and r the radius after transformation.
        """
        # Convert to numpy array
        if not isinstance(pos, np.ndarray):
            pos = np.asarray(pos)

        # Get parameters
        coord_pa, geoty_pa = self._split_quick_parameters(*args, **kwargs)

        # First apply coordinate transformation
        transformed_pos = self._coordinate.quick_call(**coord_pa, pos=pos)

        # Then compute ray distance with transformed positions
        return self._geometry.quick_f_ray_d(**geoty_pa, pos=transformed_pos)

    def quick_area_factor(self, *args: Any, pos: ArrayLike, **kwargs: Any) -> NDArray[np.float64]:
        """
        Quick area factor evaluation.

        Same as `area_factor` but accepts direct parameter values.

        Returns
        -------
        np.ndarray
            Result of quick area factor evaluation.
        """
        # Convert to numpy array efficiently
        if not isinstance(pos, np.ndarray):
            pos = np.asarray(pos)

        # Get parameters
        coord_pa, geoty_pa = self._split_quick_parameters(*args, **kwargs)

        # First apply coordinate transformation
        transformed_pos = self._coordinate.quick_call(**coord_pa, pos=pos)

        # Then compute area factor with transformed positions
        return self._geometry.quick_area_factor(**geoty_pa, pos=transformed_pos)

    def quick_ray_dist(self, *args: Any, pos: ArrayLike, **kwargs: Any) -> tuple[np.ndarray,np.ndarray]:
        """
        Quick ray distance evaluation.

        Returns
        -------
        np.ndarray
            Ray distances.
        """

        pos = np.asarray(pos)

        coord_pa, geoty_pa = self._split_quick_parameters(*args, **kwargs)

        return self._geometry.quick_ray_dist(
            **geoty_pa, pos=self._coordinate.quick_call(**coord_pa, pos=pos)
        )

    def quick_line_intersect(self, *args: Any, pos1: ArrayLike, pos2: ArrayLike, **kwargs: Any) -> np.ndarray:
        """
        Quickly compute intersection of a line with the structure.

        Returns
        -------
        np.ndarray
            Intersection points or indicators (geometry-dependent).
        """

        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)

        coord_pa, geoty_pa = self._split_quick_parameters(**kwargs)

        pos1_t = self._coordinate.quick_call(**coord_pa, pos=pos1)
        pos2_t = self._coordinate.quick_call(**coord_pa, pos=pos2)
        return self._geometry.quick_line_intersect(
            **geoty_pa, pos1=pos1_t, pos2=pos2_t
        )

    def generate_points(self, random_np: int = 1024) -> np.ndarray:
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

        coord_pa, geoty_pa = self._split_quick_parameters()

        points = self._geometry(**geoty_pa).ray_point(cpos)

        return self._coordinate(**coord_pa).inverse(points)

    def generate_slice2D(self, n_bins: int = 100, z_slice: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a 2D slice of the shape at a specific z-coordinate.

        Parameters
        ----------
        n_bins : int, optional
            Number of bins for the 2D slice, default is 100.
        z_slice : float, optional
            The z-coordinate at which to slice the shape, default is 0.0.

        Returns
        -------
        X : np.ndarray
            X coordinates of the 2D slice.
        Y : np.ndarray
            Y coordinates of the 2D slice.
        """
        ang_bins = np.linspace(0,2*np.pi,n_bins)
        x = np.sin(ang_bins)
        y = np.cos(ang_bins)
        z = np.ones_like(x) * z_slice
        pos = np.array([x, y, z]).T
        points,_ = self.ray_intersect(pos)
        return points[:,0], points[:,1]

    def generate_edge2D(self, n_angle_bins: int = 130, n_r_bins: int = 400,
                        r_min: float = 0.2, r_max: float = 3, z_l: float = 1.5, rotation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the 2D projected boundary (edge) of the shape.

        This method computes the 2D outline of the 3D structure as seen from a given projection,
        by scanning along multiple angles and radii, and finding the outermost intersection points
        in the projected plane.

        Parameters
        ----------
        n_angle_bins : int, optional
            Number of angular bins (sampling directions in the plane), default is 130.
        n_r_bins : int, optional
            Number of radial bins (sampling radii along each direction), default is 400.
        r_min : float, optional
            Minimum radius to consider for projection, default is 0.2.
        r_max : float, optional
            Maximum radius to consider for projection, default is 3.
        z_l : float, optional
            Half-length along the projection axis (z), default is 1.5.
        rotation : ndarray of shape (3, 3), optional
            3x3 rotation matrix to apply to the shape before projection, default is identity.

        Returns
        -------
        X : ndarray
            X coordinates of the 2D boundary points.
        Y : ndarray
            Y coordinates of the 2D boundary points.

        Notes
        -----
        To obtain an edge-on view along the x-z plane, use:

        >>> rotation = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0.0]]).T

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> X, Y = structure.generate_edge2D()
        >>> plt.plot(X, Y)
        """
        ang_bins = np.linspace(0,2*np.pi,n_angle_bins)
        r_bins = np.linspace(r_min,r_max,n_r_bins)

        x = np.sin(ang_bins)
        y = np.cos(ang_bins)

        z0 = np.ones(len(x))*(-z_l)
        z1 = np.ones(len(x))*(z_l)

        pos0 = np.array([x,y,z0]).T
        pos1 = np.array([x,y,z1]).T

        pos0_all = np.einsum("ij,k->ikj",pos0,r_bins).reshape(n_angle_bins*n_r_bins,3)

        pos0_all[:,2] = -z_l
        pos1_all = np.einsum("ij,k->ikj",pos1,r_bins).reshape(n_angle_bins*n_r_bins,3)
        pos1_all[:,2] = z_l

        if rotation is not None:
            pos0_all = np.matmul(pos0_all, rotation.T)
            pos1_all = np.matmul(pos1_all, rotation.T)

        lineinter = self.line_intersect(pos0_all,pos1_all)[:, 0]
        lineinter = lineinter.reshape(n_angle_bins,n_r_bins)

        R_all: np.ndarray = np.array([r_bins[lineinter[i]>0][-1] for i in range(n_angle_bins)])

        X = R_all*x
        Y = R_all*y
        return X,Y

    def generate_edge3D(self, n_phi_bins: int = 120, n_theta_bins: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the 3D surface boundary of the shape.

        This method samples the 3D surface of the structure by scanning over spherical angles,
        and computes the corresponding boundary points in 3D space.

        Parameters
        ----------
        n_phi_bins : int, optional
            Number of bins for the azimuthal angle φ (0 to 2π), default is 120.
        n_theta_bins : int, optional
            Number of bins for the polar angle θ (0 to π), default is 60.

        Returns
        -------
        X : ndarray
            X coordinates of the 3D boundary surface, shape (n_phi_bins, n_theta_bins).
        Y : ndarray
            Y coordinates of the 3D boundary surface, shape (n_phi_bins, n_theta_bins).
        Z : ndarray
            Z coordinates of the 3D boundary surface, shape (n_phi_bins, n_theta_bins).

        Notes
        -----
        n_phi_bins will be adjusted to 4*a+1, with a at least 1.
        n_theta_bins will be adjusted to 2*b + 1, with b at least 1.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> X, Y, Z = structure.generate_edge3D()
        >>> fig = plt.figure(dpi=150, figsize=plt.figaspect(1))
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='grey', linewidth=0.1, edgecolor='k', alpha=0.2)
        """
        n_phi_bins = max(int(np.ceil(n_phi_bins / 4)), 1)
        n_theta_bins = max(int(np.ceil(n_theta_bins / 2)), 1)
        # need include u = 0, pi 。v = 0, pi/2。
        u = np.linspace(0, 2* np.pi, 4*n_phi_bins+1,endpoint=True)
        v = np.linspace(0, np.pi, 2*n_theta_bins + 1, endpoint=True)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        pos = np.array([x.reshape(1,-1)[0],y.reshape(1,-1)[0],z.reshape(1,-1)[0]]).T
        pos=self._coordinate(**self.parameters).inverse(pos)

        pos_plot,_ =self.ray_intersect(pos)

        X = pos_plot.reshape(n_phi_bins*4+1,n_theta_bins*2+1,3)[:,:,0]
        Y = pos_plot.reshape(n_phi_bins*4+1,n_theta_bins*2+1,3)[:,:,1]
        Z = pos_plot.reshape(n_phi_bins*4+1,n_theta_bins*2+1,3)[:,:,2]
        return X,Y,Z

class StructureError:
    """Base class for managing error functions and evaluation methods for structure fitting."""

    _compute_error_method: dict[str, Callable] = {}

    def __init__(
        self,
        error_func: Callable | str,
        error_method: Callable | str,
    ):
        """
        Initialize error evaluation components.

        Parameters
        ----------
        error_func : Callable or str
            Error function or its name in `MinimizeFunc`.
        error_method : Callable or str
            Method for computing the error or its name in `_compute_error_method`.
        """
        self._set_error_func(error_func)
        self._set_error_method(error_method)
        self.use_ln_error: bool = False

    def _set_error_func(self, error_func: Callable | str) -> None:
        assert callable(error_func) or isinstance(error_func, str)
        self._error_func = (
            MinimizeFunc.minimize_fn[error_func]
            if isinstance(error_func, str)
            else error_func
        )
        self._error_func_name = self._error_func.__name__

        self._error_params = [
            k for k in func_required_key(self._error_func).keys()
            if "_call" not in k
        ]

    def _set_error_method(self, error_method: Callable | str) -> None:
        assert callable(error_method) or isinstance(error_method, str)

        self._error_method = (
            MethodType(self._compute_error_method[error_method], self)
            if isinstance(error_method, str)
            else error_method
        )
        self._error_method_name = self._error_method.__name__

    def is_equal(self, other: "StructureError") -> bool:
        """
        Check equality with another StructureError.

        Parameters
        ----------
        other : StructureError
            Object to compare against.

        Returns
        -------
        bool
            Whether the two objects are considered equal.
        """
        if (
            isinstance(other, StructureError)
            and (self._error_func_name == other._error_func_name)
            and (self._error_method_name == other._error_method_name)
        ):
            return True

        return False

    @classmethod
    def available_options(cls) -> dict[str, list[str]]:
        """
        Get available options for the error evaluation.

        Returns
        -------
        dict
            A dictionary of available options.
        """
        return {
            "error_func": list(MinimizeFunc.minimize_fn.keys()),
            "error_method": list(cls._compute_error_method.keys()),
        }

    def copy(self) -> "StructureError":
        """
        Create a deep copy of the StructureError instance.

        Returns
        -------
        StructureError
            A new StructureError instance with the same error function and method.
        """
        new_instance = StructureError(
            error_func=self._error_func,
            error_method=self._error_method
        )
        new_instance.use_ln_error = self.use_ln_error
        return new_instance

    @classmethod
    def compute_method_registry(cls, fn: str | Callable) -> Callable:
        if callable(fn):
            cls._compute_error_method[fn.__name__] = fn
            return fn
        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                cls._compute_error_method[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator

register_all()
class Structure3D(StructureCore, StructureError):
    """
    A 3D structure composed of a coordinate transformation and a geometry definition.

    This class combines a coordinate transformation (via `CoordinateBase`) and a
    shape definition (via `GeometryBase`). It also provides an error function from
    `MinimizeFunc` and an associated error evaluation method.
    """

    def __init__(
        self,
        coordinate: CoordinateBase | type[CoordinateBase] | str,
        geometry: GeometryBase | type[GeometryBase] | str,
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

        StructureCore.__init__(self, coordinate, geometry)
        StructureError.__init__(self, error_func, error_method)

    def fit(self, pos: np.ndarray, optimizer: None | OptimizerBase = None, estimate: bool = True) -> "ModelResult":
        """
        Fit the structure to the given positions.

        Parameters
        ----------
        pos : array-like
            The positions to fit the structure to.
        **kwargs : dict
            Additional keyword arguments for fitting.
        """
        from gal3d.optimization.result import ModelResult

        if optimizer is None:
            if "curve" in self._error_method_name:
                optimizer = Optimizer.get_plugin(name = "OptimizerScipy")(algorithm="trf")
            else:
                optimizer = Optimizer.get_plugin(name = "OptimizerScipy")(algorithm="Powell")


        parameters_set = self.parameters.new()

        if estimate:
            estimate_params = self.estimate_parameters(pos)
            filtered_estimate_params = {k: v for k, v in estimate_params.items() if k in parameters_set}
            parameters_set.set_value(**filtered_estimate_params)
            parameters_set.clip_to_bounds()

        op_res = optimizer.fit(self._error_method,
                               parameters_set,
                               func_kwargs={"pos": pos})
        for i,j in op_res.params.items():
            parameters_set[i] = j

        parameters_set.add_info(data = pos)
        return ModelResult(self, op_res, parameters_set)


    @classmethod
    def available_options(cls) -> dict[str, list[str]]:
        """
        Get available string options for the structure.

        Returns
        -------
        dict
            A dictionary of available options.
        """
        return StructureCore.available_options() | StructureError.available_options()

    def copy(self) -> "Structure3D":
        """
        Create a deep copy of the Structure3D instance.

        Returns
        -------
        Structure3D
            A new Structure3D instance with the same coordinate, geometry,
            error function, error method, and parameters.
        """
        new_instance = Structure3D(
            coordinate=self._coordinate,
            geometry=self._geometry,
            error_func=self._error_func,
            error_method=self._error_method
        )
        new_instance.parameters = self.parameters.copy()
        new_instance.use_ln_error = self.use_ln_error
        return new_instance

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
        lin1 = [f"<{self.__class__.__name__}|: ", "\n"]
        lin2 = ["   ", coor_repr, "\n"]
        lin3 = ["   ", geometry_repr]
        return "".join(lin1 + lin2 + lin3)




    def is_equal(self, other: Union["Structure3D", "StructureCore","StructureError"]) -> bool:
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
            and StructureCore.is_equal(self, other)
            and StructureError.is_equal(self, other)
        ):
            return True

        return False


@Structure3D.compute_method_registry
def isodensity_fcall(self: Structure3D, params: Sequence[float], **kwargs: Any) -> float:

    f , kwargs["r"]  = self.quick_call(*params, pos=kwargs["pos"])
    error_pa = {i: kwargs[i] for i in self._error_params}
    if self.use_ln_error:
        f = np.log(f)
    else:
        f = f - 1

    return self._error_func(f, **error_pa)


@Structure3D.compute_method_registry
def isodensity_dcall(
    self: Structure3D, params: Sequence[float], *args: Any, **kwargs: Any
) -> float:
    pos = kwargs["pos"]
    f, kwargs["r"] = self.quick_f_ray_d(*params, pos=pos)
    error_pa = {i: kwargs[i] for i in self._error_params}
    if self.use_ln_error:
        f = np.log(f)
    else:
        f = f - 1
    return self._error_func(f, **error_pa)


@Structure3D.compute_method_registry
def isodensity_dist(self: Structure3D, params: Sequence[float], **kwargs: Any) -> float:

    f, kwargs["r"] = self.quick_ray_dist(*params, pos=kwargs["pos"])
    error_pa = {i: kwargs[i] for i in self._error_params}

    return self._error_func(f, **error_pa)

@Structure3D.compute_method_registry
def isodensity_curve_fcall(
    self: Structure3D, params: Sequence[float], **kwargs: Any
) -> np.ndarray:
    pos = kwargs["pos"]
    f, r = self.quick_call(*params, pos=pos)
    if self.use_ln_error:
        f = np.log(f)
    else:
        f = f - 1
    r2 = r*r
    f = f*r/np.sqrt(np.sum(r2))
    return f

@Structure3D.compute_method_registry
def isodensity_curve_dcall(
    self: Structure3D, params: Sequence[float], **kwargs: Any
) -> np.ndarray:
    pos = kwargs["pos"]
    f, r = self.quick_f_ray_d(*params, pos=pos)
    if self.use_ln_error:
        f = np.log(f)
    else:
        f = f - 1
    r2 = r*r
    f = f*r/np.sqrt(np.sum(r2))
    return f
