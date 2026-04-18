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
        indent = "      "
        coor_repr = repr(self._coordinate(**self.parameters)).replace("\n", "\n" + indent)
        geom_repr = repr(self._geometry(**self.parameters)).replace("\n", "\n" + indent)
        return (
            f"<{self.__class__.__name__}|\n"
            f"   Coord : {coor_repr}\n"
            f"   Geom  : {geom_repr}\n"
            f"|>"
        )


    def _latex_str_(self):
        coord_pa = {k: self.parameters[k] for k in self._coordinate.PN}
        geoty_pa = {k: self.parameters[k] for k in self._geometry.PN}
        coord_obj = self._coordinate(**coord_pa)
        geom_obj  = self._geometry(**geoty_pa)

        header_coord = r"& \textbf{\text{Coordinate}} & \\"
        header_geom  = r"& \textbf{\text{Geometry}} & \\"

        lines = [
            header_coord,
            coord_obj._latex_str_(),       # include derived/latex_other
            header_geom,
            geom_obj._latex_str_(),
        ]
        return "\n".join(lines)

    def _repr_latex_(self):
        """
        Render a single LaTeX block for the whole StructureCore, containing two sections:
        - Coordinate: Name / Func / Param / Derived param
        - Geometry:   Name / Func / Param / Derived param

        Formatting:
        - One array with 3 columns: label (right aligned), colon (center), value (center)
        - Section headers span all 3 columns
        - Avoids nested $ by stripping inner math wrappers
        """
        body = self._latex_str_()

        return (
            "$\n"
            "\\large\n"
            "\\begin{array}{r c l}\n" +
            body +
            "\n\\end{array}\n"
            "$"
        )

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
        dist:  tuple[np.ndarray,np.ndarray]
            Ray distances and distance to origin.
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

        coord_pa, geoty_pa = self._split_quick_parameters(*args,**kwargs)

        pos1_t = self._coordinate.quick_call(**coord_pa, pos=pos1)
        pos2_t = self._coordinate.quick_call(**coord_pa, pos=pos2)
        return self._geometry.quick_line_intersect(
            **geoty_pa, pos1=pos1_t, pos2=pos2_t
        )

    def generate_points(
        self,
        n_points: int = 1024,
    ) -> np.ndarray:
        """
        Sample surface points using approximately uniform directions.

        Parameters
        ----------
        n_points : int, optional
            Number of sampling directions, default is 1024.

        Returns
        -------
        np.ndarray
            Surface points in the external coordinate frame.
        """

        if n_points <= 0:
            raise ValueError("n_points must be positive")

        directions, _ = fibonacci_sampling(n_points)

        coord_pa, geoty_pa = self._split_quick_parameters()
        coordinate = self._coordinate(**coord_pa)
        geometry = self._geometry(**geoty_pa)

        points_local = geometry.ray_point(directions)
        return coordinate.inverse(points_local)

    def generate_slice2D(
        self,
        n_bins: int = 100,
        z_slice: float = 0.0,
        bins: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the planar cross-section of the shape at ``z = z_slice``.

        Parameters
        ----------
        n_bins : int, optional
            Number of angular samples in the slice plane, default is 100.
        z_slice : float, optional
            Slice plane location in the external coordinate frame, default is 0.0.
        bins : np.ndarray, optional
            Explicit angular sample locations in radians. If provided, supersedes
            ``n_bins``.

        Returns
        -------
        X : np.ndarray
            X coordinates of the planar slice.
        Y : np.ndarray
            Y coordinates of the planar slice.

        Notes
        -----
        Missing intersections are returned as ``nan``.
        """
        if bins is None:
            ang_bins = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=True)
        else:
            ang_bins = np.asarray(bins, dtype=np.float64)

        if ang_bins.ndim != 1 or ang_bins.size == 0:
            raise ValueError("bins must be a non-empty 1D array")

        directions = np.column_stack(
            (
                np.sin(ang_bins),
                np.cos(ang_bins),
                np.zeros_like(ang_bins),
            )
        )
        centers = np.zeros_like(directions)
        centers[:, 2] = z_slice

        pos0 = centers - directions
        pos1 = centers + directions

        t_values = self.line_intersect(pos0, pos1)
        has_hit = np.any(t_values > 0.0, axis=1)

        t_forward = np.max(t_values, axis=1)
        points = pos0 + t_forward[:, None] * directions
        points[~has_hit] = np.nan

        return points[:, 0], points[:, 1]

    def generate_edge2D(
        self,
        n_angle_bins: int = 130,
        n_r_bins: int = 400,
        r_min: float = 0.2,
        r_max: float = 3,
        z_l: float = 1.5,
        rotation: np.ndarray | None = None,
        *,
        angle_bins: np.ndarray | None = None,
        radius_bins: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the 2D projected boundary (edge) of the shape.

        Parameters
        ----------
        n_angle_bins : int, optional
            Number of angular samples in the projected plane, default is 130.
        n_r_bins : int, optional
            Number of radial samples per angle, default is 400.
        r_min : float, optional
            Minimum sampled projected radius, default is 0.2.
        r_max : float, optional
            Maximum sampled projected radius, default is 3.
        z_l : float, optional
            Half-length of the probing line along the projection axis, default is 1.5.
        rotation : ndarray of shape (3, 3), optional
            Rotation applied before projection.
        angle_bins : ndarray, optional
            Explicit angular sample locations in radians. If provided, supersedes
            ``n_angle_bins``.
        radius_bins : ndarray, optional
            Explicit radial sample locations. If provided, supersedes
            ``n_r_bins``, ``r_min`` and ``r_max``.

        Returns
        -------
        X : ndarray
            X coordinates of the 2D boundary.
        Y : ndarray
            Y coordinates of the 2D boundary.

        Notes
        -----
        Angles with no valid line intersection are returned as ``nan``.
        To obtain an edge-on view along the x-z plane, use:

        >>> rotation = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0.0]]).T

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> X, Y = structure.generate_edge2D()
        >>> plt.plot(X, Y)
        """
        if angle_bins is None:
            ang_bins = np.linspace(0.0, 2.0 * np.pi, n_angle_bins, endpoint=True)
        else:
            ang_bins = np.asarray(angle_bins, dtype=np.float64)

        if radius_bins is None:
            r_bins = np.linspace(r_min, r_max, n_r_bins, endpoint=True)
        else:
            r_bins = np.asarray(radius_bins, dtype=np.float64)

        if ang_bins.ndim != 1 or ang_bins.size == 0:
            raise ValueError("angle_bins must be a non-empty 1D array")
        if r_bins.ndim != 1 or r_bins.size == 0:
            raise ValueError("radius_bins must be a non-empty 1D array")

        if rotation is not None:
            rotation = np.asarray(rotation, dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError("rotation must have shape (3, 3)")

        n_angle = ang_bins.size
        n_radius = r_bins.size

        x = np.sin(ang_bins)
        y = np.cos(ang_bins)

        xy = np.stack((x, y), axis=1)
        xy_grid = xy[:, None, :] * r_bins[None, :, None]

        z_bottom = np.full((n_angle, n_radius, 1), -z_l, dtype=np.float64)
        z_top = np.full((n_angle, n_radius, 1), z_l, dtype=np.float64)

        pos0_all = np.concatenate((xy_grid, z_bottom), axis=2).reshape(-1, 3)
        pos1_all = np.concatenate((xy_grid, z_top), axis=2).reshape(-1, 3)

        if rotation is not None:
            pos0_all = pos0_all @ rotation.T
            pos1_all = pos1_all @ rotation.T

        t_values = self.quick_line_intersect(pos1=pos0_all, pos2=pos1_all)
        hit = np.any(t_values > 0.0, axis=1).reshape(n_angle, n_radius)
        has_hit = np.any(hit, axis=1)

        last_hit_from_right = np.argmax(hit[:, ::-1], axis=1)
        radius_idx = n_radius - 1 - last_hit_from_right

        r_all = np.full(n_angle, np.nan, dtype=np.float64)
        r_all[has_hit] = r_bins[radius_idx[has_hit]]

        return r_all * x, r_all * y

    def generate_edge3D(
        self,
        n_phi_bins: int = 120,
        n_theta_bins: int = 60,
        *,
        phi_bins: np.ndarray | None = None,
        theta_bins: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the 3D surface mesh of the shape.

        This method samples the 3D surface of the structure by scanning over spherical angles,
        and computes the corresponding boundary points in 3D space.

        Parameters
        ----------
        n_phi_bins : int, optional
            Number of bins for the azimuthal angle φ (0 to 2π), default is 120.
        n_theta_bins : int, optional
            Number of bins for the polar angle θ (0 to π), default is 60.
        phi_bins : ndarray, optional
            Explicit φ samples in radians over [0, 2π]. If provided, supersedes `n_phi_bins`.
        theta_bins : ndarray, optional
            Explicit θ samples in radians over [0, π]. If provided, supersedes `n_theta_bins`.

        Returns
        -------
        X : ndarray
            X coordinates of the surface mesh, shape (n_phi_bins, n_theta_bins).
        Y : ndarray
            Y coordinates of the surface mesh, shape (n_phi_bins, n_theta_bins).
        Z : ndarray
            Z coordinates of the surface mesh, shape (n_phi_bins, n_theta_bins).

        Notes
        -----
        - When using integer bin counts, φ is coerced to 4a+1 and θ to 2b+1 (to include endpoints nicely).
        - When providing arrays, your exact arrays are used (ensure desired endpoints are included).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> X, Y, Z = structure.generate_edge3D()
        >>> fig = plt.figure(dpi=150, figsize=plt.figaspect(1))
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='grey', linewidth=0.1, edgecolor='k', alpha=0.2)
        """
        if phi_bins is None:
            n_phi_bins = max(int(np.ceil(n_phi_bins / 4)), 1)
            u = np.linspace(0.0, 2.0 * np.pi, 4 * n_phi_bins + 1, endpoint=True)
        else:
            u = np.asarray(phi_bins, dtype=np.float64)

        if theta_bins is None:
            n_theta_bins = max(int(np.ceil(n_theta_bins / 2)), 1)
            v = np.linspace(0.0, np.pi, 2 * n_theta_bins + 1, endpoint=True)
        else:
            v = np.asarray(theta_bins, dtype=np.float64)

        if u.ndim != 1 or u.size == 0:
            raise ValueError("phi_bins must be a non-empty 1D array")
        if v.ndim != 1 or v.size == 0:
            raise ValueError("theta_bins must be a non-empty 1D array")

        nu, nv = u.size, v.size

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        directions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        coord_pa, geoty_pa = self._split_quick_parameters()
        coordinate = self._coordinate(**coord_pa)
        geometry = self._geometry(**geoty_pa)

        pos_plot = coordinate.inverse(geometry.ray_point(directions))

        X = pos_plot[:, 0].reshape(nu, nv)
        Y = pos_plot[:, 1].reshape(nu, nv)
        Z = pos_plot[:, 2].reshape(nu, nv)
        return X, Y, Z

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

    def _latex_str_(self) -> str:
        """
        Return LaTeX rows for error configuration.
        Keep the same 3-column layout: label & : & value, and a centered title row.
        """
        import re
        def _esc(s):
            return re.sub(r"(?<!\\)_", r"\_", s)

        func_name   = _esc(self._error_func_name)
        method_name = _esc(self._error_method_name)
        title = r"& \textbf{\text{Error}} & \\"
        line1 = rf"\text{{Err func}}   & : & \text{{{func_name}}} \\"
        line2 = rf"\text{{Err method}} & : & \text{{{method_name}}} \\"
        return "\n".join([title, line1, line2])

    def _repr_latex_(self):
        body = self._latex_str_()
        return (
            "$\n"
            "\\large\n"
            "\\begin{array}{r c l}\n" +
            body +
            "\n\\end{array}\n"
            "$"
        )

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
        geom_repr = repr(self._geometry(**self.parameters))
        # indent continuation lines of nested reprs
        indent = "      "
        coor_repr_indented = coor_repr.replace("\n", "\n" + indent)
        geom_repr_indented = geom_repr.replace("\n", "\n" + indent)
        return (
            f"<{self.__class__.__name__}|\n"
            f"   Coord : {coor_repr_indented}\n"
            f"   Geom  : {geom_repr_indented}\n"
            f"   Error : {self._error_func_name} / {self._error_method_name}\n"
            f"|>"
        )

    def _latex_str_(self):
        """
        Merge StructureCore (Coordinate/Geometry) and StructureError sections
        into one 3-column array body with global ':' alignment.
        """
        core_body = StructureCore._latex_str_(self)
        err_body  = StructureError._latex_str_(self)

        return "\n".join([core_body, err_body])




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
    if "rscale" in self._error_func_name:
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
    if "rscale" in self._error_func_name:
        r2 = r*r
        f = f*r/np.sqrt(np.sum(r2))
    return f
