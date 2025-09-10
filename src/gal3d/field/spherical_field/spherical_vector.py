import logging
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy.spatial import KDTree, SphericalVoronoi

from gal3d.config import config

from .util import fibonacci_sampling, trans_to_Spherical_coordinates, unit_vector3d

logger = logging.getLogger("gal3d.preprocessing.spherical_field.spherical_vector")


__all__ = ["SphVector"]


class SphSampler:
    @staticmethod
    def fibonacci_sampling(n_sample: int = 256) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate points on the unit sphere using the Fibonacci sphere sampling method.

        Parameters
        ----------
        n_sample : int, optional, default 256
            The number of points to generate on the unit sphere.

        Returns
        -------
        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.
        """

        return fibonacci_sampling(n_sample)

    @staticmethod
    def muller_sampling(n_sample: int = 256) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate points on the unit sphere using the Muller method.

        Parameters
        ----------
        n_sample : int, optional, default 256
            The number of points to generate on the unit sphere.

        Returns
        -------
        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.
        """

        rng = np.random.default_rng(42)  # For reproducibility
        u = rng.normal(size=n_sample)
        v = rng.normal(size=n_sample)
        w = rng.normal(size=n_sample)
        cartesian_coords = unit_vector3d(np.array([u, v, w]).T)
        sampling_sphere_coor = trans_to_Spherical_coordinates(cartesian_coords)

        return cartesian_coords, sampling_sphere_coor

    @staticmethod
    def polar_method(n_sample: int = 256) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate points on the unit sphere using the polar method.

        Parameters
        ----------
        n_sample : int, optional, default 256
            The number of points to generate on the unit sphere.

        Returns
        -------
        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.
        """

        rng = np.random.default_rng(42)  # For reproducibility
        theta = rng.uniform(0, 2 * np.pi, n_sample)
        phi = np.arccos(2 * rng.uniform(0, 1, n_sample) - 1)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        pos = np.column_stack([x, y, z])
        sph = trans_to_Spherical_coordinates(pos)

        return pos, sph



class SphVector:
    """The coordinates of N points uniformly distributed on the unit sphere"""

    METHOD: dict[str, Callable[[int], tuple[np.ndarray, np.ndarray]]] = {
    "fibonacci": SphSampler.fibonacci_sampling,
    "muller": SphSampler.muller_sampling,
    "polar": SphSampler.polar_method
    }

    @classmethod
    @lru_cache(maxsize = 20)
    def create(cls, n_sample: int = 512, method: str = "fibonacci", *, verbose: bool = True) -> "SphVector":

        return cls(n_sample=n_sample, method=method, verbose=verbose, pos=None)

    def __new__(cls, n_sample: int = 512, method: str = "fibonacci", pos: np.ndarray | None = None, *, verbose: bool = True) -> "SphVector":
        """
        Create or retrieve a cached SphVector instance, if not user-defined.
        """
        # only use cache when not user-defined
        if pos is None:
            return cls.create(n_sample, method, verbose=verbose)
        else:
            return super().__new__(cls)

    def __init__(self, n_sample: int = 512, method: str = "fibonacci", pos: np.ndarray | None = None, *, verbose: bool = True):
        """
        Initialize the SphVector class with N points uniformly distributed on the unit sphere.

        Parameters
        ----------
        n_sample : int, optional, default 512
            The number of points to generate on the unit sphere.

        method : str, optional, default 'fibonacci'
            The method used to generate points on the sphere. Options are 'fibonacci', 'muller' or 'polar'.

        pos : ndarray, shape (n, 3), optional
            user defined (x, y, z) of each point on the unit sphere.

        Attributes
        ----------
        num : int
            The number of points on the sphere, equal to n_sample.

        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.

        voronoi : SphericalVoronoi
            Voronoi diagrams on the surface of the sphere. This is an instance of `scipy.spatial.SphericalVoronoi`.

        area : ndarray
            The areas of the Voronoi regions on the sphere.

        uniformity : float
            The ratio of the standard deviation to the mean of the Voronoi region areas,
            which measures the uniformity of the point distribution on the sphere.
        """

        # Additional sampling methods can be implemented here if needed.
        if pos is not None:
            self.num = pos.shape[0]
            self.pos = pos
            self.sph = trans_to_Spherical_coordinates(pos)
            method = "user_defined"
        else:
            self.num = n_sample
            self.pos, self.sph = self.METHOD[method](self.num)
        self.voronoi = SphericalVoronoi(self.pos)
        self.voronoi.sort_vertices_of_regions()
        self.area = SphericalVoronoi.calculate_areas(self.voronoi)
        target_area = 4*np.pi/n_sample
        self.uniformity = 1 - np.mean(np.abs(self.area - target_area))/target_area

        if verbose:
            logger.info(
                "%d points on the sphere by %s method have the uniformity of %.3f",
                self.num, method, self.uniformity * 100
            )
        self._tree: KDTree =  KDTree(self.pos)


    def assign_points(self, pos: np.ndarray) -> np.ndarray:
        """
        Assign each point in `pos` to the nearest ray.

        Parameters
        ----------
        pos : ndarray, shape (m, 3)
            Cartesian coordinates (x, y, z) of the points to be assigned to the nearest ray.

        Returns
        -------
        indices : ndarray, shape (m,)
            The indices of the nearest rays for each point in `pos`.
        """

        # For small self.pos sets (n<100), use vectorized calculation
        if len(self.pos) < 100 and len(pos) < 10000:
            # Convert to unit vectors
            pos_norm = np.sqrt(np.sum(pos**2, axis=1, keepdims=True))
            pos_unit = pos / np.where(pos_norm > 0, pos_norm, 1.0)

            # Dot product approach (find max cosine similarity)
            return np.argmax(np.dot(pos_unit, self.pos.T), axis=1)
        else:
            # Fall back to KDTree for larger point sets
            return self._tree.query(pos, k=1, workers=config.general.number_of_threads)[1]

    @staticmethod
    def cal_uniformity(pos: np.ndarray, cached_voronoi: SphericalVoronoi | None =None) -> float:
        pos_uni = unit_vector3d(pos)
        if cached_voronoi is None or not np.array_equal(cached_voronoi.points, pos_uni):
            cached_voronoi = SphericalVoronoi(pos_uni)
            cached_voronoi.sort_vertices_of_regions()
        area = SphericalVoronoi.calculate_areas(cached_voronoi)
        target_area = 4 * np.pi / len(pos)
        uniformity = 1 - np.mean(np.abs(area - target_area)) / target_area
        return uniformity

