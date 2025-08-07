import logging
import os

import numpy as np
from scipy.spatial import KDTree, SphericalVoronoi

from ...configuration import config_parser
from .util import *

logger = logging.getLogger('gal3d.preprocessing.spherical_field.spherical_vector')


__all__ = ["SphVector"]


class SphVector:
    '''The coordinates of N points uniformly distributed on the unit sphere'''

    def __init__(self, N_sample=512, method='fibonacci'):
        '''
        Initialize the SphVector class with N points uniformly distributed on the unit sphere.

        Parameters:
        -----------
        N_sample : int, optional, default 512
            The number of points to generate on the unit sphere.

        method : str, optional, default 'fibonacci'
            The method used to generate points on the sphere. Options are 'fibonacci' or 'muller'.

        Attributes:
        -----------
        num : int
            The number of points on the sphere, equal to N_sample.

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
        '''

        METHOD = {
            'fibonacci': self.fibonacci_sampling,
            'muller': self.muller_sampling,
            'golden_spiral': self.golden_spiral_sampling,
            'thomson': self.thomson_sampling,
        }
        # Additional sampling methods can be implemented here if needed.
        self.num = N_sample
        self.pos, self.sph = METHOD[method](self.num)
        self.voronoi = SphericalVoronoi(self.pos)
        self.voronoi.sort_vertices_of_regions()
        self.area = SphericalVoronoi.calculate_areas(self.voronoi)
        target_area = 4*np.pi/N_sample
        self.uniformity = 1 - np.mean(np.abs(self.area - target_area))/target_area
        logger.info(
            f"{self.num} points on the sphere by {method} method have the uniformity of {(self.uniformity*100):.3f}%"
        )
        self._tree =  None


    def assign_points(self, pos):
        '''
        Assign each point in `pos` to the nearest ray.

        Parameters
        ----------
        pos : ndarray, shape (m, 3)
            Cartesian coordinates (x, y, z) of the points to be assigned to the nearest ray.

        Returns
        -------
        indices : ndarray, shape (m,)
            The indices of the nearest rays for each point in `pos`.
        '''
        if self._tree is None:
            self._tree = KDTree(self.pos)
        
        return  self._tree.query(pos,k=1,workers = os.cpu_count())[1]

    @staticmethod
    def fibonacci_sampling(Num_sampling: int = 256):
        '''
        Generate points on the unit sphere using the Fibonacci sphere sampling method.

        Parameters
        ----------
        Num_sampling : int, optional, default 256
            The number of points to generate on the unit sphere.

        Returns
        -------
        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.
        '''

        return fibonacci_sampling(Num_sampling)

    @staticmethod
    def muller_sampling(Num_sampling=256):
        '''
        Generate points on the unit sphere using the Muller method.

        Parameters
        ----------
        Num_sampling : int, optional, default 256
            The number of points to generate on the unit sphere.

        Returns
        -------
        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.
        '''

        u = np.random.normal(size=Num_sampling)
        v = np.random.normal(size=Num_sampling)
        w = np.random.normal(size=Num_sampling)
        cartesian_coords = unit_vector3d(np.array([u, v, w]).T)
        sampling_sphere_coor = trans_to_Spherical_coordinates(cartesian_coords)

        return cartesian_coords, sampling_sphere_coor
    
    @staticmethod
    def golden_spiral_sampling(Num_sampling: int = 256):
        """
        Generate points using the golden spiral method with improved uniformity.
        """
        indices = np.arange(0, Num_sampling, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/Num_sampling)
        theta = np.pi * (1 + 5**0.5) * indices  # Golden angle
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        pos = np.column_stack([x, y, z])
        sph = trans_to_Spherical_coordinates(pos)
        
        return pos, sph
    
    @staticmethod
    def thomson_sampling(Num_sampling: int = 256, iterations: int = 300):
        """
        Generate points using a Thomson problem solver (electrostatic repulsion).
        
        Points repel each other as if they were electrons, settling into a uniform distribution.
        Uses a simulated annealing approach to prevent getting stuck in local minima.
        """
        # Start with random points instead of fibonacci to avoid bias
        np.random.seed(42)  # For reproducibility
        pos = np.random.normal(size=(Num_sampling, 3))
        pos = unit_vector3d(pos)

        # Track best configuration
        best_pos = pos.copy()
        best_uniformity = SphVector.cal_uniformity(pos)
        
        # Cached voronoi for efficiency
        voronoi_cache = SphericalVoronoi(pos)
        
        # Iteratively adjust positions to minimize potential energy
        initial_learning_rate = 0.2
        
        for i in range(iterations):
            # Annealing-based learning rate
            t = 1.0 - i/iterations
            learning_rate = initial_learning_rate * t
            
            # More efficient force calculation (vectorized)
            forces = np.zeros_like(pos)
            
            # For each point, calculate force from all other points
            for j in range(Num_sampling):
                diff = pos - pos[j:j+1]  # Keep dimensions for broadcasting
                dist_squared = np.sum(diff * diff, axis=1)
                
                # Avoid division by zero and extreme forces
                dist_squared = np.maximum(dist_squared, 1e-10)
                
                # Coulomb's law (inverse square)
                magnitude = 1.0 / dist_squared
                magnitude[j] = 0.0  # No self-interaction
                
                # Apply force in direction of displacement
                forces += diff * magnitude.reshape(-1, 1)
            
            # Apply forces with normalization to maintain unit sphere
            new_pos = pos + learning_rate * forces
            new_pos = unit_vector3d(new_pos)
            
            # Calculate energy and uniformity of new configuration
            if i % 10 == 0 or i == iterations - 1:  # Check periodically to save computation
                uniformity = SphVector.cal_uniformity(new_pos, voronoi_cache)
                
                # Keep track of best configuration
                if uniformity > best_uniformity:
                    best_uniformity = uniformity
                    best_pos = new_pos.copy()
                    
                logger.debug(f"Thomson iteration {i}, uniformity: {uniformity:.4f}, best: {best_uniformity:.4f}")
            
            pos = new_pos
                
        sph = trans_to_Spherical_coordinates(best_pos)
        return best_pos, sph
    @staticmethod
    def cal_uniformity(pos, cached_voronoi=None):
        pos_uni = unit_vector3d(pos)
        if cached_voronoi is None or not np.array_equal(cached_voronoi.points, pos_uni):
            cached_voronoi = SphericalVoronoi(pos_uni)
            cached_voronoi.sort_vertices_of_regions()
        area = SphericalVoronoi.calculate_areas(cached_voronoi)
        target_area = 4 * np.pi / len(pos)
        uniformity = 1 - np.mean(np.abs(area - target_area)) / target_area
        return uniformity
