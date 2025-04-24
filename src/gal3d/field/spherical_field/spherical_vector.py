import logging


import numpy as np
from scipy.spatial import SphericalVoronoi

from .util import *
from ...configuration import config_parser


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
        }
        # TODO other method
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

    def assign_points(self, pos):
        '''
        Assign each point in `pos` to the nearest point on the sphere.

        Parameters:
        -----------
        pos : ndarray, shape (m, 3)
            Cartesian coordinates (x, y, z) of the points to be assigned to the nearest point on the sphere.

        Returns:
        --------
        indices : ndarray, shape (m,)
            The indices of the nearest points on the sphere for each point in `pos`.
        '''
        pos_uni = unit_vector3d(pos)

        batchsize = config_parser['general'].getint('batchsize',fallback=200000)  # prevent memory overflow
        n_split = len(pos_uni) // batchsize
        n_split = max(n_split, 1)

        logger.info(f"Splitting points into {n_split} parts, prevent memory overflow")

        target_pos_split = np.array_split(pos_uni, n_split, axis=0)

        return np.concatenate(
            [np.argmax(Matmul(i, self.pos.T), axis=1) for i in target_pos_split], axis=0
        )

    @staticmethod
    def fibonacci_sampling(Num_sampling: int = 256):
        '''
        Generate points on the unit sphere using the Fibonacci sphere sampling method.

        Parameters:
        -----------
        Num_sampling : int, optional, default 256
            The number of points to generate on the unit sphere.

        Returns:
        --------
        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.
        '''
        logger.info(
            f"Sampling {Num_sampling} random points on the sphere by fibonacci method"
        )
        return fibonacci_sampling(Num_sampling)

    @staticmethod
    def muller_sampling(Num_sampling=256):
        '''
        Generate points on the unit sphere using the Muller method.

        Parameters:
        -----------
        Num_sampling : int, optional, default 256
            The number of points to generate on the unit sphere.

        Returns:
        --------
        pos : ndarray, shape (n, 3)
            Cartesian coordinates (x, y, z) of each point on the unit sphere.

        sph : ndarray, shape (n, 3)
            Spherical coordinates (r, phi, theta) of each point on the unit sphere.
        '''
        logger.info(
            f"Sampling {Num_sampling} random points on the sphere by muller method"
        )

        u, v, w = np.random.randn(3, Num_sampling)
        # unit_vector3d(np.array([u,v,w]).T)
        # norm = vector_length3d(np.array([u,v,w]))
        # x, y, z = (u,v,w) / norm
        sampling_pos = unit_vector3d(np.array([u, v, w]).T)
        sampling_sphere_coor = trans_to_Spherical_coordinates(sampling_pos)

        return sampling_pos, sampling_sphere_coor
    @staticmethod
    def cal_uniformity(pos):
        pos_uni = unit_vector3d(pos)
        voronoi = SphericalVoronoi(pos_uni)
        voronoi.sort_vertices_of_regions()
        area = SphericalVoronoi.calculate_areas(voronoi)
        #self.uniformity = np.std(self.area) / np.mean(self.area)
        target_area = 4*np.pi/len(pos)
        uniformity = 1-np.mean(np.abs(area - target_area))/target_area
        return uniformity
