import logging
from functools import cached_property

import numpy as np

from .util import shrink_sphere_center as ssc
from .util import center_of_mass, centroid, moment_of_inertia, abc_vect
from ..util.array_operate import vector_length3d


logger = logging.getLogger('gal3d.particle.global_calculator')


class GlobalCalculator:
    """
    A class to compute and store global properties of a collection of particles,
    such as their center of mass, moment of inertia, and shape parameters.

    Attributes
    ----------
    pos : numpy.ndarray
        A 2D array of shape (N, 3) representing the positions of N particles in 3D space.
    mass : numpy.ndarray
        A 1D array of shape (N,) representing the masss (e.g., masses) of the particles.

    Methods
    -------
    ssc_center()
        Computes the center of the particles using the shrink-sphere method.
    mass_center()
        Computes the center of mass of the particles.
    shape_center()
        Computes the centroid (geometric center) of the particles.
    moi()
        Computes the moment of inertia tensor of the particles.
    abc()
        Computes the principal axes (a, b, c) of the particles.
    shrink_sphere_center(pos, mass, shrink_factor=0.7, begin_r=None, min_points=100, itermax=100)
        Static method to compute the center using the shrink-sphere method.
    moment_of_inertia(pos, mass)
        Static method to compute the moment of inertia tensor.
    abc_vector(pos, mass)
        Static method to compute the principal axes (a, b, c).
    """

    def __init__(self, pos, mass):
        """
        Parameters
        ----------
        pos : numpy.ndarray
            A 2D array of shape (N, 3) representing the positions of N particles in 3D space.
        mass : numpy.ndarray
            A 1D array of shape (N,) representing the mass of the particles.
        """
        pos = self._shape_check(pos)
        r = vector_length3d(pos)
        ind = np.argsort(r)

        self.pos = pos[ind]
        self.mass = mass[ind]
        self.r = r[ind]

    def _shape_check(self, pos):
        '''
        Ensure the input positions have the correct shape (n, 3).

        Parameters:
            pos: ndarray
                The input positions to be checked and reshaped if necessary.

        Returns:
            pos: ndarray, shape(n,3)
                The reshaped positions.
        '''
        if len(np.shape(pos)) != 2:
            logger.info(f"pos is 1d array with shape={np.shape(pos)}, so we reshape it")
            pos = np.array(pos).reshape(-1, 3)
        if np.shape(pos)[1] == 3:
            return pos
        if np.shape(pos)[0] == 3:
            logger.info(f"pos have the shape= {np.shape(pos)}, so we transpose it")
            return np.array(pos).T
        logger.info(
            f"pos have the shape={np.shape(pos)}, target shape: (n,3), so we reshape this"
        )
        return np.array(pos).reshape(-1, 3)

    @cached_property
    def ssc_center(self):
        """
        Computes the center of the particles using the shrink-sphere method.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (3,) representing the center of the particles.
        """
        return self.shrink_sphere_center(self.pos, self.mass)

    @cached_property
    def mass_center(self):
        """
        Computes the center of mass of the particles.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (3,) representing the center of mass.
        """
        return center_of_mass(self.pos, self.mass)

    @cached_property
    def shape_center(self):
        """
        Computes the centroid (geometric center) of the particles.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (3,) representing the centroid.
        """
        return centroid(self.pos)

    @cached_property
    def moi(self):
        """
        Computes the moment of inertia tensor of the particles.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (3, 3) representing the moment of inertia tensor.
        """
        return moment_of_inertia(self.pos, self.mass)

    @cached_property
    def abc(self):
        """
        Computes the principal axes (a, b, c) of the particles.

        Returns
        -------
        tuple
            A tuple of three floats representing the lengths of the principal axes.
        """
        return abc_vect(self.pos, self.mass)

    @staticmethod
    def shrink_sphere_center(
        pos, mass, shrink_factor=0.7, begin_r=None, min_points=100, itermax=100
    ):
        """
        Computes the center of the particles using the shrink-sphere method.

        Parameters
        ----------
        pos : numpy.ndarray
            A 2D array of shape (N, 3) representing the positions of N particles in 3D space.
        mass : numpy.ndarray
            A 1D array of shape (N,) representing the masss (e.g., masses) of the particles.
        shrink_factor : float, optional
            The factor by which the sphere is shrunk in each iteration. Default is 0.7.
        begin_r : float, optional
            The initial radius of the sphere. If None, it is computed as half the range of the x-coordinates.
        min_points : int, optional
            The minimum number of points required inside the sphere to continue shrinking. Default is 100.
        itermax : int, optional
            The maximum number of iterations to perform. Default is 100.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (3,) representing the center of the particles.
        """
        begin_r = begin_r or (np.max(pos[:, 0]) - np.min(pos[:, 0])) / 2

        logger.info(f"Using a begin_r= {begin_r:.2f}")

        cen, final_r, v_r, iternum = ssc(
            np.array(pos),
            np.array(mass),
            min_points,
            0,
            shrink_factor,
            begin_r,
            itermax,
        )

        logger.info(f"Iteration num= {iternum}")

        if iternum > itermax:
            logger.error(
                f"shrink_sphere_center failed to converge after {iternum} iterations"
            )

        logger.info(f"After iteration, final_r= {final_r:.2f}")

        return cen

    @staticmethod
    def moment_of_inertia(pos, mass):
        """
        Computes the moment of inertia tensor of the particles.

        Parameters
        ----------
        pos : numpy.ndarray
            A 2D array of shape (N, 3) representing the positions of N particles in 3D space.
        mass : numpy.ndarray
            A 1D array of shape (N,) representing the masss (e.g., masses) of the particles.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (3, 3) representing the moment of inertia tensor.
        """
        return moment_of_inertia(pos, mass)

    @staticmethod
    def abc_vector(pos, mass):
        """
        Computes the principal axes (a, b, c) of the particles.

        Parameters
        ----------
        pos : numpy.ndarray
            A 2D array of shape (N, 3) representing the positions of N particles in 3D space.
        mass : numpy.ndarray
            A 1D array of shape (N,) representing the masss (e.g., masses) of the particles.

        Returns
        -------
        tuple
            A tuple of three floats representing the lengths of the principal axes.
        """
        return abc_vect(pos, mass)
