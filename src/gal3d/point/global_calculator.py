import logging
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike

from gal3d.util.array_operate import Auto3DShape, vector_length3d

from .util import (
    abc_vect,
    center_of_mass,
    centroid,
    moment_of_inertia,
    shrink_sphere_center as ssc,
)

logger = logging.getLogger("gal3d.particle.global_calculator")


class GlobalCalculator(Auto3DShape):
    """
    A class to compute and store global properties of a collection of particles,
    such as their center of mass, moment of inertia tensor, and principal axes.

    Attributes
    ----------
    pos : numpy.ndarray
        Sorted particle positions based on their radius from origin.
    mass : numpy.ndarray
        Sorted particle masses corresponding to `pos`.
    r : numpy.ndarray
        Radial distances of particles from origin, sorted.

    """

    def __init__(self, pos: ArrayLike, mass: ArrayLike, recenter: bool = True, sort_by_radius: bool = False):
        """
        Parameters
        ----------
        pos : ArrayLike
            Particle positions.
        mass : ArrayLike
            Particle masses.
        recenter : bool, optional
            Whether to recenter positions using the shrink-sphere method. Default is True.
        sort_by_radius : bool, optional
            Whether to sort particles by their radial distance from the origin. Default is False.
        """

        posarr: np.ndarray = self.to_3d_array(pos)
        massarr: np.ndarray = np.array(mass)
        if recenter:
            cen: np.ndarray = self.shrink_sphere_center(posarr,massarr)
            posarr = posarr - cen
            logger.info("Recentered positions by subtracting center: %s", cen)
        r: np.ndarray = vector_length3d(posarr)
        if sort_by_radius:
            ind = np.argsort(r)
            self.pos = posarr[ind]
            self.mass = massarr[ind]
            self.r = r[ind]
        else:
            self.pos = posarr
            self.mass = massarr
            self.r = r

        if self.pos.shape[0] != self.mass.shape[0]:
            raise ValueError(
                f"Mismatch between number of positions ({self.pos.shape[0]}) and masses ({self.mass.shape[0]})."
            )

    def __del__(self):
        """
        Clean up large data arrays to assist garbage collection.
        """
        for attr in ["pos", "mass", "r", "_ssc_center", "_mass_center", "_shape_center", "_moi", "_abc"]:
            if hasattr(self, attr):
                delattr(self, attr)

    @cached_property
    def ssc_center(self) -> np.ndarray:
        """
        Computes the center using the shrink-sphere method.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (3,) representing the computed center.
        """
        return self.shrink_sphere_center(self.pos, self.mass)

    @cached_property
    def mass_center(self) -> np.ndarray:
        """
        Computes the mass-weighted center (center of mass).

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (3,) representing the center of mass.
        """
        return center_of_mass(self.pos, self.mass)

    @cached_property
    def shape_center(self) -> np.ndarray:
        """
        Computes the geometric center (centroid) of the particles.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (3,) representing the centroid.
        """
        return centroid(self.pos)

    @cached_property
    def moi(self) -> np.ndarray:
        """
        Computes the moment of inertia tensor of the particle distribution.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (3, 3) representing the moment of inertia tensor.
        """
        return moment_of_inertia(self.pos, self.mass)

    @cached_property
    def abc(self) -> tuple[np.ndarray,np.ndarray]:
        """
        Computes the principal axes lengths (a, b, c) based on the inertia tensor.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple ([a, b, c], rotation_matrix), where a >= b >= c are the principal axis lengths.
        """
        return abc_vect(self.pos, self.mass)

    @staticmethod
    def shrink_sphere_center(
        pos: np.ndarray, mass: np.ndarray, shrink_factor: float=0.7, begin_r: float | None =None, min_points: int=100, itermax: int =100
    ) -> np.ndarray:
        """
        Computes the center using the shrink-sphere method.

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

        logger.debug("Using a begin_r= %.2f", begin_r)

        cen, final_r, v_r, iternum = ssc(
            np.array(pos),
            np.array(mass),
            min_points,
            0,
            shrink_factor,
            begin_r,
            itermax,
        )

        logger.debug("Iteration num= %d", iternum)

        if iternum > itermax:
            logger.error(
                "shrink_sphere_center failed to converge after %d iterations", iternum
            )

        logger.debug("After iteration, final_r= %.2f", final_r)

        return cen

    @staticmethod
    def moment_of_inertia(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        """
        Computes the moment of inertia tensor.

        Parameters
        ----------
        pos : numpy.ndarray
            Position array of shape (N, 3).
        mass : numpy.ndarray
            Mass array of shape (N,).

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (3, 3) representing the moment of inertia tensor.
        """
        return moment_of_inertia(pos, mass)

    @staticmethod
    def compute_abc(pos: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """
        Computes the principal axes lengths (a, b, c) based on the inertia tensor.

        Parameters
        ----------
        pos : numpy.ndarray
            Position array of shape (N, 3).
        mass : numpy.ndarray
            Mass array of shape (N,).

        Returns
        -------
        tuple of numpy.ndarray
            A tuple ([a, b, c], rotation_matrix), where a >= b >= c are the principal axis lengths.
        """
        return abc_vect(pos, mass)

    def as_dict(self) -> dict:
        """
        Returns a dictionary of all computed global properties.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            'ssc_center', 'mass_center', 'shape_center', 'abc'
        """
        return {"ssc_center": self.ssc_center.tolist(),
                "mass_center": self.mass_center.tolist(),
                "shape_center": self.shape_center.tolist(),
                "abc": self.abc}

