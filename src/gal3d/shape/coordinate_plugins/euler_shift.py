from typing import Any

import numpy as np

from gal3d.shape.coordinate import CoordinateBase
from gal3d.util.array_operate import Rotate, RotateAndShift, Shift

from ._rotation_eular_util import EulerAngles

__all__ = ["EulerShift","ShiftEuler","ShiftOnly","RotateOnly"]

class ShiftOnly(CoordinateBase):
    PN = ("x", "y", "z")
    LB = {"x": -0.2, "y": -0.2, "z": -0.2}
    UB = {"x": 0.2, "y": 0.2, "z": 0.2}

    def __init__(self, x: float, y: float, z: float, **kwargs: Any):
        super().__init__(x=x, y=y, z=z)

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        return Shift(pos, self["pos"])

    @classmethod
    def default_parameters(cls):
        """
        Returns a default set of parameters for the EulerShift transformation.
        """
        return cls.create_parameters(
            x=0.0, y=0.0, z=0.0,
        )

    def inverse(self, pos: np.ndarray) -> np.ndarray:
        return Shift(pos, -self["pos"])

    def jacobian(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N_p = len(pos)
        d_Px = np.array([-np.ones(N_p), np.zeros(N_p), np.zeros(N_p)])
        d_Py = np.array([np.zeros(N_p), -np.ones(N_p), np.zeros(N_p)])
        d_Pz = np.array([np.zeros(N_p), np.zeros(N_p), -np.ones(N_p)])
        return (d_Px.T, d_Py.T, d_Pz.T)

    @staticmethod
    def quick_call(x, y, z, pos):
        pc = np.asarray([x, y, z], dtype=np.float64)
        return Shift(pos, pc)

    @staticmethod
    def quick_inverse(x, y, z, pos):
        pc = np.asarray([x, y, z], dtype=np.float64)
        return Shift(pos, -pc)

    @staticmethod
    def quick_jacobian(x, y, z, pos):
        N_p = len(pos)
        d_Px = np.array([-np.ones(N_p), np.zeros(N_p), np.zeros(N_p)])
        d_Py = np.array([np.zeros(N_p), -np.ones(N_p), np.zeros(N_p)])
        d_Pz = np.array([np.zeros(N_p), np.zeros(N_p), -np.ones(N_p)])
        return (d_Px.T, d_Py.T, d_Pz.T)
    @classmethod
    def estimate_parameters(cls, pos: np.ndarray) -> dict:
        """
        Estimate the parameters for the Shift transformation based on the provided positions.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        dict
            A dictionary containing the estimated parameters.
        """
        pos = cls.to_3d_array(pos)
        # Compute the centroid of the positions
        centroid = np.median(pos, axis=0)
        return {
            "x": centroid[0],
            "y": centroid[1],
            "z": centroid[2],
        }

class RotateOnly(CoordinateBase):
    PN = ("ang1", "ang2", "ang3")
    LB = {"ang1": -np.pi, "ang2": -np.pi/2, "ang3": -np.pi}
    UB = {"ang1": np.pi, "ang2": np.pi/2, "ang3": np.pi}
    EulerSeq = "zyx"

    def __init__(self, ang1: float, ang2: float, ang3: float, **kwargs: Any):
        super().__init__(ang1=ang1, ang2=ang2, ang3=ang3)
        from ._rotation_eular_util import EulerAngles
        self._seq = kwargs.get("seq", RotateOnly.EulerSeq)
        self._rotation = EulerAngles.from_euler(
            seq=self._seq, angles=[ang1, ang2, ang3]
        )

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        return Rotate(pos, self._rotation.as_matrix())

    def inverse(self, pos: np.ndarray) -> np.ndarray:
        return Rotate(pos, self._rotation.as_matrix().T)

    @classmethod
    def default_parameters(cls):
        """
        Returns a default set of parameters for the EulerShift transformation.
        """
        return cls.create_parameters(
            ang1=0.0, ang2=0.0, ang3=0.0
        )

    def jacobian(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos1 = pos
        d_ang1, d_ang2, d_ang3 = self._rotation.jacobian_euler(pos=pos1, seq=self._seq)
        return (d_ang1, d_ang2, d_ang3)

    @staticmethod
    def quick_call(ang1, ang2, ang3, pos):
        from ._rotation_eular_util import EulerAngles
        rot_matrix = EulerAngles.from_euler(
            seq=RotateOnly.EulerSeq, angles=[ang1, ang2, ang3]
        ).as_matrix()
        return Rotate(pos, rot_matrix)

    @staticmethod
    def quick_inverse(ang1, ang2, ang3, pos):
        from ._rotation_eular_util import EulerAngles
        matrix = EulerAngles.from_euler(
            seq=RotateOnly.EulerSeq, angles=[ang1, ang2, ang3]
        ).as_matrix()
        return Rotate(pos, matrix.T)

    @staticmethod
    def quick_jacobian(ang1, ang2, ang3, pos):
        from ._rotation_eular_util import EulerAngles
        Rt = EulerAngles.from_euler(seq=RotateOnly.EulerSeq, angles=[ang1, ang2, ang3])
        d_ang1, d_ang2, d_ang3 = Rt.jacobian_euler(pos=pos, seq=RotateOnly.EulerSeq)
        return (d_ang1, d_ang2, d_ang3)
    @classmethod
    def estimate_parameters(cls, pos: np.ndarray) -> dict:
        """
        Estimate the parameters for the EulerShift transformation based on the provided positions.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        dict
            A dictionary containing the estimated parameters.
        """
        pos = cls.to_3d_array(pos)
        try:
            U, S, Vt = np.linalg.svd(pos, full_matrices=False)
            axes = Vt  # shape (3,3)

            # to eular angle
            rot = EulerAngles.from_matrix(axes)
            angles = rot.as_euler(cls.EulerSeq, degrees=False)
        except ValueError:
            angles = [0.0, 0.0, 0.0]

        return {
            "ang1": angles[0],
            "ang2": angles[1],
            "ang3": angles[2],
        }

    @classmethod
    def mat_to_angle(cls, mat: np.ndarray) -> np.ndarray:
        """
        Convert a rotation matrix to Euler angles.

        Parameters
        ----------
        mat : numpy.ndarray
            A 3x3 rotation matrix.

        Returns
        -------
        numpy.ndarray
            The corresponding Euler angles.
        """
        return EulerAngles.from_matrix(mat).as_euler(cls.EulerSeq, degrees=False)
class EulerShift(CoordinateBase):

    PN = ("x", "y", "z", "ang1", "ang2", "ang3")  ##!!!! not use set !!!!
    LB = {
        "x": -0.2,
        "y": -0.2,
        "z": -0.2,
        "ang1": -np.pi,
        "ang2": -np.pi / 2,
        "ang3": -np.pi,
    }
    UB = {"x": 0.2, "y": 0.2, "z": 0.2, "ang1": np.pi, "ang2": np.pi / 2, "ang3": np.pi}

    EulerSeq = "zyx"

    def __init__(self, x: float, y: float, z: float, ang1: float, ang2: float, ang3: float, **kwargs: Any):
        """
        Initialize the EulerShift object with translation and rotation parameters.

        Parameters
        ----------
        x, y, z : float
            The center position coordinates.
        ang1, ang2, ang3 : float
            The Euler angles for rotation, in units of pi.
        **kwargs : dict
            Additional keyword arguments, including:
            - seq : str, optional
                The sequence of Euler angles for rotation. Default is 'zyx'.
        """
        super().__init__(x=x,y=y,z=z,ang1=ang1,ang2=ang2,ang3=ang3)

        self._seq = kwargs.get("seq", EulerShift.EulerSeq)
        self._rotation = EulerAngles.from_euler(
            seq=self._seq, angles=[ang1, ang2, ang3]
        )

    @classmethod
    def default_parameters(cls):
        """
        Returns a default set of parameters for the EulerShift transformation.
        """
        return cls.create_parameters(
            x=0.0, y=0.0, z=0.0, ang1=0.0, ang2=0.0, ang3=0.0
        )

    @classmethod
    def mat_to_angle(cls, mat: np.ndarray) -> np.ndarray:
        """
        Convert a rotation matrix to Euler angles.

        Parameters
        ----------
        mat : numpy.ndarray
            A 3x3 rotation matrix.

        Returns
        -------
        numpy.ndarray
            The corresponding Euler angles.
        """
        return EulerAngles.from_matrix(mat).as_euler(cls.EulerSeq, degrees=False)

    def jacobian(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Jacobian of the transformed positions with respect to the translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple of six arrays representing the Jacobian matrices for the transformed positions
            with respect to the translation and rotation parameters.
        """
        pos1 = Shift(pos, -self["pos"])
        N_p = len(pos)
        d_ang1, d_ang2, d_ang3 = self._rotation.jacobian_euler(pos=pos1, seq=self._seq)
        self._rotation.as_matrix()

        d_Px = np.array(
            [
                -np.ones(N_p),
                np.zeros(N_p),
                np.zeros(N_p),
            ]
        )
        d_Py = np.array([np.zeros(N_p), -np.ones(N_p), np.zeros(N_p)])
        d_Pz = np.array([np.zeros(N_p), np.zeros(N_p), -np.ones(N_p)])

        return (d_Px.T, d_Py.T, d_Pz.T, d_ang1, d_ang2, d_ang3)

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        """
        Transform the given positions using the current translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        numpy.ndarray
            The transformed positions.
        """
        pos = self.to_3d_array(pos)
        if np.allclose(self["pos"], 0):
            return Rotate(pos, self._rotation.as_matrix())
        return Shift(Rotate(pos, self._rotation.as_matrix()), self["pos"])

    def inverse(self, pos: np.ndarray) -> np.ndarray:
        """
        Inverse transform the given positions using the current translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be inverse transformed.

        Returns
        -------
        numpy.ndarray
            The inverse transformed positions.
        """
        pos = self.to_3d_array(pos)
        return Rotate(Shift(pos.copy(), -self["pos"]), self._rotation.as_matrix().T)

    @classmethod
    def estimate_parameters(cls, pos: np.ndarray) -> dict:
        """
        Estimate the parameters for the EulerShift transformation based on the provided positions.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        dict
            A dictionary containing the estimated parameters.
        """
        pos = cls.to_3d_array(pos)
        # Compute the centroid of the positions
        centroid = np.median(pos, axis=0)
        try:
            new_pos = pos - centroid
            U, S, Vt = np.linalg.svd(new_pos, full_matrices=False)
            axes = Vt  # shape (3,3)

            # to eular angle
            rot = EulerAngles.from_matrix(axes)
            angles = rot.as_euler(cls.EulerSeq, degrees=False)
        except ValueError:
            angles = [0.0, 0.0, 0.0]

        return {
            "x": centroid[0],
            "y": centroid[1],
            "z": centroid[2],
            "ang1": angles[0],
            "ang2": angles[1],
            "ang3": angles[2],
        }

    @staticmethod
    def quick_call(x: float, y: float, z: float, ang1: float, ang2: float, ang3: float, pos: np.ndarray) -> np.ndarray:
        """
        Quickly transform the given positions using the specified translation and rotation parameters.

        Parameters
        ----------
        x, y, z : float
            The center position coordinates.
        ang1, ang2, ang3 : float
            The Euler angles for rotation, in units of pi.
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        numpy.ndarray
            The transformed positions.
        """
        pc = np.asarray([x, y, z], dtype=np.float64)

        rot_matrix = EulerAngles.from_euler(
            seq=EulerShift.EulerSeq, angles=[ang1, ang2, ang3]
        ).as_matrix()
        return RotateAndShift(pos, rot_matrix, pc)

    @staticmethod
    def quick_inverse(x: float, y: float, z: float, ang1: float, ang2: float, ang3: float, pos: np.ndarray) -> np.ndarray:
        """
        Quickly inverse transform the given positions using the specified translation and rotation parameters.

        Parameters
        ----------
        x, y, z : float
            The center position coordinates.
        ang1, ang2, ang3 : float
            The Euler angles for rotation, in units of pi.
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be inverse transformed.

        Returns
        -------
        numpy.ndarray
            The inverse transformed positions.
        """
        pc: np.ndarray = np.asarray([x, y, z],dtype=np.float64)
        matrix = EulerAngles.from_euler(
            seq=EulerShift.EulerSeq, angles=[ang1, ang2, ang3]
        ).as_matrix()
        return Rotate(Shift(pos.copy(), -pc), matrix.T)

    @staticmethod
    def quick_jacobian(x: float, y: float, z: float, ang1: float, ang2: float, ang3: float, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        pc: np.ndarray = np.asarray([x, y, z],dtype=np.float64)

        EulerAngles.from_euler(
            seq=EulerShift.EulerSeq, angles=[ang1, ang2, ang3]
        ).as_matrix()
        pos1 = Shift(pos, -pc)
        N_p = len(pos)
        Rt = EulerAngles.from_euler(seq=EulerShift.EulerSeq, angles=[ang1, ang2, ang3])
        d_ang1, d_ang2, d_ang3 = Rt.jacobian_euler(pos=pos1, seq=EulerShift.EulerSeq)

        Rt.as_matrix()

        d_Px = np.array(
            [
                -np.ones(N_p),
                np.zeros(N_p),
                np.zeros(N_p),
            ]
        )
        d_Py = np.array([np.zeros(N_p), -np.ones(N_p), np.zeros(N_p)])
        d_Pz = np.array([np.zeros(N_p), np.zeros(N_p), -np.ones(N_p)])

        return (d_Px.T, d_Py.T, d_Pz.T, d_ang1, d_ang2, d_ang3)



class ShiftEuler(EulerShift):
    """
    Similar to EulerShift, but applies a shift to the positions before rotation.
    may be useful for iterative optimization.
    """

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        """
        Transform the given positions using the current translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        numpy.ndarray
            The transformed positions.
        """
        pos = self.to_3d_array(pos)
        if np.allclose(self["pos"], 0):
            return Rotate(pos, self._rotation.as_matrix())
        return Rotate(Shift(pos, self["pos"]), self._rotation.as_matrix())

    def inverse(self, pos: np.ndarray) -> np.ndarray:
        """
        Inverse transform the given positions using the current translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be inverse transformed.

        Returns
        -------
        numpy.ndarray
            The inverse transformed positions.
        """
        pos = self.to_3d_array(pos)
        return Shift(Rotate(pos,self._rotation.as_matrix().T), -self["pos"])


@ShiftOnly.derived
@EulerShift.derived
@ShiftEuler.derived
def pos(params):
    return np.array([params["x"], params["y"], params["z"]])

@RotateOnly.derived
@EulerShift.derived
@ShiftEuler.derived
def angle(params):
    return np.array([params["ang1"], params["ang2"], params["ang3"]])
