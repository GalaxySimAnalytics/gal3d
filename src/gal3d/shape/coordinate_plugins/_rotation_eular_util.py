import numpy as np
from scipy.spatial.transform import Rotation

from gal3d.util.array_operate import Rotate

__all__ = ["EulerAngles"]


class EulerAngles(Rotation):
    """
    A class to handle 3D rotations using Euler angles.

    This class extends the `Rotation` class to provide additional functionality
    for 3D rotations, including the computation of Jacobians for Euler angles.

    Attributes
    ----------
    None
    """

    def jacobian_euler(self, pos: np.ndarray, seq: str = "zyx") -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Compute the Jacobian of the rotated position with respect to the Euler angles.

        Given a set of positions and a sequence of Euler angles, this function computes
        the Jacobian matrix that describes how the rotated positions change with respect
        to changes in the Euler angles.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be rotated.
        seq : str, optional
            The sequence of Euler angles to use for the rotation. Default is 'zyx'.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple of three Nx3x3 arrays representing the Jacobian matrices for the
            rotated positions with respect to each Euler angle.
        """
        d_theta1, d_theta2, d_theta3 = self.d_euler(seq)

        return (Rotate(pos, d_theta1), Rotate(pos, d_theta2), Rotate(pos, d_theta3))

    def d_euler(self, seq: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Compute the derivative matrices for the Euler angles.

        This function calculates the derivative matrices for the given sequence of Euler angles.
        These matrices are used to compute the Jacobian of the rotation.

        Parameters
        ----------
        seq : str
            The sequence of Euler angles to use for the rotation.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple of three 3x3 arrays representing the derivative matrices for each Euler angle.
        """
        angle = {}
        angle[seq[0]], angle[seq[1]], angle[seq[2]] = self.as_euler(seq)

        C1, S1 = np.cos(angle["z"]), np.sin(angle["z"])
        C2, S2 = np.cos(angle["y"]), np.sin(angle["y"])
        C3, S3 = np.cos(angle["x"]), np.sin(angle["x"])

        angle["R_z"] = np.array([[C1, -S1, 0], [S1, C1, 0], [0, 0, 1]])
        angle["d_R_z"] = np.array([[-S1, -C1, 0], [C1, -S1, 0], [0, 0, 1]])
        angle["R_y"] = np.array([[C2, 0, S2], [0, 1, 0], [-S2, 0, C2]])
        angle["d_R_y"] = np.array([[-S2, 0, C2], [0, 1, 0], [-C2, 0, -S2]])
        angle["R_x"] = np.array(
            [
                [1, 0, 0],
                [0, C3, -S3],
                [0, S3, C3],
            ]
        )
        angle["d_R_x"] = np.array(
            [
                [1, 0, 0],
                [0, -S3, -C3],
                [0, C3, -S3],
            ]
        )

        return (
            np.dot(
                np.dot(angle[f"d_R_{seq[0]}"], angle[f"R_{seq[1]}"]),
                angle[f"R_{seq[2]}"],
            ),
            np.dot(
                np.dot(angle[f"R_{seq[0]}"], angle[f"d_R_{seq[1]}"]),
                angle[f"R_{seq[2]}"],
            ),
            np.dot(
                np.dot(angle[f"R_{seq[0]}"], angle[f"R_{seq[1]}"]),
                angle[f"d_R_{seq[2]}"],
            ),
        )
