
import numpy as np


def ellipsoid_fit(X):
    """
    Fit an ellipsoid to a set of 3D points using least squares.

    Parameters
    ----------
    X : numpy.ndarray
        A 2D array of shape (n, 3) where each row represents a 3D point (x, y, z).

    Returns
    -------
    center : numpy.ndarray
        A 1D array of shape (3,) representing the center of the fitted ellipsoid.
    evecs : numpy.ndarray
        A 2D array of shape (3, 3) representing the eigenvectors of the ellipsoid.
    radii : numpy.ndarray
        A 1D array of shape (3,) representing the radii of the ellipsoid along each eigenvector.
    v : numpy.ndarray
        A 1D array of shape (10,) representing the coefficients of the ellipsoid equation.

    Notes
    -----
    The function fits an ellipsoid to a set of 3D points by solving a least squares problem.
    The ellipsoid is represented by the equation:

    v[0]*x^2 + v[1]*y^2 + v[2]*z^2 + 2*v[3]*x*y + 2*v[4]*x*z + 2*v[5]*y*z + 2*v[6]*x + 2*v[7]*y + 2*v[8]*z + v[9] = 0

    The function returns the center, eigenvectors, radii, and coefficients of the fitted ellipsoid.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(100, 3)  # Random 3D points
    >>> center, evecs, radii, v = ellipsoid_fit(X)
    >>> print("Center:", center)
    >>> print("Eigenvectors:", evecs)
    >>> print("Radii:", radii)
    >>> print("Coefficients:", v)
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v