import numpy as np

def f_shaped_ellipsoid(
    a: float, b: float, c: float, Sa: float, Sb: float, Sc: float,
    pos: np.ndarray
) -> np.ndarray: ...

def f_shaped_ellipsoid_jacobian(
    a: float, b: float, c: float, Sa: float, Sb: float, Sc: float,
    pos: np.ndarray
) -> tuple[np.ndarray, ...]: ...

def IntersectRaysEllipsoid_S(
    a: float, b: float, c: float, Sa: float, Sb: float, Sc: float,
    pos: np.ndarray, maxIterations: int
) -> tuple[np.ndarray, np.ndarray]: ...

def f_ray_shaped_ellipsoid(
    a: float, b: float, c: float, Sa: float, Sb: float, Sc: float,
    pos: np.ndarray, maxIterations: int
) -> np.ndarray: ...

def IntersectLinesEllipsoid_S(
    a: float, b: float, c: float, Sa: float, Sb: float, Sc: float,
    pos1: np.ndarray, pos2: np.ndarray, maxIteration: int
) -> np.ndarray: ...
