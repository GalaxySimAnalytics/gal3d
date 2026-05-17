"""
Theoretical density distributions.

Coordinate-transform hierarchy
-------------------------------
CoordinateTransform  (ABC, extensibility hook)
    └── FieldCoordinate      constant centre / orientation / axis ratios  [STABLE]
    └── RadialFieldCoordinate  radially-varying q(m), s(m)               [FUTURE]

Density hierarchy
-----------------
DensitySource
    └── TheoreticalDensityDistribution  (abstract)
            ├── CombinedDensityDistribution
            ├── PlummerSphere
            ├── DoubleExponentialDisk
            └── PowerLawSphere
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self

import numpy as np

from gal3d.density import DensitySource

THEORETICAL_MODELS: dict[str, type] = {}


# ======================================================================
# CoordinateTransform — extensibility ABC
# ======================================================================
class CoordinateTransform(ABC):
    """
    Abstract coordinate transformation from world frame to elliptical frame.

    Subclasses must implement :meth:`world_to_elliptical`.

    The only contract is:

    * Input: world-frame positions, shape ``(n, 3)``
    * Output: elliptical-frame positions, shape ``(n, 3)``, such that
      ``np.linalg.norm(out, axis=-1)`` equals the elliptical radius *m*.

    This ABC is the extension hook for future radially-varying transforms
    (e.g. ``RadialFieldCoordinate``).
    """

    @abstractmethod
    def world_to_elliptical(self, pos_world: np.ndarray) -> np.ndarray:
        """
        Map world-frame positions to the model's elliptical frame.

        Parameters
        ----------
        pos_world : ndarray, shape (n, 3)
        Returns
        -------
        ndarray, shape (n, 3)
        """


# ======================================================================
# FieldCoordinate — constant ellipticity / orientation  [STABLE]
# ======================================================================
@dataclass
class FieldCoordinate(CoordinateTransform):
    """
    Constant geometric transformation: translation + rotation + axis scaling.

    Parameters
    ----------
    center_pos : array_like, shape (3,)
        Centre in world coordinates.  Default ``[0, 0, 0]``.
    angles : (float, float, float)
        XYZ Euler angles ``(ax, ay, az)`` in radians.  Default ``(0, 0, 0)``.
    scales : array_like, shape (3,)
        Per-axis scale factors ``(a, b, c)`` with all values > 0.
        Axis ratios are ``q = b/a``, ``s = c/a``.
        Default ``[1, 1, 1]`` (spherical).
    """

    center_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scales: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=float))

    def __post_init__(self) -> None:
        self.center_pos = np.asarray(self.center_pos, dtype=float)
        self.scales = np.asarray(self.scales, dtype=float)
        if self.center_pos.shape != (3,):
            raise ValueError("center_pos must have shape (3,)")
        if self.scales.shape != (3,) or np.any(self.scales <= 0):
            raise ValueError("scales must have shape (3,) and all values > 0")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def rotation_matrix_xyz(angles: tuple[float, float, float]) -> np.ndarray:
        """
        Build a combined XYZ Euler rotation matrix ``Rz @ Ry @ Rx``.

        Parameters
        ----------
        angles : (float, float, float)
            ``(ax, ay, az)`` in radians.

        Returns
        -------
        ndarray, shape (3, 3)
        """
        ax, ay, az = angles
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rz @ ry @ rx

    # ------------------------------------------------------------------
    # CoordinateTransform implementation
    # ------------------------------------------------------------------
    def world_to_elliptical(self, pos_world: np.ndarray) -> np.ndarray:
        """
        Shift → rotate (inverse) → scale to produce elliptical coordinates.

        Parameters
        ----------
        pos_world : ndarray, shape (n, 3)

        Returns
        -------
        ndarray, shape (n, 3)
        """
        p = np.asarray(pos_world, dtype=float)
        shifted = p - self.center_pos
        rot = self.rotation_matrix_xyz(self.angles)
        body = shifted @ rot.T  # world → body (Rᵀ = R⁻¹)
        return body / self.scales  # apply axis scaling


# ======================================================================
# TheoreticalDensityDistribution — abstract base
# ======================================================================
class TheoreticalDensityDistribution(DensitySource):
    """
    Abstract base for analytical 3-D density models.

    Subclasses must implement :meth:`_evaluate_density_generic`, which
    receives positions already in the model's elliptical frame.

    Parameters
    ----------
    coordinate : CoordinateTransform, optional
        World-to-elliptical transform.  Defaults to a spherical
        :class:`FieldCoordinate` centred at the origin.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        THEORETICAL_MODELS[cls.__name__] = cls

    @classmethod
    def available_models(cls) -> list[str]:
        """Return names of all registered :class:`TheoreticalDensityDistribution` subclasses."""
        return list(THEORETICAL_MODELS.keys())

    def __init__(self, coordinate: CoordinateTransform | None = None) -> None:
        self.coordinate: CoordinateTransform = coordinate if coordinate is not None else FieldCoordinate()

    # ------------------------------------------------------------------
    # Coordinate convenience (FieldCoordinate only)
    # ------------------------------------------------------------------
    def set_coordinate(
        self: Self,
        center_pos: tuple | list | np.ndarray | None = None,
        angles: tuple[float, float, float] | None = None,
        scales: tuple | list | np.ndarray | None = None,
    ) -> Self:
        """
        Update :class:`FieldCoordinate` parameters in-place (fluent API).

        Raises ``TypeError`` if :attr:`coordinate` is not a
        :class:`FieldCoordinate`; in that case, assign :attr:`coordinate`
        directly.

        Parameters
        ----------
        center_pos : array_like, shape (3,), optional
        angles : (float, float, float), optional  — XYZ Euler angles in radians
        scales : array_like, shape (3,), optional — per-axis scale factors (> 0)
        """
        if not isinstance(self.coordinate, FieldCoordinate):
            raise TypeError(
                "set_coordinate only supports FieldCoordinate. Assign self.coordinate directly for other transforms."
            )
        if center_pos is not None:
            self.coordinate.center_pos = np.asarray(center_pos, dtype=float)
        if angles is not None:
            self.coordinate.angles = tuple(angles)  # type: ignore[assignment]
        if scales is not None:
            self.coordinate.scales = np.asarray(scales, dtype=float)
        return self

    # ------------------------------------------------------------------
    # DensitySource protocol
    # ------------------------------------------------------------------
    def _evaluate_density(self, pos: np.ndarray) -> np.ndarray:
        pos_ell = self.coordinate.world_to_elliptical(pos)
        return self._evaluate_density_generic(pos_ell)

    def _evaluate_density_generic(self, pos_elliptical: np.ndarray) -> np.ndarray:
        """
        Evaluate the model density in elliptical coordinates.

        Parameters
        ----------
        pos_elliptical : ndarray, shape (m, 3)
            Positions in the model's elliptical frame.

        Returns
        -------
        ndarray, shape (m,)

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement _evaluate_density_generic.")

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def __add__(
        self, other: TheoreticalDensityDistribution | CombinedDensityDistribution
    ) -> CombinedDensityDistribution:
        if isinstance(other, CombinedDensityDistribution):
            return CombinedDensityDistribution(self, *other.components)
        if isinstance(other, TheoreticalDensityDistribution):
            return CombinedDensityDistribution(self, other)
        raise NotImplementedError("Addition is only supported between TheoreticalDensityDistribution instances.")


# ======================================================================
# CombinedDensityDistribution
# ======================================================================
class CombinedDensityDistribution(TheoreticalDensityDistribution):
    """
    Sum of two or more :class:`TheoreticalDensityDistribution` components.

    Created automatically via the ``+`` operator.

    Parameters
    ----------
    *components : TheoreticalDensityDistribution
    """

    def __init__(self, *components: TheoreticalDensityDistribution) -> None:
        # No shared coordinate; each component owns its own transform.
        self.components = components

    def _evaluate_density(self, pos: np.ndarray) -> np.ndarray:
        return sum(c._evaluate_density(pos) for c in self.components)  # type: ignore[return-value]


# ======================================================================
# Concrete models
# ======================================================================
class PlummerSphere(TheoreticalDensityDistribution):
    """
    Plummer (1911) density model.

    Parameters
    ----------
    total_mass : float
    scale_radius : float  — Plummer scale radius *b*
    coordinate : CoordinateTransform, optional
    """

    def __init__(self, total_mass: float, scale_radius: float, coordinate: CoordinateTransform | None = None) -> None:
        super().__init__(coordinate)
        self.total_mass = total_mass
        self.scale_radius = scale_radius

    def _evaluate_density_generic(self, pos: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(pos, axis=-1)
        b, M = self.scale_radius, self.total_mass
        return (3 * M * b**2 / (4 * np.pi)) / (r**2 + b**2) ** 2.5


class DoubleExponentialDisk(TheoreticalDensityDistribution):
    """
    Double-exponential disk: ``ρ ∝ exp(-R/h) exp(-|z|/z₀)``.

    Parameters
    ----------
    sigma0 : float        — central surface density
    scale_length : float  — radial scale length *h*
    scale_height : float  — vertical scale height *z₀*
    coordinate : CoordinateTransform, optional
    """

    def __init__(
        self, sigma0: float, scale_length: float, scale_height: float, coordinate: CoordinateTransform | None = None
    ) -> None:
        super().__init__(coordinate)
        self.sigma0 = sigma0
        self.scale_length = scale_length
        self.scale_height = scale_height

    def _evaluate_density_generic(self, pos: np.ndarray) -> np.ndarray:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        R = np.sqrt(x**2 + y**2)
        h, z0 = self.scale_length, self.scale_height
        return self.sigma0 / (2 * z0) * np.exp(-R / h) * np.exp(-np.abs(z) / z0)


class PowerLawSphere(TheoreticalDensityDistribution):
    """
    Hernquist-style power-law density model.

    Parameters
    ----------
    total_mass : float
    scale_radius : float  — scale radius *b*
    slope : float         — inner logarithmic slope *γ*
    coordinate : CoordinateTransform, optional
    """

    def __init__(
        self, total_mass: float, scale_radius: float, slope: float, coordinate: CoordinateTransform | None = None
    ) -> None:
        super().__init__(coordinate)
        self.total_mass = total_mass
        self.scale_radius = scale_radius
        self.slope = slope

    def _evaluate_density_generic(self, pos: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(pos, axis=-1)
        b, M, gamma = self.scale_radius, self.total_mass, self.slope
        coeff = (3 - gamma) * M / (4 * np.pi * b**3)
        denom = (r / b) ** gamma * (1 + r / b) ** (4 - gamma)
        return coeff / denom
