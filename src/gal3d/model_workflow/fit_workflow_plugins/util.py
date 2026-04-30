
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gal3d.optimization.optimizer import OptimizeResult
from gal3d.optimization.result import ModelResult
from gal3d.shape import StructureCore

ArrayF = NDArray[np.floating[Any]]


def _periodic_diff(a: float, b: float, period: float = 2 * np.pi) -> float:
    """
    Minimal absolute difference between two angles on a circle.

    Parameters
    ----------
    a, b : float
        Angles in radians.
    period : float, optional
        Period of the angles (default is 2π).

    Returns
    -------
    float
        Minimal absolute difference between a and b, accounting for periodicity.
    """
    d = (a - b + 0.5 * period) % period - 0.5 * period
    return float(np.abs(d))


def _prepare_bins(
    r: np.ndarray,
    rmin: float,
    rmax: float,
    nbins: int,
    bins: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare radial bin edges and representative radii.

    Parameters
    ----------
    r : ndarray of float
        Particle radii.
    rmin, rmax : float
        Minimum and maximum radii for binning.
    nbins : int
        Number of radial bins.
    bins : {'equal', 'log', 'lin'}
        Binning scheme:
        * ``'equal'`` – equal number of particles per bin
        * ``'log'``   – logarithmically spaced in radius
        * ``'lin'``   – linearly spaced in radius

    Returns
    -------
    bin_edges : ndarray of float
        Array of length ``nbins + 1`` with bin edges.
    rbins : ndarray of float
        Representative radius for each bin (e.g. geometric mean).
    """

    def equal_bins(r: np.ndarray, N: int) -> np.ndarray:
        sorted_r = np.sort(r[(r >= rmin) & (r <= rmax)])
        step = max(len(sorted_r) // N, 1)
        return np.array(
            [sorted_r[i * step] for i in range(N)] + [sorted_r[-1]],
            dtype=float,
        )

    if bins == "equal":
        full = equal_bins(r, nbins * 2)
        bin_edges = full[0::2]
        rbins     = full[1::2]
    elif bins == "log":
        bin_edges = np.geomspace(rmin, rmax, nbins + 1)
        rbins     = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif bins == "lin":
        bin_edges = np.linspace(rmin, rmax, nbins + 1)
        rbins     = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        raise ValueError(f"Unknown bins scheme: {bins!r}")

    return bin_edges.astype(float), rbins.astype(float)



class EllipsoidResultBuilder:
    """
    Mixin that provides :meth:`_build_model_result` for iterative ellipsoid
    fitting workflows.

    Both :class:`IterateEllipsoidParticles` and
    :class:`IterateEllipsoidDensity` inherit from this class so that the
    result-packaging logic is defined in exactly one place.

    The method packs the converged semi-axes, rotation matrix and mean
    shell density into a :class:`~gal3d.optimization.result.ModelResult`,
    attaching per-parameter iteration uncertainties (absolute change between
    the last two iterations) to every fitted parameter.

    Parameters stored in the result
    --------------------------------
    a : float
        Length of the longest semi-axis.
    eps_ab : float
        Ellipticity :math:`1 - b/a`.
    eps_bc : float
        Ellipticity :math:`1 - c/b`.
    ang1, ang2, ang3 : float
        Euler angles of the principal-axis frame in the lab frame.
    info : float
        Mean volumetric density inside the shell,
        :math:`\\bar{\\rho} = M_\\mathrm{shell} / V_\\mathrm{shell}`.
    """
    _default_structure = StructureCore("RotateOnly", "Ellipsoid")
    _min_axis_ratio = 0.001 # avoid unphysical axis ratios and numerical instability

    @staticmethod
    def _axis_ratio_error(a: ArrayF, a_prev: ArrayF) -> float:
        return float(np.abs(a[2]*a_prev[0]/ (a[0]*a_prev[2]) - 1.0)
                    + np.abs(a[1]*a_prev[0]/ (a[0]*a_prev[1]) - 1.0)
        )

    @classmethod
    def _to_new_ellipsoid(cls, a_old: ArrayF, a_new: ArrayF, volume_conservation: bool = True) -> ArrayF:
        if volume_conservation:
            # Rescale the new axes to preserve the volume of the ellipsoid,
            # which can help stabilize the iteration when the axis ratios are far from unity.
            old_vol = np.prod(a_old)
            new_vol = np.prod(a_new)
            scale = (old_vol / new_vol) ** (1/3)
            return a_new * scale
        else:
            # fix major axis (a_old[0]), update b/a and c/a from a_new
            a0 = abs(float(a_old[0]))
            b_a = np.clip(float(a_new[1]/a_new[0]), cls._min_axis_ratio, 1.0)
            c_a = np.clip(float(a_new[2]/a_new[0]), cls._min_axis_ratio, 1.0)
            return np.array([a0, b_a*a0, c_a*a0], dtype=float)

    @classmethod
    def _build_model_result(
        cls,
        abc: np.ndarray,
        rot: np.ndarray,
        abc_prev: np.ndarray,
        rot_prev: np.ndarray,
        n_iter_done: int,
        err: float,
        shell_density: float,
        extra_info: dict[str, Any] | None = None,
    ) -> ModelResult:
        """
        Package ellipsoid iteration results into a ``ModelResult``.

        Parameters
        ----------
        abc : ndarray of shape (3,)
            Converged semi-axes :math:`(a \\geq b \\geq c)`.
        rot : ndarray of shape (3, 3)
            Converged rotation matrix; columns are principal axes in the
            lab frame.
        abc_prev : ndarray of shape (3,)
            Semi-axes from the **penultimate** iteration, used to estimate
            parameter uncertainties.
        rot_prev : ndarray of shape (3, 3)
            Rotation matrix from the penultimate iteration.
        n_iter_done : int
            Number of iterations actually performed.
        err : float
            Final axis-ratio convergence measure (stored as ``cost``).
        shell_density : float
            Mean volumetric density of the shell.  For the discrete
            workflow this is :math:`M_\\mathrm{shell}/V_\\mathrm{shell}`;
            for the continuous workflow it is the mean surface density
            :math:`\\oint \\rho\\,d\\Omega / \\oint d\\Omega`.
            Stored via ``params.add_info``.
        extra_info : dict, optional
            Any extra info to be added to the result parameters

        Returns
        -------
        ModelResult
        """
        # ---- axis length ----
        params = cls._default_structure.parameters.deepcopy()
        params["a"] = abc[0]
        params.get_parameter("a").err = float(np.abs(abc[0] - abc_prev[0]))

        # ---- ellipticities ----
        params["eps_ab"] = 1.0 - abc[1] / abc[0]
        params.get_parameter("eps_ab").err = float(
            np.abs(abc[1] / abc[0] - abc_prev[1] / abc_prev[0])
        )

        params["eps_bc"] = 1.0 - abc[2] / abc[1]
        params.get_parameter("eps_bc").err = float(
            np.abs(abc[2] / abc[1] - abc_prev[2] / abc_prev[1])
        )

        # ---- orientation angles ----
        ang      = cls._default_structure._coordinate.mat_to_angle(rot)       # type: ignore
        ang_prev = cls._default_structure._coordinate.mat_to_angle(rot_prev)  # type: ignore
        params["ang1"], params["ang2"], params["ang3"] = ang
        for k, (aa, bb) in enumerate(zip(ang, ang_prev, strict=False), start=1):
            # for ellipsoid, angles are periodic with period π/2 (not 2π) because of the symmetry
            params.get_parameter(f"ang{k}").err = _periodic_diff(aa, bb, period = np.pi/2)

        # ---- pack result ----
        params.add_info(parameter=shell_density)
        if extra_info is not None:
            params.add_info(**extra_info)

        opt = OptimizeResult(
            params=params,
            fun=None,
            start_fun=None,
            start_params=None,
            n_iterations=n_iter_done,
            cost=err,
        )
        return ModelResult(cls._default_structure, opt, params)
