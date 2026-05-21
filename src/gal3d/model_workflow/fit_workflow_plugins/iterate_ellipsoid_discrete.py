r"""
Iterative ellipsoidal shape estimation workflow plugin for Gal3D.

This module implements an iterative mass-moment, or inertia-tensor, method for
estimating the 3-D ellipsoidal shape of a particle distribution. The basic
workflow is:

1. Select particles within a radial shell or enclosed ellipsoid.
2. Compute the weighted inertia tensor of the selected particles.
3. Diagonalize the tensor to obtain the principal axes and axis lengths.
4. Update the trial ellipsoid and repeat until convergence.

Two method families are supported:

* Shell methods (S1-S3), which operate on ellipsoidal shells.
* Enclosed methods (E1-E3), which operate on enclosed ellipsoids.

Within each family, three weighting schemes can be used when computing the
tensor:

* ``None`` for an unweighted tensor, with ``w = 1``.
* ``"r2"`` for spherical-radius weighting, with :math:`w \propto r^{-2}`.
* ``"rell2"`` for ellipsoidal-radius weighting, with
  :math:`w \propto r_{\mathrm{ell}}^{-2}`.

For the ellipsoidal-radius weighting, the radius is

.. math::

   r_{\mathrm{ell}}^2 = x^2 + \frac{y^2}{(b/a)^2} + \frac{z^2}{(c/a)^2}.

Notes
-----
This implementation is inspired by the iterative shape measurement in
`pynbody.analysis.halo.shape <https://github.com/pynbody/pynbody/blob/master/pynbody/analysis/halo.py#L483>`_
and follows the discussion in Zemp et al. (2011), *On Determining the Shape of
Matter Distributions*, ApJS, 197(2), 30,
https://doi.org/10.1088/0067-0049/197/2/30

Examples
--------
A minimal usage example with a :class:`gal3d.point.Particles` instance named
``particles``::

    from gal3d.model_workflow.fit_workflow_plugins.iterate_ellipsoid_discrete import IterateEllipsoidParticles

    workflow = IterateEllipsoidParticles()

    result_s1 = workflow(
        particles,
        r=np.geomspace(rmin, rmax, nbins),
        max_iterations=20,
        tol=1e-3,
        is_enclosed=False,
        shell_frac=0.1,
        weight_method=None,
    )

    result_e3 = workflow(
        particles,
        r=np.geomspace(rmin, rmax, nbins),
        max_iterations=20,
        tol=1e-3,
        is_enclosed=True,
        weight_method="rell2",
    )
"""

import logging
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as _ScipyRotation

from gal3d.model_workflow.fit_workflow import FitInput, FitWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.point.util import abc_vect
from gal3d.util.errors import InsufficientPointsError

from .util import EllipsoidResultBuilder

logger = logging.getLogger("gal3d.fit_workflow_plugins")

# Type aliases
ArrayF = NDArray[np.floating[Any]]
ArrayI = NDArray[np.int_]


class IterateEllipsoidParticles(FitWorkflowBase, EllipsoidResultBuilder):
    """
    Workflow for estimating ellipsoidal shape using an iterative mass–moment
    (inertia tensor) method.

    At each target radius ``r`` (the equivalent radius :math:`(abc)^{1/3}`),
    the algorithm:

    1. Initialises a trial ellipsoid :math:`(a, b, c) = (r, r, r)`.
    2. Evaluates the ellipsoid function :math:`f = x^2/a^2 + y^2/b^2 + z^2/c^2`
       via ``quick_call``.
    3. Derives the ellipsoidal radius
       :math:`r_\\mathrm{ell} = a\\sqrt{f}`.
    4. Selects particles in the ellipsoidal shell
       :math:`(1-\\delta)a \\le r_\\mathrm{ell} \\le (1+\\delta)a`
       (or enclosed ellipsoid :math:`f < 1`).
    5. Computes the weighted inertia tensor, diagonalises, and updates axes.
    6. Repeats until axis-ratio convergence or ``max_iterations`` reached.
    """

    @staticmethod
    def condition(obj: FitInput) -> bool:
        if type(obj).__name__ == "Particles":
            logger.debug("Select IterateEllipsoidParticles for Particles")
            return True
        raise TypeError("Unsupported object type")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_weight(
        self, weight_method: Literal["r2", "rell2"] | None, mass: ArrayF, r_sph: ArrayF, r_ell: ArrayF
    ) -> ArrayF:
        if weight_method == "r2":
            return mass / np.maximum(r_sph, 1e-9) ** 2
        if weight_method == "rell2":
            return mass / np.maximum(r_ell, 1e-9) ** 2
        return mass

    @staticmethod
    def _sanitize_rotation(R: ArrayF) -> ArrayF:
        if np.linalg.det(R) < 0:
            R = -R
        return R

    @staticmethod
    def _compute_shell_density(sel_mass: ArrayF, abc: ArrayF, shell_frac: float, is_enclosed: bool) -> float:
        """Mean volumetric density in the selected region."""
        vol_unit = (4.0 / 3.0) * np.pi * np.prod(abc)
        if is_enclosed:
            vol = vol_unit
        else:
            vol = vol_unit * ((1.0 + shell_frac) ** 3 - (1.0 - shell_frac) ** 3)
        return float(np.sum(sel_mass) / vol) if vol > 0 else 0.0

    def _extract_particle_data(self, obj: FitInput) -> tuple[ArrayF, ArrayF, ArrayF]:
        """Extract pos, mass, r from a Particles instance."""
        particles = obj.density_source if hasattr(obj, "density_source") else obj
        try:
            return (particles.pos, particles.mass, particles.r)  # type: ignore
        except AttributeError as e:
            raise TypeError("Expected a Particles input with pos, mass, r attributes") from e

    def _init_trial_ellipsoid(
        self, r: float, init_parameters: dict | None, volume_conserve: bool
    ) -> tuple[ArrayF, ArrayF]:
        """Build initial axes ``a`` and rotation matrix ``R`` from warm-start params."""
        abc = np.ones(3, dtype=float) * r
        abc_init = np.ones(3, dtype=float)
        R: ArrayF = np.eye(3, dtype=float)
        if init_parameters:
            abc_init[1] = (1.0 - init_parameters.get("eps_ab", 0.0)) * abc_init[0]
            abc_init[2] = (1.0 - init_parameters.get("eps_bc", 0.0)) * abc_init[1]
            ang1 = float(init_parameters.get("ang1", 0.0))
            ang2 = float(init_parameters.get("ang2", 0.0))
            ang3 = float(init_parameters.get("ang3", 0.0))
            R = np.array(_ScipyRotation.from_euler("zyx", [ang1, ang2, ang3]).as_matrix(), dtype=float)
        a: ArrayF = self._to_new_ellipsoid(abc, abc_init, volume_conservation=volume_conserve)
        return a, R

    def _prefilter_spherical(
        self, pos: ArrayF, mass: ArrayF, r_p: ArrayF, a: ArrayF, shell_frac: float, is_enclosed: bool
    ) -> tuple[ArrayF, ArrayF, ArrayF]:
        """Conservative spherical cut to reduce work before the ellipsoid call."""
        if is_enclosed:
            mask = r_p < a[0] * (1.0 + shell_frac)
        else:
            mask = (r_p > a[2] * (1.0 - shell_frac)) & (r_p < a[0] * (1.0 + shell_frac))
        return pos[mask], mass[mask], r_p[mask]

    def _select_ellipsoidal(
        self, f: ArrayF, r_ell: ArrayF, a: ArrayF, shell_frac: float, is_enclosed: bool, min_particles: int, r: float
    ) -> NDArray[np.bool_]:
        """Return the boolean selection mask and validate the particle count."""
        if is_enclosed:
            sel_mask = f < 1.0
        else:
            lo = (1.0 - shell_frac) * a[0]
            hi = (1.0 + shell_frac) * a[0]
            sel_mask = (r_ell >= lo) & (r_ell <= hi)

        n_sel = int(np.sum(sel_mask))
        if n_sel < min_particles:
            raise InsufficientPointsError(
                f"Only {n_sel} particles selected at r={r:.4g} "
                f"(minimum {min_particles}). "
                f"shell_frac={shell_frac}, is_enclosed={is_enclosed}."
            )
        return sel_mask

    def _iterate_once(
        self,
        pos: ArrayF,
        mass: ArrayF,
        r_p: ArrayF,
        a: ArrayF,
        R: ArrayF,
        *,
        shell_frac: float,
        is_enclosed: bool,
        weight_method: Literal["r2", "rell2"] | None,
        min_particles: int,
        volume_conserve: bool,
        r: float,
    ) -> tuple[ArrayF, ArrayF, ArrayF]:
        """
        One full iteration: pre-filter → quick_call → select → inertia tensor
        → axis update.

        Returns
        -------
        a_new : ArrayF
            Updated semi-axes.
        R_new : ArrayF
            Updated rotation matrix.
        sel_mass : ArrayF
            Masses of the selected particles (needed for shell density).
        """
        # --- spherical pre-filter ---
        pre_pos, pre_mass, pre_r_p = self._prefilter_spherical(pos, mass, r_p, a, shell_frac, is_enclosed)
        if pre_pos.size == 0:
            raise InsufficientPointsError(f"No particles survived spherical pre-filter at r={r:.4g}.")

        # --- ellipsoid function: f = x²/a² + y²/b² + z²/c² ---
        ang = self._default_structure._coordinate.mat_to_angle(R)  # type: ignore[attr-defined]
        eps_ab = 1.0 - a[1] / a[0]
        eps_bc = 1.0 - a[2] / a[1]
        f, _ = self._default_structure.quick_call(*ang, a[0], eps_ab, eps_bc, pos=pre_pos)

        # r_ell = a * sqrt(f)  <==>  sqrt(x² + y²/(b/a)² + z²/(c/a)²)
        r_ell = a[0] * np.sqrt(np.clip(f, 0.0, None))

        # --- particle selection ---
        sel_mask = self._select_ellipsoidal(f, r_ell, a, shell_frac, is_enclosed, min_particles, r)
        sel_pos = pre_pos[sel_mask]
        sel_mass = pre_mass[sel_mask]
        sel_r_p = pre_r_p[sel_mask]
        sel_r_ell = r_ell[sel_mask]

        # --- weighted inertia tensor → new axes ---
        eff_mass = self._apply_weight(weight_method, sel_mass, sel_r_p, sel_r_ell)
        abc_new, axes = abc_vect(sel_pos, eff_mass)
        R_new = self._sanitize_rotation(np.array(axes, dtype=float))
        a_new_raw = np.sqrt(np.abs(abc_new) * 3.0)
        a_new = self._to_new_ellipsoid(a, a_new_raw, volume_conservation=volume_conserve)

        return a_new, R_new, sel_mass

    # ------------------------------------------------------------------
    # Core fit at a single radius
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        obj: FitInput,
        r: float,
        *,
        max_iterations: int = 20,
        tol: float = 1e-3,
        is_enclosed: bool = False,
        weight_method: Literal["r2", "rell2"] | None = None,
        shell_frac: float = 0.1,
        min_particles: int = 12,
        volume_conserve: bool = False,
        init_parameters: dict | None = None,
        **kwargs: Any,
    ) -> ModelResult:
        """
        Fit the ellipsoidal shape at a single target radius ``r``.

        Parameters
        ----------
        obj : FitInput
            Input object containing particle data.
        r : float
            Target equivalent radius :math:`(abc)^{1/3}`.
        max_iterations : int, optional
            Maximum iterations per radius (default 20).
        tol : float, optional
            Convergence tolerance on axis-ratio changes (default 1e-3).
        is_enclosed : bool, optional
            If ``False`` (default) select an ellipsoidal *shell*; if
            ``True`` select the enclosed ellipsoid (:math:`f < 1`).
        weight_method : {None, 'r2', 'rell2'}, optional
            Weighting applied to particle masses:

            * ``None``    – uniform (S1 / E1)
            * ``'r2'``    – spherical :math:`r^{-2}` (S2 / E2)
            * ``'rell2'`` – ellipsoidal :math:`r_\\mathrm{ell}^{-2}` (S3 / E3)
        shell_frac : float, optional
            Half-width of the ellipsoidal shell as a fraction of ``a``
            (default 0.1, i.e. the shell spans
            :math:`[0.9a,\\,1.1a]` in :math:`r_\\mathrm{ell}`).
        min_particles : int, optional
            Minimum selected particles required; raises
            :exc:`~gal3d.util.errors.InsufficientPointsError` if fewer
            are found (default 12).
        volume_conserve : bool, optional
            If ``True``, rescale axes each iteration so that
            :math:`(abc)^{1/3} = r` is preserved.  If ``False`` (default), the
            major axis ``a`` is held fixed.
        init_parameters : dict, optional
            Warm-start: supplies the axis *ratios* and *orientation* from the
            previous radius.  The size is always reset so that
            :math:`(abc)^{1/3} = r`.  Recognised keys: ``eps_ab``,
            ``eps_bc``, ``ang1``, ``ang2``, ``ang3``.

        Returns
        -------
        ModelResult

        Raises
        ------
        InsufficientPointsError
            If fewer than ``min_particles`` particles are selected.
        """
        pos, mass, r_p = self._extract_particle_data(obj)
        a, R = self._init_trial_ellipsoid(r, init_parameters, volume_conserve)

        a_prev: ArrayF = a.copy()
        R_prev: ArrayF = R.copy()
        sel_mass: ArrayF = np.empty(0, dtype=float)

        for _n_iter in range(max_iterations):
            a_prev, R_prev = a.copy(), R.copy()
            a, R, sel_mass = self._iterate_once(
                pos,
                mass,
                r_p,
                a,
                R,
                shell_frac=shell_frac,
                is_enclosed=is_enclosed,
                weight_method=weight_method,
                min_particles=min_particles,
                volume_conserve=volume_conserve,
                r=r,
            )
            if self._axis_ratio_error(a, a_prev) <= tol:
                break

        err = self._axis_ratio_error(a, a_prev)
        shell_density = self._compute_shell_density(sel_mass, a, shell_frac, is_enclosed)
        return self._build_model_result(a, R, a_prev, R_prev, _n_iter + 1, err, shell_density)
