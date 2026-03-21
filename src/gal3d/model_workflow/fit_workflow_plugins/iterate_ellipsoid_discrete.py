"""
Iterative ellipsoidal shape estimation workflow plugin for Gal3D.

This module implements an iterative mass–moment (inertia tensor) method to
estimate the 3‑D ellipsoidal shape of a particle distribution. The basic idea
is:

1. Select particles within a radial shell or enclosed ellipsoid.
2. Compute the (possibly weighted) inertia tensor of the selected particles.
3. Diagonalize the tensor to obtain the principal axes and axis lengths.
4. Update the trial ellipsoid and repeat from step 1 until convergence.

Two families of methods are supported:

* **Shell methods (S1–S3)** – particles are selected in an *ellipsoidal shell*.
* **Enclosed methods (E1–E3)** – particles are selected inside an *enclosed
  ellipsoid*.

Within each family, three weighting schemes :math:`w(r)` can be used for the
mass when computing the inertia tensor:

* ``None`` / unweighted  (``w = 1``)
* ``"r2"``     – spherical radius weighting :math:`w \\propto r^{-2}`
* ``"rell2"``  – ellipsoidal radius weighting
  :math:`w \\propto r_\\mathrm{ell}^{-2}`,
  where

  .. math::

      r_\\mathrm{ell}^2 = x^2 +
                         \frac{y^2}{(b/a)^2} +
                         \frac{z^2}{(c/a)^2}.

Both families and all weightings are available through the
:class:`IterateEllipsoidWorkflow`.

Notes
-----
This implementation is inspired by the iterative shape measurement in
``pynbody.analysis.halo.shape``(https://github.com/pynbody/pynbody/blob/master/pynbody/analysis/halo.py#L483)
and follows the ideas discussed in:

    Zemp, M., Gnedin, O. Y., Gnedin, N. Y., & Kravtsov, A. V. (2011),
    *On Determining the Shape of Matter Distributions*, ApJS, 197(2), 30,
    https://doi.org/10.1088/0067-0049/197/2/30


Examples
--------
A minimal usage example with a :class:`gal3d.point.Particles` instance
``particles``::

    from gal3d.model_workflow.fit_workflow_plugins.iterate_ellipsoid_discrete import (
        IterateEllipsoidParticles,
    )

    workflow = IterateEllipsoidParticles()

    # S1: ellipsoidal shell, unweighted
    result_s1 = workflow(
        particles,
        r=np.geomspace(rmin, rmax, nbins),
        max_iterations=20,
        tol=1e-3,
        is_enclosed=False,        # shell
        shell_frac=0.1,
        weight_method=None,       # w = 1
    )

    # E3: enclosed ellipsoid, w ~ r_ell^{-2}
    result_e3 = workflow(
        particles,
        r=np.geomspace(rmin, rmax, nbins),
        max_iterations=20,
        tol=1e-3,
        is_enclosed=True,         # enclosed ellipsoid
        weight_method="rell2",    # w ~ r_ell^{-2}
    )

The returned :class:`gal3d.optimization.result.ModelResult` contains the best‑fit
axis length ``a`` and axis ratios ``eps_ab``, ``eps_bc``, the three Euler
angles ``ang1``, ``ang2``, ``ang3``, and an estimate of their iteration
errors.

"""

import logging
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as _ScipyRotation

from gal3d.model_workflow.fit_workflow import FitWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.point.util import abc_vect
from gal3d.util.errors import InsufficientPointsError

from .util import EllipsoidResultBuilder

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer
    from gal3d.point import Particles

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
    def condition(obj: Union["Gal3DAnalyzer", "Particles"]) -> bool:
        if type(obj).__name__ == "Particles":
            logger.debug("Select IterateEllipsoidParticles for Particles")
            return True
        raise TypeError("Unsupported object type")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_weight(
        self,
        weight_method: Literal["r2", "rell2"] | None,
        mass: ArrayF,
        r_sph: ArrayF,
        r_ell: ArrayF,
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
    def _update_axes(a_old: ArrayF, a_new_raw: ArrayF) -> ArrayF:
        a_new = np.sqrt(np.abs(a_new_raw) * 3.0)
        scale = (np.prod(a_old) / np.prod(a_new)) ** (1.0 / 3.0)
        return a_new * scale

    @staticmethod
    def _compute_shell_density(
        sel_mass: ArrayF,
        abc: ArrayF,
        shell_frac: float,
        is_enclosed: bool,
    ) -> float:
        """Mean volumetric density in the selected region."""
        vol_unit = (4.0 / 3.0) * np.pi * np.prod(abc)
        if is_enclosed:
            vol = vol_unit
        else:
            vol = vol_unit * ((1.0 + shell_frac) ** 3 - (1.0 - shell_frac) ** 3)
        return float(np.sum(sel_mass) / vol) if vol > 0 else 0.0


    def _extract_particle_data(self, obj: "Particles") -> tuple[ArrayF, ArrayF, ArrayF]:
        """Extract pos, mass, r from a Particles."""
        return (
            np.asarray(obj.pos, dtype=float),
            np.asarray(obj.mass, dtype=float),
            np.asarray(obj.r, dtype=float),)

    def _init_trial_ellipsoid(
        self, r: float, init_parameters: dict | None, volume_conserve: bool
    ) -> tuple[ArrayF, ArrayF]:
        """Build initial axes a and rotation matrix R from warm-start params."""
        abc = np.ones(3, dtype=float) * r
        abc_init = np.ones(3, dtype=float)
        R: ArrayF = np.eye(3, dtype=float)
        if init_parameters:
            abc_init[1] = (1.0 - init_parameters.get("eps_ab", 0.0)) * abc_init[0]
            abc_init[2] = (1.0 - init_parameters.get("eps_bc", 0.0)) * abc_init[1]
            ang1 = float(init_parameters.get("ang1", 0.0))
            ang2 = float(init_parameters.get("ang2", 0.0))
            ang3 = float(init_parameters.get("ang3", 0.0))
            R = np.array(_ScipyRotation.from_euler("zyx", [ang1, ang2, ang3]).as_matrix(),dtype=float,)
        a: ArrayF = self._to_new_ellipsoid(abc, abc_init, volume_conservation=volume_conserve)
        return a, R

    # ------------------------------------------------------------------
    # Core fit at a single radius
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        obj: "Particles",
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
        obj : Gal3DAnalyzer or Particles
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
            major axis ``a`` is held fixed (following the warm-start value).
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
        # --- extract particles ---
        pos, mass, p_r = self._extract_particle_data(obj)
        # --- initialise axes & rotation ---
        a, R = self._init_trial_ellipsoid(r, init_parameters, volume_conserve)
        # --- iteration loop ---
        a_prev: ArrayF = a.copy()
        R_prev: ArrayF = R.copy()
        iteration_counter = 0
        err = tol + 1.0
        sel_mass: ArrayF = np.empty(0, dtype=float)
        while err > tol and iteration_counter < max_iterations:
            a_prev = a.copy()
            R_prev = R.copy()
            # --- spherical pre-filter: generous conservative cut ---
            if is_enclosed:
                pre_mask = p_r < a[0] * (1.0 + shell_frac)
            else:
                pre_mask = ((p_r > a[2] * (1.0 - shell_frac))& (p_r < a[0] * (1.0 + shell_frac)))
            if not np.any(pre_mask):
                break
            pre_pos = pos[pre_mask]
            pre_mass = mass[pre_mask]
            pre_r = p_r[pre_mask]
            # --- ellipsoid function via quick_call ---
            # returns (f, r_transformed), where f = x²/a² + y²/b² + z²/c²
            ang = self._default_structure._coordinate.mat_to_angle(R)  # type: ignore[attr-defined]
            eps_ab = 1.0 - a[1] / a[0]
            eps_bc = 1.0 - a[2] / a[1]
            f, _ = self._default_structure.quick_call(
                *ang, a[0], eps_ab, eps_bc, pos=pre_pos
            )
            # r_ell = a * sqrt(f)  <==>  sqrt(x² + y²/(b/a)² + z²/(c/a)²)
            r_ell = a[0] * np.sqrt(np.clip(f, 0.0, None))
            # --- particle selection ---
            if is_enclosed:
                sel_mask = f < 1.0
            else:
                lo = (1.0 - shell_frac) * a[0]
                hi = (1.0 + shell_frac) * a[0]
                sel_mask = (r_ell >= lo) & (r_ell <= hi)
            n_sel = int(np.sum(sel_mask))
            if n_sel < min_particles:
                raise InsufficientPointsError(f"Only {n_sel} particles selected at r={r:.4g} "
                    f"(minimum {min_particles}). shell_frac={shell_frac}, is_enclosed={is_enclosed}.")
            sel_pos = pre_pos[sel_mask]
            sel_mass = pre_mass[sel_mask]
            sel_r = pre_r[sel_mask]
            sel_r_ell = r_ell[sel_mask]
            # --- weighted inertia tensor ---
            eff_mass = self._apply_weight(weight_method, sel_mass, sel_r, sel_r_ell)
            abc_new, axes = abc_vect(sel_pos, eff_mass)
            R = self._sanitize_rotation(np.array(axes, dtype=float))
            # convert eigenvalues to axis lengths, then rescale per volume_conserve
            a_new = np.sqrt(np.abs(abc_new) * 3.0)
            a = self._to_new_ellipsoid(a, a_new, volume_conservation=volume_conserve)
            iteration_counter += 1
            err = self._axis_ratio_error(a, a_prev)
        shell_density = self._compute_shell_density(sel_mass, a, shell_frac, is_enclosed)
        return self._build_model_result(
            a, R, a_prev, R_prev, iteration_counter, err, shell_density
        )
