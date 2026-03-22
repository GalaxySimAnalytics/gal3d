from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from gal3d.util.array_operate import Auto3DShape

if TYPE_CHECKING:
    from gal3d.model_workflow.fit_workflow import FitWorkflowBase
    from gal3d.optimization.result import ModelResult
    from gal3d.visualization.show import ImageData

class DensitySource(Auto3DShape):
    """
    Base class for density estimation, providing a common interface for
    different density estimation methods.

    Subclasses must implement :meth:`_evaluate_density`.

    The default :meth:`project_2d` delegates to :meth:`_project_2d_los`,
    which performs adaptive Simpson line-of-sight integration.  Subclasses
    may override :meth:`project_2d` with a faster algorithm (e.g. SPH
    rendering for particle data) while :meth:`_project_2d_los` remains
    available as a fallback.
    """

    def __call__(self, pos: ArrayLike) -> np.ndarray:
        """
        Evaluate density at one or more positions.

        Parameters
        ----------
        pos : array_like, shape (3,) or (m, 3)
            Query position(s) in 3-D Cartesian coordinates.

        Returns
        -------
        numpy.ndarray
            Density value(s).  Returns a scalar array of shape ``()`` when
            *pos* is 1-D, or shape ``(m,)`` when *pos* is 2-D.
        """
        pos3d = self.to_3d_array(pos)
        is_1d = np.array(pos).ndim == 1
        density = self._evaluate_density(pos3d)
        if is_1d:
            return density[0]
        return density

    def _evaluate_density(self, pos: np.ndarray) -> np.ndarray:
        """
        Evaluate density at an array of 3-D positions.

        Parameters
        ----------
        pos : ndarray, shape (m, 3)
            Query positions in 3-D Cartesian coordinates.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Density values at each position.

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # ------------------------------------------------------------------
    # Line-of-sight projection (always available, never overridden)
    # ------------------------------------------------------------------

    def project_2d_los(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        resolution: int = 200,
        rotation_matrix: np.ndarray | None = None,
        z_range: tuple[float, float] | None = None,
        nz_min: int = 33,
        nz_max: int = 4097,
        rtol: float = 1e-4,
        atol: float = 1e-8,
    ) -> "ImageData":
        """
        Project the 3-D density along the *z*-axis using adaptive Simpson
        line-of-sight integration.

        This method is **never overridden** by subclasses; it is the
        authoritative LOS implementation.  :meth:`project_2d` calls this by
        default and may be overridden to use a faster algorithm.

        Parameters
        ----------
        x_range : (float, float)
            Minimum and maximum extent along the *x*-axis of the output
            image, in the same physical units as the positions.
        y_range : (float, float)
            Minimum and maximum extent along the *y*-axis of the output
            image.
        resolution : int, optional
            Number of pixels along each axis.  Default ``200``.
        rotation_matrix : ndarray, shape (3, 3), optional
            Rotation applied to query positions before evaluating the
            density.  ``None`` (default) means no rotation.
        z_range : (float, float), optional
            Integration limits along the line of sight.  If ``None`` a
            symmetric range ``[-max(Δx, Δy), +max(Δx, Δy)]`` is used.
        nz_min : int, optional
            Minimum number of *z* samples (odd) for the base Simpson grid.
            Default ``33``.
        nz_max : int, optional
            Maximum number of *z* samples before the integrator gives up on
            a pixel.  Default ``4097``.
        rtol : float, optional
            Relative tolerance for the adaptive integrator.  Default ``1e-4``.
        atol : float, optional
            Absolute tolerance for the adaptive integrator.  Default ``1e-8``.

        Returns
        -------
        ImageData
            2-D surface-density map with pixel-centre coordinates and axis
            extents.

        See Also
        --------
        gal3d.visualization.los_integr.LOSIntegrator :
            The underlying adaptive integrator.
        """
        from gal3d.visualization.los_integr import LOSIntegrator
        from gal3d.visualization.show import ImageData

        im_value = LOSIntegrator(
            source=self,
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
            rotation_matrix=rotation_matrix,
            z_range=z_range,
            nz_min=nz_min,
            nz_max=nz_max,
            rtol=rtol,
            atol=atol,
        ).run()

        xs = np.linspace(x_range[0], x_range[1], resolution + 1)
        ys = np.linspace(y_range[0], y_range[1], resolution + 1)
        xs = 0.5 * (xs[:-1] + xs[1:])
        ys = 0.5 * (ys[:-1] + ys[1:])
        return ImageData(im_value, xs=xs, ys=ys, xrange=x_range, yrange=y_range)

    def shape_at(
        self,
        r: float | Iterable[float],
        workflow: Optional["FitWorkflowBase"] = None,
        progress: bool = True,
        warm_start: bool = True,
        **kwargs: Any
        ) -> "ModelResult":
        """
        Fit a shape model to the density distribution at one or more radii.

        Parameters
        ----------
        r : float or iterable of float
            Radius or sequence of radii at which to perform the fit.
        workflow : FitWorkflowBase, optional
            The fitting workflow to use.  If ``None`` (default), a default
            workflow is used.
        progress : bool, optional
            Show a ``tqdm`` progress bar when iterating over radii.  Default is ``True``.
        warm_start : bool, optional
            If ``True`` (default), pass the best-fit parameters of the previous
            radius as the initial guess for the next radius when fitting a sequence of radii.
        **kwargs
            Additional arguments for fitting, passed to the workflow.

        Returns
        -------
        ModelResult
            The result of the fitting process.
        """
        from gal3d.model_workflow.fit_workflow import FitWorkflow, FitWorkflowBase

        fit = FitWorkflow.get_plugin("IterateEllipsoidDensity") if workflow is None else workflow
        # check fit is a class or instance of FitWorkflowBase
        fitter = fit() if isinstance(fit, type) else fit

        if isinstance(fitter, FitWorkflowBase):
            return fitter(self, r, warm_start=warm_start, progress=progress, **kwargs)
        else:
            raise TypeError("workflow must be a FitWorkflowBase subclass or instance")

    # ------------------------------------------------------------------
    # Public projection API (subclasses may override)
    # ------------------------------------------------------------------
    def project_2d(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        resolution: int = 200,
        rotation_matrix: np.ndarray | None = None,
        z_range: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> "ImageData":
        """
        Project the 3-D density onto a 2-D surface-density image.

        The base implementation delegates to :meth:`_project_2d_los`.
        Subclasses may override this method with a faster algorithm (e.g.
        SPH rendering); :meth:`_project_2d_los` remains accessible as the
        LOS fallback regardless of any override.

        Parameters
        ----------
        x_range : (float, float)
            Minimum and maximum extent along the *x*-axis of the output
            image, in the same physical units as the positions.
        y_range : (float, float)
            Minimum and maximum extent along the *y*-axis of the output
            image.
        resolution : int, optional
            Number of pixels along each axis.  Default ``200``.
        rotation_matrix : ndarray, shape (3, 3), optional
            Rotation applied to query positions before evaluating the
            density.  ``None`` (default) means no rotation.
        z_range : (float, float), optional
            Integration limits along the line of sight.  If ``None`` a
            symmetric range ``[-max(Δx, Δy), +max(Δx, Δy)]`` is used.

        Returns
        -------
        ImageData
            2-D surface-density map with pixel-centre coordinates and axis
            extents.

        See Also
        --------
        DensitySource.project_2d_los :
            The always-available LOS integration implementation.
        """
        #TODO these parameters may be set in config
        nz_min: int = 33
        nz_max: int = 4097
        rtol: float = 1e-4
        atol: float = 1e-8

        return self.project_2d_los(
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
            rotation_matrix=rotation_matrix,
            z_range=z_range,
            nz_min=nz_min,
            nz_max=nz_max,
            rtol=rtol,
            atol=atol,
        )
