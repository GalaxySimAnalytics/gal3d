"""
Workflow for model fitting.
"""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeAlias, Union

from tqdm import tqdm

from gal3d.optimization.result import EmptyModelResult, ModelResult
from gal3d.plugin import PluginBase, PluginManager
from gal3d.util.errors import FitDataError

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer
    from gal3d.density import DensitySource
    from gal3d.point import Particles

logger = logging.getLogger("gal3d.fit_workflow")

FitInput: TypeAlias = Union["Gal3DAnalyzer", "Particles", "DensitySource"]


class FitWorkflowBase(PluginBase):
    """
    Base class for all fitting workflows.

    Subclasses must implement:
    - _fit_single(self, obj, r: float, **kwargs) -> ModelResult:
        Fit the model at a single radius ``r``.
    - condition(analyzer: "Gal3DAnalyzer") -> bool:
        Returns True if this workflow should be selected for the given analyzer.

    Registration:
    Subclasses are automatically registered as plugins via __init_subclass__.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register the subclass as a fitting workflow plugin.
        """
        super().__init_subclass__(**kwargs)
        FitWorkflow.register(cls)

    @staticmethod
    def condition(obj: FitInput) -> bool:
        """
        Condition for selecting the fitting workflow.

        Parameters
        ----------
        obj : FitInput
            The analyzer, particle, or density source instance.

        Returns
        -------
        bool
            True if this workflow should be used, False otherwise.
        """
        return False

    def _fit_single(self, obj: FitInput, r: float, **kwargs: Any) -> ModelResult:
        """
        Fit the model at a single radius.

        Parameters
        ----------
        obj : FitInput
            The analyzer, particle, or density source instance.
        r : float
            The radius at which to fit the model.
        **kwargs
            Additional arguments for fitting.

        Returns
        -------
        ModelResult
            The result of the fitting process.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        raise NotImplementedError

    def __call__(
        self,
        obj: FitInput,
        r: float | Iterable[float],
        *,
        progress: bool = True,
        warm_start: bool = True,
        **kwargs: Any,
    ) -> ModelResult:
        """
        Fit the model at one or multiple radii.

        Handles scalar and sequence inputs uniformly.  For a sequence of
        radii the previous result is optionally used as the initial
        parameter guess for the next radius (warm start).

        Parameters
        ----------
        obj : FitInput
            The analyzer, particle, or density source instance.
        r : float or iterable of float
            Radius or sequence of radii at which to perform the fit.
        progress : bool, optional
            Show a ``tqdm`` progress bar when iterating over radii.
            Default is ``True``.
        warm_start : bool, optional
            If ``True`` (default), pass the best-fit parameters of the
            previous radius as ``init_parameters`` to the next call of
            :meth:`_fit_single`.  Has no effect for scalar ``r``.
        **kwargs
            Extra keyword arguments forwarded to :meth:`_fit_single`.

        Returns
        -------
        ModelResult
            * Scalar ``r``  → single :class:`ModelResult` (or
              :class:`EmptyModelResult` on :exc:`FitDataError`).
            * Sequence ``r`` → aggregated :class:`ModelResult` summed
              over all successful radii, or :class:`EmptyModelResult`
              if every radius failed.
        """
        if not isinstance(r, Iterable):
            # ---- scalar path ----
            try:
                return self._fit_single(obj, float(r), **kwargs)
            except FitDataError as exc:
                logger.error("Error fitting radius %.4g: %s", r, exc)
                return EmptyModelResult()

        # ---- sequence path ----
        results: list[ModelResult] = []
        pending_skips: dict[str, list[float]] = {}

        def _log_warning_below_pbar(msg: str, pbar: tqdm | None) -> None:
            if pbar is not None and not pbar.disable:
                with pbar.external_write_mode():
                    logger.warning(msg)
            else:
                logger.warning(msg)

        def _flush_pending(pbar: tqdm | None) -> None:
            for etype, radii in pending_skips.items():
                if not radii:
                    continue
                if len(radii) == 1:
                    msg = f"{etype}: skipped radius {radii[0]:.4g}"
                else:
                    msg = f"{etype}: skipped {len(radii)} radii from {radii[0]:.4g} to {radii[-1]:.4g}"
                _log_warning_below_pbar(msg, pbar)
            pending_skips.clear()

        pbar = tqdm(r, desc="Fitting radii", disable=not progress)

        try:
            for i in pbar:
                radius = float(i)
                kw = dict(kwargs)

                if warm_start and results:
                    last = results[-1]
                    kw.setdefault("init_parameters", {key: last[key][0] for key in last.keys()})

                try:
                    result = self._fit_single(obj, radius, **kw)
                    if pending_skips:
                        _flush_pending(pbar)
                    results.append(result)
                except FitDataError as exc:
                    etype = type(exc).__name__
                    pending_skips.setdefault(etype, []).append(radius)

        except KeyboardInterrupt:
            _log_warning_below_pbar("Interrupted by user; returning partial results.", pbar)

        # ----
        if pending_skips:
            _flush_pending(pbar)

        if results:
            return sum(results[1:], results[0])
        return EmptyModelResult()


class FitWorkflow(PluginManager[FitWorkflowBase]):
    """
    Factory class for accessing registered fitting workflows.

    This manager loads all FitWorkflowBase plugins from the specified module,
    and selects the appropriate workflow based on the analyzer's condition.

    Attributes
    ----------
    _plugins : dict[str, type[FitWorkflowBase]]
        Registered workflow plugins.
    _plugin_module : str
        Module path to search for workflow plugins.
    _base_class : type
        The base class for all workflow plugins.
    """

    _plugins = {}
    _plugin_module = "gal3d.model_workflow.fit_workflow_plugins"
    _base_class = FitWorkflowBase

    @classmethod
    def get_workflow(cls, obj: FitInput | str) -> FitWorkflowBase:
        """
        Select and instantiate the appropriate fitting workflow for the input object.

        Parameters
        ----------
        obj : FitInput | str
            The analyzer, particle, or density source instance, or workflow name.

        Returns
        -------
        FitWorkflowBase
            The selected workflow instance.

        Raises
        ------
        TypeError
            If no valid workflow is found for the object.
        """
        cls.available_plugins()

        if isinstance(obj, str):
            return cls.get_plugin(obj)()

        for wf_cls in cls._plugins.values():
            if wf_cls.condition(obj):
                return wf_cls()
        raise TypeError("No valid workflow found for input object.")
