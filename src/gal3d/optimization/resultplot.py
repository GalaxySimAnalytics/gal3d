from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.container import ErrorbarContainer
    from matplotlib.lines import Line2D

    from .result import ModelResult


SeriesLike = str | np.ndarray | list[float] | tuple[float, ...] | float | int
ErrorLike = SeriesLike | bool | None


class _ResultPlotBase:
    _default_style: dict[str, Any] = {}

    def __init__(self, result: ModelResult) -> None:
        self.result = result

    def _resolve(self, value: SeriesLike | None) -> np.ndarray | None:
        """Resolve a SeriesLike value to a numpy array."""
        if value is None:
            return None
        if isinstance(value, str):
            return np.asarray(self.result[value])
        return np.asarray(value)

    def _latex_labels(self) -> dict[str, str]:
        """Attempt to retrieve LaTeX labels for the result's columns."""
        try:
            labels = self.result.structure.PNlatex()
            if isinstance(labels, dict):
                return labels
        except Exception:
            pass
        return {}

    def _default_label(self, value: SeriesLike, label: str | None | bool) -> str | None:
        """Determine the default label for a plot based on the input value and label parameter."""
        if isinstance(label, str) or label is None:
            return label
        if not label:
            return None
        if isinstance(value, str):
            latex = self._latex_labels().get(value)
            if latex:
                return f"${latex}$"
            return value
        return None

    def _normalize_error(self, err: ErrorLike, ref: SeriesLike) -> np.ndarray | None:
        """Normalize error input for errorbar plotting."""
        if isinstance(err, bool):
            if not err or not isinstance(ref, str):
                return None
            try:
                return self._resolve(f"{ref}_err")
            except KeyError:
                return None
        return self._resolve(err)

    def _merge_style(self, **kwargs: Any) -> dict[str, Any]:
        """Merge the default style with additional keyword arguments for plotting."""
        style = self._default_style.copy()
        style.update(kwargs)
        return style

    def _get_ax(self, ax: Axes | None) -> Axes:
        """Get the matplotlib Axes to plot on. If ax is None, returns the current Axes."""
        if ax is not None:
            return ax
        import matplotlib.pyplot as plt

        return plt.gca()


class ResultErrorbar(_ResultPlotBase):
    def __call__(
        self,
        x: SeriesLike,
        y: SeriesLike,
        ax: Axes | None = None,
        *,
        yerr: ErrorLike = True,
        xerr: ErrorLike = False,
        fmt: str = "o",
        label: str | None | bool = True,
        c: Any | None = None,
        ecolor: Any | None = None,
        elinewidth: float | None = 0.9,
        capsize: float | None = 2.0,
        barsabove: bool = False,
        lolims: np.ndarray | bool = False,
        uplims: np.ndarray | bool = False,
        xlolims: np.ndarray | bool = False,
        xuplims: np.ndarray | bool = False,
        errorevery: int | tuple[int, int] = 1,
        capthick: float | None = 0.9,
        markersize: float = 2.0,
        linewidth: float = 1.0,
        alpha: float = 0.78,
        data: Any = None,
        **kwargs: Any,
    ) -> ErrorbarContainer:
        """Plot error bars.

        Parameters
        ----------
        x : SeriesLike
            The x values to plot. Can be a column name in the result, a numpy array,
        y : SeriesLike
            The y values to plot. Can be a column name in the result, a numpy array,
        ax : Axes | None, optional
            The matplotlib Axes to plot on. If None, uses the current Axes. Default is None.
        yerr : ErrorLike, optional
            The y error values. Can be a column name in the result, a numpy array,
            a float, or a boolean. If True, attempts to use a column named "{y}_err" in the result.
            If False or not found, no y error bars are plotted. Default is True.
        xerr : ErrorLike, optional
            The x error values. Can be a column name in the result, a numpy array,
            a float, or a boolean. If True, attempts to use a column named "{x}_err
            in the result. If False or not found, no x error bars are plotted. Default is False.
        fmt : str, optional
            The format string for the data points. Default is "o" (circle markers).
        label : str | None | bool, optional
            The label for the data series. If True, uses the column name as the label.
            If False, no label is used. If a string is provided, it is used as the label. Default is True.
        c : Any | None, optional
            The color of the data points and error bars. Can be any matplotlib color specification.
            Default is None (uses default color cycle).
        ecolor : Any | None, optional
            The color of the error bars. Can be any matplotlib color specification.
            If None, uses the same color as the data points. Default is None.
        elinewidth : float | None, optional
            The line width of the error bars. If None, uses the default line width. Default is 0.9.
        capsize : float | None, optional
            The size of the error bar caps. If None, uses the default cap size. Default is 2.0.
        barsabove : bool, optional
            If True, draws the error bars above the data points.
            Default is False (draws error bars below the data points).
        lolims : np.ndarray | bool, optional
            If True, indicates that the y error bars are lower limits. Can also be an array of
            booleans indicating which points are lower limits. Default is False.
        uplims : np.ndarray | bool, optional
            If True, indicates that the y error bars are upper limits. Can also be an array of
            booleans indicating which points are upper limits. Default is False.
        xlolims : np.ndarray | bool, optional
            If True, indicates that the x error bars are lower limits. Can also be an array of
            booleans indicating which points are lower limits. Default is False.
        xuplims : np.ndarray | bool, optional
            If True, indicates that the x error bars are upper limits. Can also be an array of
            booleans indicating which points are upper limits. Default is False.
        errorevery : int | tuple[int, int], optional
            Controls the frequency of error bars. If an integer n, plots every nth error bar.
            If a tuple (n, m), plots every nth error bar starting from the mth. Default is 1 (plots all error bars).
        capthick : float | None, optional
            The thickness of the error bar caps. If None, uses the same thickness as elinewidth. Default is 0.9.
        markersize : float, optional
            The size of the data point markers. Default is 2.0.
        linewidth : float, optional
            The line width of the data points. Default is 1.0.
        alpha : float, optional
            The transparency of the data points and error bars. Default is 0.78.
        data : Any, optional
            An optional data object to pass to the plotting function. Default is None.
        **kwargs : Any
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        ErrorbarContainer
            The container object returned by the matplotlib errorbar function, which includes the plotted data points and error bars.
        """
        ax = self._get_ax(ax)
        artist = ax.errorbar(
            self._resolve(x),  # type: ignore [arg-type]
            self._resolve(y),  # type: ignore [arg-type]
            yerr=self._normalize_error(yerr, y),
            xerr=self._normalize_error(xerr, x),
            fmt=fmt,
            label=self._default_label(y, label),
            c=c,
            ecolor=ecolor,
            elinewidth=elinewidth,
            capsize=capsize,
            barsabove=barsabove,
            lolims=lolims,
            uplims=uplims,
            xlolims=xlolims,
            xuplims=xuplims,
            errorevery=errorevery,
            capthick=capthick,
            markersize=markersize,
            linewidth=linewidth,
            alpha=alpha,
            data=data,
            **self._merge_style(**kwargs),
        )
        return artist


class ResultScatter(_ResultPlotBase):
    def __call__(
        self,
        x: SeriesLike,
        y: SeriesLike,
        *,
        ax: Axes | None = None,
        label: str | None | bool = True,
        s: float | np.ndarray = 16,
        c: Any | None = None,
        marker: str = "o",
        cmap: Any | None = None,
        norm: Any | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 0.82,
        linewidths: float = 0.0,
        edgecolors: Any = "none",
        data: Any = None,
        **kwargs: Any,
    ) -> PathCollection:
        """Plot a scatter plot.

        Parameters
        ----------
        x : SeriesLike
            The x values to plot. Can be a column name in the result, a numpy array, or a list of floats.
        y : SeriesLike
            The y values to plot. Can be a column name in the result, a numpy array, or a list of floats.
        ax : Axes | None, optional
            The matplotlib Axes to plot on. If None, uses the current Axes. Default is None.
        label : str | None | bool, optional
            The label for the data series. If True, uses the column name as the label.
            If False, no label is used. If a string is provided, it is used as the label. Default is True.
        s : float | np.ndarray, optional
            The size of the markers. Can be a single float for uniform size or an array of floats for varying sizes. Default is 16.
        c : Any | None, optional
            The color of the markers. Can be a single color format string, a sequence of colors,
            or a sequence of numbers to be mapped to colors using the cmap and norm parameters.
            Default is None (uses default color cycle).
        marker : str, optional
            The marker style. Can be any valid matplotlib marker style. Default is "o" (circle).
        cmap : Any | None, optional
            The colormap to use if c is a sequence of numbers. Default is None.
        norm : Any | None, optional
            The normalization to use if c is a sequence of numbers. Default is None.
        vmin : float | None, optional
            The minimum data value that corresponds to the lower limit of the colormap.
            Used if c is a sequence of numbers. Default is None.
        vmax : float | None, optional
            The maximum data value that corresponds to the upper limit of the colormap.
            Used if c is a sequence of numbers. Default is None.
        alpha : float, optional
            The transparency of the markers. Default is 0.82.
        linewidths : float, optional
            The width of the marker edges. Default is 0.0 (no edge).
        edgecolors : Any, optional
            The color of the marker edges. Can be a single color format string or a sequence of colors.
            Default is "none" (no edge).
        data : Any, optional
            An optional data object to pass to the plotting function. Default is None.
        **kwargs : Any
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        PathCollection
            The collection object returned by the matplotlib scatter function, which includes the plotted markers.

        """
        ax = self._get_ax(ax)
        artist = ax.scatter(
            self._resolve(x),  # type: ignore [arg-type]
            self._resolve(y),  # type: ignore [arg-type]
            label=self._default_label(y, label),
            s=s,
            c=c,
            marker=marker,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            linewidths=linewidths,
            edgecolors=edgecolors,
            data=data,
            **self._merge_style(**kwargs),
        )
        return artist


class ResultPlot(_ResultPlotBase):
    _default_style = {"solid_capstyle": "round", "solid_joinstyle": "round"}

    def __call__(
        self,
        x: SeriesLike,
        y: SeriesLike,
        *,
        ax: Axes | None = None,
        label: str | None | bool = True,
        color: Any | None = None,
        linestyle: str = "-",
        linewidth: float = 1.6,
        marker: str | None = None,
        markersize: float = 3.0,
        drawstyle: str | None = None,
        alpha: float = 0.92,
        markerfacecolor: Any | None = None,
        markeredgecolor: Any | None = None,
        fillstyle: str | None = None,
        data: Any = None,
        **kwargs: Any,
    ) -> list[Line2D]:
        """Plot a line plot.

        Parameters
        ----------
        x : SeriesLike
            The x values to plot. Can be a column name in the result, a numpy array
        y : SeriesLike
            The y values to plot. Can be a column name in the result, a numpy array or a list of floats.
        ax : Axes | None, optional
            The matplotlib Axes to plot on. If None, uses the current Axes. Default is None.
        label : str | None | bool, optional
            The label for the data series. If True, uses the column name as the label.
            If False, no label is used.
            If a string is provided, it is used as the label. Default is True.
        color : Any | None, optional
            The color of the line. Can be any matplotlib color specification.
            Default is None (uses default color cycle).
        linestyle : str, optional
            The style of the line. Can be any valid matplotlib linestyle.
            Default is "-" (solid line).
        linewidth : float, optional
            The width of the line. Default is 1.6.
        marker : str | None, optional
            The marker style for the data points. Can be any valid matplotlib marker style.
            If None, no markers are plotted. Default is None.
        markersize : float, optional
            The size of the markers. Default is 3.0.
        drawstyle : str | None, optional
            The drawing style for the line. Can be any valid matplotlib drawstyle.
            Default is None (uses default draw style).
        alpha : float, optional
            The transparency of the line. Default is 0.92.
        markerfacecolor : Any | None, optional
            The color of the marker faces. Can be any matplotlib color specification.
            If None, uses the same color as the line. Default is None.
        markeredgecolor : Any | None, optional
            The color of the marker edges. Can be any matplotlib color specification.
            If None, uses the same color as the line. Default is None.
        fillstyle : str | None, optional
            The fill style for the markers. Can be any valid matplotlib fillstyle.
            Default is None (uses default fill style).
        data : Any, optional
            An optional data object to pass to the plotting function. Default is None.
        **kwargs : Any
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        list[Line2D]
            The list of Line2D objects returned by the matplotlib plot function, which includes the plotted line and markers.
        """
        ax = self._get_ax(ax)
        artist = ax.plot(
            self._resolve(x),  # type: ignore [arg-type]
            self._resolve(y),  # type: ignore [arg-type]
            label=self._default_label(y, label),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            drawstyle=drawstyle,
            alpha=alpha,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            fillstyle=fillstyle,
            data=data,
            **self._merge_style(**kwargs),
        )
        return artist
