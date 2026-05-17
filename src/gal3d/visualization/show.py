"""
- show_image / show_contour: display helpers with consistent normalization reuse
- add_colorbar: attach colorbar to any ScalarMappable
"""

from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colorizer import ColorizingArtist
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter


@dataclass
class ImageData:
    """
    Container for a projected image and its grid information.

    Attributes
    ----------
    value : np.ndarray
        Image array of shape (ny, nx).
    xs : np.ndarray
        X bin centers of shape (nx,).
    ys : np.ndarray
        Y bin centers of shape (ny,).
    xrange : tuple[float, float]
        (xmin, xmax)
    yrange : tuple[float, float]
        (ymin, ymax)
    """

    value: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    xrange: tuple[float, float]
    yrange: tuple[float, float]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])

    @property
    def nx(self) -> int:
        return self.xs.size

    @property
    def ny(self) -> int:
        return self.ys.size

    @property
    def pixel_area(self) -> float:
        return (self.xrange[1] - self.xrange[0]) * (self.yrange[1] - self.yrange[0]) / (self.nx * self.ny)

    @property
    def total_quantity(self) -> float:
        return np.sum(self.value) * self.pixel_area


def show_image(
    imageData: np.ndarray | ImageData,
    extent: tuple[float, float, float, float] | None = None,
    axesObj: plt.Axes | None = None,
    scale: str = "linear",
    logscale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    clip: bool = True,
    noErase: bool = False,
) -> AxesImage:
    """
    Display a 2D image with optional scaling and color normalization.

    Parameters
    ----------
    imageData : 2D ndarray
        The image data to be displayed.
    extent : tuple, optional
        The bounding box in data coordinates as (left, right, bottom, top). Default is None.
    axesObj : matplotlib.axes.Axes, optional
        The axes object to plot the image on. If None, the current axes are used. Default is None.
    scale : str, optional
        Scaling to apply to the color normalization. Options are "linear", "log", "symlog", etc. Default is "linear".
    logscale : boolean, optional
        If True, use a log-scaled colorbar and log-spaced contours. Default is True.
    vmin : float, optional
        Minimum data value for color normalization. Default is the minimum value in imageData.
    vmax : float, optional
        Maximum data value for color normalization. Default is the maximum value in imageData.
    cmap : str, optional
        Colormap to use for the image. Default is "jet".
    noErase : bool, optional
        If True, do not clear the current figure before plotting. Default is False.

    Returns
    -------
    axesImg : matplotlib.image.AxesImage
        The image object created by imshow.
    """
    if isinstance(imageData, ImageData):
        im_arr = np.asarray(imageData.value).copy()
        extent = extent if extent is not None else imageData.extent
    else:
        im_arr = np.asarray(imageData).copy()

    if logscale:
        scale = "log"
    if scale == "linear":
        cont_color = colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
    elif scale == "log":
        vmin = vmin or np.percentile(im_arr[im_arr > 0], 1)
        vmax = vmax or np.max(im_arr[im_arr > 0])
        cont_color = colors.LogNorm(vmin=vmin, vmax=vmax, clip=clip)
    elif scale == "symlog":
        cont_color = colors.SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax, clip=clip)
    else:
        raise ValueError(f"Unsupported scale: {scale}")

    if axesObj is None:
        if not noErase:
            plt.clf()
        axesImg = plt.imshow(
            im_arr, interpolation="nearest", origin="lower", aspect="equal", extent=extent, norm=cont_color, cmap=cmap
        )
    else:
        axesImg = axesObj.imshow(
            im_arr, interpolation="nearest", origin="lower", aspect="equal", extent=extent, norm=cont_color, cmap=cmap
        )
    return axesImg


def show_contour(
    imageData: np.ndarray | ImageData,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    axesObj: plt.Axes | None = None,
    withfilter: bool = False,
    sigma: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    nlevels: int = 10,
    levels: np.ndarray | None = None,
    logscale: bool = True,
    noErase: bool = False,
    color: str = "k",
    linewidth: float = 0.5,
    linestyle: str = "-",
) -> QuadContourSet:
    """
    Contour-plot an image. Can auto-generate log- or linear-spaced levels.

    Parameters
    ----------
    imageData : 2D ndarray
        2D Numpy array
    xs : array-like
        x-coordinates of bin centres
    ys : array-like
        y-coordinates of bin centres
    axesObj : instance of matplotlib.axes.Axes, optional
        Axes instance to receive the plotting commands
    withfilter : boolean, optional
        If True, use gaussian_filter. Default is False.
    sigma: float, optional
        Used in gaussian_filter
    vmin : float, optional
        Minimum value to use for the color scale. Default is arr.min().
    vmax : float, optional
        Maximum value to use for the color scale. Default is arr.max().
    nlevels : int, optional
        Number of levels to use for contours. Default is 10.
    levels : sequence (tuple, list, or Numpy array) of float or None, optional
        contour intensity levels to be plotted (if log=True, then these should be
        log10 of the original values)
    logscale : boolean, optional
        If True, use a log-scaled colorbar and log-spaced contours. Default is True.
    noErase : bool, optional
        If True, draws the contours into an existing plot window without erasing
        existing content. Default is False.
    color : str, optional
        color for contour lines. Default is `k`.
    linewidth : float, optional
        lines width for contour lines. Default is 0.5.
    linestyle : str, optional
        line styles for contour lines. Default is `-`.

    Returns
    -------
    axesCont : QuadContourSet
    """
    if isinstance(imageData, ImageData):
        im_arr = np.asarray(imageData.value).copy()
        xs = xs if xs is not None else imageData.xs
        ys = ys if ys is not None else imageData.ys
    else:
        im_arr = np.asarray(imageData).copy()
        if xs is None or ys is None:
            raise ValueError("xs and ys must be provided when imageData is not an ImageData instance")

    if logscale:
        vmin = vmin or np.min(im_arr[im_arr > 0])
        vmax = vmax or np.max(im_arr[im_arr > 0])
        if levels is None:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels)
        # cont_color = colors.LogNorm(vmin = vmin, vmax = vmax)
    else:
        vmin = vmin or np.min(im_arr)
        vmax = vmax or np.max(im_arr)
        if levels is None:
            levels = np.linspace(vmin, vmax, nlevels)
        # cont_color = colors.Normalize(vmin = vmin, vmax = vmax)
    levels = np.atleast_1d(levels)
    if withfilter:
        sigma = sigma or 1
        im_arr = gaussian_filter(im_arr, sigma=sigma)
    if axesObj is None:
        if not noErase:
            plt.clf()
        axesCont = plt.contour(xs, ys, im_arr, levels, colors=color, linewidths=linewidth, linestyles=linestyle)

    else:  # user supplied a matplotlib.axes.Axes object to receive the plotting commands
        axesCont = axesObj.contour(xs, ys, im_arr, levels, colors=color, linewidths=linewidth, linestyles=linestyle)
    return axesCont


def add_colorbar(
    mappable: ScalarMappable | ColorizingArtist,
    ax: plt.Axes | None = None,
    loc: str = "right",
    size: str = "5%",
    pad: float = 0.05,
    label_pad: float = 2,
    tick_label_size: float = 10,
) -> Colorbar:
    """
    Add a colorbar to a 'mappable' (ScalarMappable/ColorizingArtist) with flexible placement.

    Parameters
    ----------
    mappable : instance of object implementing "mappable" interface
        E.g., instance of Image, ContourSet, etc. -- basically any Artist subclass that
        inherits from the ScalarMappable mixin
        https://matplotlib.org/api/cm_api.html
    loc : str, optional
        location for colorbar -- one of "right", "left", "top", "bottom"
    size : str, optional
        relative size for colorbar as fraction of main plot, as a percentage (e.g. "2%")
    pad : float, optional
        padding between colorbar and main plot
    label_pad : str, optional
        padding between colorbar and its tick labels
    tick_label_size : float, optional
        font size for tick labels

    Returns
    -------
    cbar : instance of matplotlib.colorbar.Colorbar
        The generated colorbar

    Example
    -------
    >>> img = plt.imshow(data)
    >>> add_colorbar(img, loc="right")
    """
    if loc in ["top", "bottom"]:
        orient = "horizontal"
        if loc == "top":
            tickPos = "top"
        else:
            tickPos = "bottom"
    else:
        orient = "vertical"
        if loc == "left":
            tickPos = "left"
        else:
            tickPos = "right"
    axe = ax or cast("plt.Axes | None", getattr(mappable, "axes", None))
    if axe is None:
        raise ValueError("No axes found for colorbar. Pass ax=... or provide a mappable attached to an Axes.")
    fig = axe.figure
    divider = make_axes_locatable(axe)
    cbar_axes = divider.append_axes(loc, size=size, pad=pad)
    cbar = fig.colorbar(mappable, cax=cbar_axes, orientation=orient)

    # fiddle with tick label locations
    if loc in ["left", "right"]:
        cbar_axis = cbar_axes.yaxis
    else:
        cbar_axis = cbar_axes.xaxis
    cbar_axis.set_ticks_position(tickPos)
    cbar_axis.set_label_position(tickPos)
    cbar.ax.tick_params(labelsize=tick_label_size, pad=label_pad)
    return cbar
