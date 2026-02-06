"""
2D image generation helpers for particle data and model visualization.

Includes:
- hist_2d: produce density-like 2D histograms as ImageData
- render_2d: high-quality SPH rendering wrapper returning ImageData
- show_image / show_contour: display helpers with consistent normalization reuse
- add_colorbar: attach colorbar to any ScalarMappable
- which_pos_to_rotation: helper to construct a rotation from axis selection
"""
# This code is inspired by or adapted from the plotting utilities in the following repository:
# https://github.com/perwin/barprofiles_paper/blob/main/plotutils.py

from collections.abc import Sequence
from typing import Any, Literal, cast, overload

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

from gal3d.configuration import config
from gal3d.util.array_operate import Rotate

from .model_projector import ImageData
from .render_wrapper import PyRenderImage, PyRenderImageFloat, get_kernel, get_render_image


def which_pos_to_rotation(which_pos: Sequence[int]) -> np.ndarray:
    """
    Build a rotation matrix that maps an axis order (e.g., (0,1) for xy) to view orientation.

    Parameters
    ----------
    which_pos : Sequence[int]
        Two axes indices from {0,1,2}. The third axis is inferred.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix (float64).
    """
    order = list(which_pos)
    for i in [0, 1, 2]:
        if i not in order:
            order.append(i)
            break
    h1 = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        h1[i] = np.eye(3)[int(order[i])]
    return h1.T

def hist_2d(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    parameters: np.ndarray | None = None,
    density: bool = True,
    gridsize: tuple[int,int]= (100, 100),
    nbins: int | None = None,
    x_logscale: bool =False,
    y_logscale: bool =False,
    x_range: tuple[float,float] | None =None,
    y_range: tuple[float,float] | None =None,
    **kwargs: Any,
) -> ImageData:
    """
    Generate a 2D histogram as ImageData.

    If `parameters` is given, returns a weighted average in each bin.

    Parameters
    ----------
    x, y : array-like
        Coordinates.
    weights : array-like, optional
        Weights per point.
    parameters : array-like, optional
        Values to average in bins (weighted by `weights`).
    density : bool, default True
        Normalize by bin area.
    gridsize : (ny, nx), default (100, 100)
        Bin counts per axis. If `nbins` is given, overrides both to (nbins, nbins).
    nbins : int, optional
        Set both axes to the same number of bins.
    x_logscale, y_logscale : bool, default False
        If True, log10-transform the axis before binning and compute ranges accordingly.
    x_range, y_range : (min, max), optional
        If None, use min/max of data (or log10 thereof).

    Returns
    -------
    ImageData
        The binned image with coordinates and extents.
    """
    if nbins is not None:
        gridsize = (nbins, nbins)

    if y_range is not None:
        if len(y_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        y_range = (
            (np.log10(np.min(y)), np.log10(np.max(y)))
            if y_logscale
            else (np.min(y), np.max(y))
        )

    if x_range is not None:
        if len(x_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        x_range = (
            (np.log10(np.min(x)), np.log10(np.max(x)))
            if x_logscale
            else (np.min(x), np.max(x))
        )

    x = np.log10(x) if x_logscale else x
    y = np.log10(y) if y_logscale else y

    ind = np.where(
        (x > x_range[0]) & (x < x_range[1]) & (y > y_range[0]) & (y < y_range[1])
    )

    x = x[ind[0]]
    y = y[ind[0]]

    if weights is not None:
        weights = weights[ind[0]]

    if parameters is not None:
        parameters = parameters[ind[0]]
        if weights is None:
            weights = np.ones_like(parameters)

    def _histogram_generator(weights):
        return np.histogram2d(
            y, x, weights=weights, bins=(gridsize[1],gridsize[0]), range=(y_range, x_range)
        )

    if parameters is not None:
        hist, ys, xs = _histogram_generator(weights * parameters)
        hist_norm, _, _ = _histogram_generator(weights)
        valid = hist_norm > 0
        hist[valid] /= hist_norm[valid]
    else:
        hist, ys, xs = _histogram_generator(weights)

    if density:
        area = np.diff(xs).reshape(-1, 1) * np.diff(ys)
        hist = hist / area

    xs = 0.5 * (xs[:-1] + xs[1:])
    ys = 0.5 * (ys[:-1] + ys[1:])

    return ImageData(hist, xs, ys, x_range, y_range)

@overload
def render_2d(
    pos: np.ndarray,
    mass: np.ndarray,
    hsm: np.ndarray,
    which_pos: Sequence[int] = ...,
    rotation_matrix: np.ndarray | None = ...,
    x_range: Sequence[float] = ...,
    y_range: Sequence[float] = ...,
    nbins: int | None = ...,
    subsample: int | None = ...,
    ret_image: Literal[True] = True
) -> ImageData: ...
@overload
def render_2d(
    pos: np.ndarray,
    mass: np.ndarray,
    hsm: np.ndarray,
    which_pos: Sequence[int] = ...,
    rotation_matrix: np.ndarray | None = ...,
    x_range: Sequence[float] = ...,
    y_range: Sequence[float] = ...,
    nbins: int | None = ...,
    subsample: int | None = ...,
    ret_image: Literal[False] = False
) -> PyRenderImage | PyRenderImageFloat: ...
def render_2d(pos: np.ndarray, mass: np.ndarray, hsm: np.ndarray, which_pos: Sequence[int] = (0, 1),
            rotation_matrix: np.ndarray | None = None,
            x_range: Sequence[float] = (-15, 15),
            y_range: Sequence[float] = (-15, 15),
            nbins: int | None = None,
            subsample: int | None = None,
            ret_image: bool = True
            ) -> (ImageData | PyRenderImage | PyRenderImageFloat):
    """
    SPH-based 2D rendering (fast and smooth). Returns ImageData by default.

    Parameters
    ----------
    pos : (N, 3) array
        Particle positions.
    mass : (N,) array
        Particle masses.
    hsm : (N,) array
        Particle smoothing lengths.
    which_pos : sequence of 2 ints, default (0, 1)
        Indices of pos to use for x and y axes (e.g., (0, 2) for xz).
    rotation_matrix : (3, 3) array, optional
        If given, rotate positions by this matrix before rendering. Should be a proper rotation matrix.
    x_range, y_range : (min, max), default (-15, 15)
        Extent of the rendered image in physical units.
    nbins : int, optional
        Number of pixels along each axis. If None, uses config.sph_render.resolution.
    subsample : int, optional
        Subsampling factor for rendering. If None, uses config.sph_render.subsample.
    ret_image : bool, default True
        If True, return the rendered image as ImageData. If False, return the renderer object itself.

    Returns
    -------
    ImageData or PyRenderImage
        The rendered image as ImageData if ret_image is True, otherwise the renderer object.
    """
    if nbins is None:
        nbins = config.sph_render.resolution
    if subsample is None:
        subsample = config.sph_render.subsample

    render = get_render_image(x_range[0], x_range[1], y_range[0], y_range[1],
                              nbins, nbins, get_kernel(), subsample, subsample)
    # Ensure rotation_matrix dtype matches particle.pos
    if rotation_matrix is not None:
        rot = rotation_matrix.T.astype(pos.dtype)
        pos = Rotate(pos, rot)

    render.add_particle(pos[:,which_pos[0]],pos[:, which_pos[1]],mass,hsm)

    if ret_image:
        return render.get_image()
    else:
        return render


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
            im_arr,
            interpolation="nearest",
            origin="lower",
            aspect="equal",
            extent=extent,
            norm=cont_color,
            cmap=cmap,
        )
    else:
        axesImg = axesObj.imshow(
            im_arr,
            interpolation="nearest",
            origin="lower",
            aspect="equal",
            extent=extent,
            norm=cont_color,
            cmap=cmap,
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
        axesCont = plt.contour(
            xs,
            ys,
            im_arr,
            levels,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
        )

    else:  # user supplied a matplotlib.axes.Axes object to receive the plotting commands
        axesCont = axesObj.contour(
            xs,
            ys,
            im_arr,
            levels,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
        )
    return axesCont


def add_colorbar(
    mappable: ScalarMappable | ColorizingArtist,
    ax: plt.Axes | None = None,
    loc: str = "right",
    size: str ="5%",
    pad: float =0.05,
    label_pad: float =2,
    tick_label_size: float = 10
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
