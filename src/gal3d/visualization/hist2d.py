"""
2D image generation helpers for particle data and model.

Includes:
- hist_2d: produce density-like 2D histograms as ImageData
- render_2d: high-quality SPH rendering wrapper returning ImageData
- which_pos_to_rotation: helper to construct a rotation from axis selection
"""

from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np

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
