
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from matplotlib.patches import ConnectionPatch, Rectangle
from numpy.typing import NDArray

from gal3d.point import Particles
from gal3d.util.array_operate import Rotate

from .hist2d import (
    add_colorbar,
    hist_2d,
    render_2d,
    show_contour,
    show_image,
    which_pos_to_rotation,
)
from .model_projector import ModelProjectorBase


def show_data_model(
    axes: list[plt.Axes],
    model: ModelProjectorBase,
    data: Particles,
    which_pos: Sequence[int] = (0, 1),
    rotation_matrix: np.ndarray | None = None,
    x_range: tuple[float, float] = (-15, 15),
    y_range: tuple[float, float] = (-15, 15),
    z_range: tuple[float, float] = (-20, 20),
    nbins: int = 200,
    logscale: bool = True,
    cmap: str = "turbo",
    nlevels: int | tuple[int, int] | None = 20,
    linewidth: float = 0.8,
    color: str = "k",
    linestyle: str = "-",
    render: bool = True
) -> tuple[tuple[AxesImage, AxesImage, AxesImage], tuple[QuadContourSet, QuadContourSet, QuadContourSet]]:
    if rotation_matrix is None:
        rotation_matrix = np.eye(3)
    if render:
        data_image, xs, ys = render_2d(
            data.pos,data.mass,data.hsm,
            which_pos=which_pos,
            rotation_matrix=rotation_matrix,
            x_range=x_range,
            y_range=y_range,
            nbins=nbins,
        )
    else:
        pos = Rotate(data.pos, rotation_matrix.T)
        data_image, xs, ys = hist_2d(
            pos[:, which_pos[0]],
            pos[:, which_pos[1]],
            weights=data.mass,
            x_range=x_range,
            y_range=y_range,
            density=True,
            nbins=nbins,
        )

    if nlevels is None:
        nlevels = int(np.sqrt(data_image.size))
    if isinstance(nlevels, int):
        nlevel1 = nlevels
        nlevel2 = nlevels
        nlevel3 = nlevels
    else:
        nlevel1 = nlevels[0]
        nlevel2 = nlevels[1]
        nlevel3 = nlevels[-1]

    data_im = show_image(
        data_image,
        axesObj=axes[0],
        extent=(*x_range, *y_range),
        logscale=logscale,
        cmap=cmap,
    )

    data_contour = show_contour(
        data_image,
        xs,
        ys,
        withfilter=True,
        sigma=0.9,
        axesObj=axes[0],
        nlevels=nlevel1,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        logscale=logscale,
    )

    rota = which_pos_to_rotation(which_pos)
    rota = Rotate(rotation_matrix, rota.T)
    model_image, xs, ys = model.image(
        x_range=x_range, y_range=y_range, nbins=nbins, z_range=z_range, rotation=rota
    )

    model_im = show_image(
        model_image,
        axesObj=axes[1],
        extent=(*x_range, *y_range),
        logscale=logscale,
        cmap=cmap,
        vmin=data_im.colorizer.vmin,
        vmax=data_im.colorizer.vmax,
    )

    model_contour = show_contour(
        model_image,
        xs,
        ys,
        withfilter=True,
        sigma=0.9,
        axesObj=axes[1],
        vmin=data_im.colorizer.vmin,
        vmax=data_im.colorizer.vmax,
        nlevels=nlevel2,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        logscale=logscale,
    )

    residual_im = show_image(
        np.abs(data_image - model_image),
        axesObj=axes[2],
        extent=(*x_range, *y_range),
        logscale=logscale,
        cmap=cmap,
        vmin=data_im.colorizer.vmin,
        vmax=data_im.colorizer.vmax,
    )

    residual_contour = show_contour(
        np.abs(data_image - model_image),
        xs,
        ys,
        withfilter=True,
        sigma=0.9,
        axesObj=axes[2],
        nlevels=nlevel3,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        logscale=logscale,
        vmin=data_im.colorizer.vmin,
        vmax=data_im.colorizer.vmax,
    )

    return (data_im, model_im, residual_im), (
        data_contour,
        model_contour,
        residual_contour,
    )


def set_tick_params(*args, **kwargs):
    for i in args:
        i.tick_params(**kwargs)


def set_xlabel(*args, **kwargs):
    for i in args:
        i.set_xlabel(**kwargs)


def plot_zoom(
    main_axs: Axes,
    zoom_axs: Axes,
    xy: tuple[float, float] = (0, 0),
    length: float = 10,
    height: float = 10,
    linestyle: str = ":",
    linewidth: float = 1,
    color: str = "red",
    zoom_loc: str = "right",
    arrowstyle: str = "-",
    arrowcolor: str = "red",
    arrowwidth: float = 1,
) -> tuple[Rectangle, ConnectionPatch, ConnectionPatch]:
    """
    Create a zoomed view of a specific region in the main axes.

    This function highlights a rectangular region on the main axes and connects it
    with lines to a zoomed view in another axes object.

    Parameters
    ----------
    main_axs : matplotlib.axes.Axes
        The main axes containing the original plot.
    zoom_axs : matplotlib.axes.Axes
        The axes where the zoomed region will be shown.
    xy : tuple of float, default=(0, 0)
        The bottom-left coordinates of the zoom rectangle.
    length : float, default=10
        Width of the zoom rectangle.
    height : float, default=10
        Height of the zoom rectangle.
    linestyle : str, default=':'
        Line style for the zoom rectangle.
    linewidth : float, default=1
        Line width for the zoom rectangle.
    color : str, default='red'
        Color of the zoom rectangle.
    zoom_loc : str, default='right'
        Location of the zoom window relative to the main plot.
        Options: 'right', 'left', 'top', 'bottom'
    arrowstyle : str, default='-'
        Style of the connection lines.
    arrowcolor : str, default='red'
        Color of the connection lines.
    arrowwidth : float, default=1
        Width of the connection lines.

    Returns
    -------
    Tuple[Rectangle, ConnectionPatch, ConnectionPatch]
        The rectangle patch and the two connection lines.

    Raises
    ------
    ValueError
        If an invalid zoom_loc value is provided.
    """

    square = Rectangle(
        xy,
        length,
        height,
        linestyle=linestyle,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )
    main_axs.add_patch(square)

    # Determine connection points based on zoom location
    if zoom_loc == "right":
        line1 = ((xy[0] + length, xy[1]), (xy[0], xy[1]))
        line2 = ((xy[0] + length, xy[1] + height), (xy[0], xy[1] + height))
    elif zoom_loc == "left":
        line1 = ((xy[0], xy[1]), (xy[0] + length, xy[1]))
        line2 = ((xy[0], xy[1] + height), (xy[0] + length, xy[1] + height))
    elif zoom_loc == "top":
        line1 = ((xy[0] + length, xy[1] + height), (xy[0] + length, xy[1]))
        line2 = ((xy[0], xy[1] + height), (xy[0], xy[1]))
    elif zoom_loc == "bottom":
        line1 = ((xy[0] + length, xy[1]), (xy[0] + length, xy[1] + height))
        line2 = ((xy[0], xy[1]), (xy[0], xy[1] + height))
    else:
        raise ValueError(
            f"Invalid zoom_loc '{zoom_loc}'. Must be one of: 'right', 'left', 'top', 'bottom'"
        )

    # Create connection patches
    con1 = ConnectionPatch(
        xyA=line1[0],
        coordsA=main_axs.transData,
        xyB=line1[1],
        coordsB=zoom_axs.transData,
        arrowstyle=arrowstyle,
        color=arrowcolor,
        linewidth=arrowwidth,
    )
    main_axs.add_artist(con1)

    con2 = ConnectionPatch(
        xyA=line2[0],
        coordsA=main_axs.transData,
        xyB=line2[1],
        coordsB=zoom_axs.transData,
        arrowstyle=arrowstyle,
        color=arrowcolor,
        linewidth=arrowwidth,
    )
    main_axs.add_artist(con2)

    return square, con1, con2


def show_image_model_residual(
    data: Particles,
    model: ModelProjectorBase,
    large_box_x_range: tuple[float, float] = (-15, 15),
    large_box_y_range: tuple[float, float] = (-15, 15),
    zoom_x_range: tuple[float, float] = (-5, 5),
    zoom_y_range: tuple[float, float] = (-5, 5),
    depth_z_range: tuple[float, float] = (-20, 20),
    nbins_large: int = 200,
    nbins_zoom: int = 100,
    nlevels_large: int | tuple[int, int] = 13,
    nlevels_zoom: int | tuple[int, int] = 17,
    which_pos_all: list[tuple[int, int]] | None = None,
    rotation_matrix: NDArray[np.float64] | None = None,
    cmap: str = "turbo",
    title_text: list[str] | None= None,
    titlesize: float = 25,
    ylabel_all: list[str] | None = None,
    xlabel_all: list[str] | None = None,
    labelsize: float = 13,
    savefile: str | None = None,
) -> plt.Figure | list[plt.Figure]:
    """
    Create a comprehensive visualization comparing observed data, model, and residuals.

    This function generates a detailed visualization that shows the observed data,
    model projections, and their residuals in both large-scale and zoomed-in views.
    It's useful for evaluating the quality of model fits to observational data.

    Parameters
    ----------
    data : Particles
        The particle data to visualize.
    model : ModelProjectorBase
        The model projector to generate model images.
    large_box_x_range : tuple of float, default=(-15, 15)
        X-axis range for the large box view in kpc.
    large_box_y_range : tuple of float, default=(-15, 15)
        Y-axis range for the large box view in kpc.
    zoom_x_range : tuple of float, default=(-5, 5)
        X-axis range for the zoomed view in kpc.
    zoom_y_range : tuple of float, default=(-5, 5)
        Y-axis range for the zoomed view in kpc.
    depth_z_range : tuple of float, default=(-20, 20)
        Depth range in z-direction for projection in kpc.
    nbins_large : int, default=200
        Number of bins for the large box view.
    nbins_zoom : int, default=100
        Number of bins for the zoomed view.
    nlevels_large : int, default=13
        Number of contour levels for the large box view.
    nlevels_zoom : int, default=17
        Number of contour levels for the zoomed view.
    which_pos_all : list of tuples, default=[(0,1), (0,2)]
        Axis pairs to use for projections (e.g., [(0,1)] for xy projection).
    rotation_matrix : ndarray, default=np.eye(3)
        3x3 rotation matrix to apply before projection.
    cmap : str, default='turbo'
        Colormap to use for the images.
    title_text : list of str, default=['Face', 'Edge']
        Titles for each projection view.
    titlesize : float, default=25
        Font size for titles.
    ylabel_all : list of str, default=["Data", "Model", "Residual"]
        Labels for the y-axis rows (data, model, residual).
    xlabel_all : list of str, default=["R [kpc]", "R [kpc]", "R [kpc]", "R [kpc]"]
        Labels for the x-axes.
    labelsize : float, default=13
        Font size for axis labels.
    savefile : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    fig or list of figs
        The generated figure(s).

    Notes
    -----
    This function creates a complex multi-panel figure showing:
    - Original data projections
    - Model projections
    - Residuals between data and model
    - Both large-scale and zoomed-in views of each

    Each projection can be shown from different viewpoints (e.g., face-on, edge-on).
    """
    if xlabel_all is None:
        xlabel_all = ["R [kpc]", "R [kpc]", "R [kpc]", "R [kpc]"]
    if ylabel_all is None:
        ylabel_all = ["Data", "Model", "Residual"]
    if title_text is None:
        title_text = ["Face", "Edge"]
    if which_pos_all is None:
        which_pos_all = [(0, 1), (0, 2)]
    fig = plt.figure(dpi=300, figsize=(17, 13))
    gs = fig.add_gridspec(3, 4, hspace=0, wspace=0)
    axes = [[plt.subplot(gs[i, j]) for j in range(4)] for i in range(3)]

    allpanels = []
    for i in range(2):
        h1 = show_data_model(
            [axes[0][2 * i], axes[1][2 * i], axes[2][2 * i]],
            which_pos=which_pos_all[i],
            rotation_matrix=rotation_matrix,
            cmap=cmap,
            model=model,
            data=data,
            x_range=large_box_x_range,
            y_range=large_box_y_range,
            z_range=depth_z_range,
            nbins=nbins_large,
            nlevels=nlevels_large,
        )
        allpanels.append(h1)
        h2 = show_data_model(
            [axes[0][2 * i + 1], axes[1][2 * i + 1], axes[2][2 * i + 1]],
            which_pos=which_pos_all[i],
            rotation_matrix=rotation_matrix,
            cmap=cmap,
            model=model,
            data=data,
            x_range=zoom_x_range,
            y_range=zoom_y_range,
            z_range=depth_z_range,
            nbins=nbins_zoom,
            nlevels=nlevels_zoom,
        )
        allpanels.append(h2)

    set_tick_params(
        axes[0][0],
        axes[0][1],
        axes[0][2],
        axes[0][3],
        axes[1][0],
        axes[1][1],
        axes[1][2],
        axes[1][3],
        axes[2][0],
        axes[2][1],
        axes[2][2],
        axes[2][3],
        axis="y",
        which="both",
        direction="out",
        left=True,
        right=True,
        labelright=False,
        labelleft=False,
    )

    set_tick_params(
        axes[0][0],
        axes[0][1],
        axes[0][2],
        axes[0][3],
        axes[1][0],
        axes[1][1],
        axes[1][2],
        axes[1][3],
        axis="x",
        which="both",
        direction="out",
        bottom=True,
        top=True,
        labelbottom=False,
        labeltop=False,
    )

    set_tick_params(
        axes[2][0],
        axes[2][1],
        axes[2][2],
        axes[2][3],
        axis="x",
        which="both",
        direction="out",
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=False,
    )

    for i in range(4):
        position = axes[0][i].get_position()
        cb_ax = fig.add_axes(
            (position.x0, position.y1, position.x1 - position.x0, (1 - position.y1) / 6)
        )
        cb_ax.set_visible(False)
        cb = add_colorbar(
            allpanels[i][0][0],
            ax=cb_ax,
            loc="top",
            size="100%",
            pad=-(1 - position.y1) / 12,
        )
        cb.set_label(r"$\Sigma_{*}\ [M_{\odot}/\mathrm{kpc^2}]$", fontsize=10)

    for i in range(3):
        plot_zoom(
            axes[i][0],
            axes[i][1],
            xy=(zoom_x_range[0], zoom_y_range[0]),
            length=(zoom_x_range[1] - zoom_x_range[0]),
            height=(zoom_y_range[1] - zoom_y_range[0]),
            linewidth=2,
            arrowwidth=1.5,
            arrowstyle="->",
        )
        plot_zoom(
            axes[i][2],
            axes[i][3],
            xy=(zoom_x_range[0], zoom_y_range[0]),
            length=(zoom_x_range[1] - zoom_x_range[0]),
            height=(zoom_y_range[1] - zoom_y_range[0]),
            linewidth=2,
            arrowwidth=1.5,
            arrowstyle="->",
        )

    for i in range(2):
        position1 = axes[0][2 * i].get_position()
        position2 = axes[0][2 * i + 1].get_position()
        fig.text(
            (position1.x0 + position2.x1) / 2,
            0.95,
            title_text[i],
            fontsize=titlesize,
            va="center",
            ha="center",
        )

    for i in range(3):
        axes[i][0].set_ylabel(
            ylabel_all[i],
            fontsize=labelsize,
        )

    for i in range(4):
        axes[-1][i].set_xlabel(
            xlabel_all[i],
            fontsize=labelsize,
        )

    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    return fig
