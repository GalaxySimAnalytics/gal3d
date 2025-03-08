

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import gaussian_filter

# see https://github.com/perwin/barprofiles_paper/blob/main/plotutils.py, many useful func to plot


def hist_2d(x,y,weights=None,parameters=None,density=True,
            gridsize=(100,100),nbins=None,
            x_logscale=False,y_logscale=False,x_range=None,y_range=None,**kwargs):
    '''
    plt.imshow(data,origin='lower')
    '''
    if nbins is not None:
        gridsize = (nbins, nbins)
    
    if y_range is not None:
        if len(y_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        y_range = [np.log10(np.min(y)), np.log10(np.max(y))] if y_logscale else [np.min(y), np.max(y)]

            
    if x_range is not None:
        if len(x_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        x_range = [np.log10(np.min(x)), np.log10(np.max(x))] if x_logscale else [np.min(x), np.max(x)]


    
    x = np.log10(x) if x_logscale else x
    y = np.log10(y) if y_logscale else y

    
    ind = np.where((x > x_range[0]) & (x < x_range[1]) &
                   (y > y_range[0]) & (y < y_range[1]))

    x = x[ind[0]]
    y = y[ind[0]]
    
    if weights is not None:
            weights = weights[ind[0]]

    if parameters is not None:
        parameters = parameters[ind[0]]
        if weights is None:
            weights = np.ones_like(parameters)
    
    def _histogram_generator(weights):
        return np.histogram2d(y, x, weights=weights, bins=gridsize, range=[y_range, x_range])
    
    if parameters is not None:
        hist, ys, xs = _histogram_generator(weights * parameters)
        hist_norm, _, _ = _histogram_generator(weights)
        valid = hist_norm > 0
        hist[valid] /= hist_norm[valid]
    else:
        hist, ys, xs = _histogram_generator(weights)
    
    if density:
        area = (np.diff(xs).reshape(-1,1)*np.diff(ys))
        hist = hist/area
        
    xs = .5 * (xs[:-1] + xs[1:])
    ys = .5 * (ys[:-1] + ys[1:])
    return hist,xs,ys


def show_image( imageData,extent=None,axesObj=None,  logscale=True, vmin=None,vmax=None, cmap="jet",noErase=False):
    
    if logscale:
        vmin = np.min(imageData[imageData > 0]) if vmin is None else vmin
        vmax = np.max(imageData[imageData > 0]) if vmax is None else vmax
        cont_color = colors.LogNorm(vmin = vmin, vmax = vmax)
    else:
        vmin = np.min(imageData) if vmin is None else vmin
        vmax = np.max(imageData) if vmax is None else vmax
        cont_color = colors.Normalize(vmin = vmin, vmax = vmax)
        
    imageData[imageData < vmin] = vmin
    imageData[imageData > vmax] = vmax
    
    if axesObj is None:
        if noErase is False:
            plt.clf()
        axesImg = plt.imshow(imageData,interpolation='nearest',origin='lower',aspect='equal',
                             extent=extent,norm=cont_color,cmap=cmap)
    else:
        axesImg = axesObj.imshow(imageData,interpolation='nearest',origin='lower',aspect='equal',
                             extent=extent,norm=cont_color,cmap=cmap)
        
    return axesImg


def show_contour( imageData,xs,ys,axesObj=None,withfilter=False,sigma=None, vmin=None,vmax=None,
                 nlevels=10, levels=None,logscale=True,noErase=False,
                color='k', linewidth=0.5, linestyle='-'):
    """
    Function which contour-plots an image.
    
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
        used in  gaussian_filter
        
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

    noErase = set this equal to True to draw the contours into an existing plot
        window without erase things first (only used if axesObj is None)
        
    color = color for the contours
    
    linewidth = float
    
    linestyle = one of 'solid', 'dashed', 'dashdot', 'dotted'
    
    """

    if logscale:
        vmin = np.min(imageData[imageData > 0]) if vmin is None else vmin
        vmax = np.max(imageData[imageData > 0]) if vmax is None else vmax
        levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels) if levels is None else levels
        #cont_color = colors.LogNorm(vmin = vmin, vmax = vmax)
    else:
        vmin = np.min(imageData) if vmin is None else vmin
        vmax = np.max(imageData) if vmax is None else vmax
        levels = np.linspace(vmin, vmax, nlevels) if levels is None else levels
        #cont_color = colors.Normalize(vmin = vmin, vmax = vmax)
    levels = np.atleast_1d(levels)
    if withfilter:
        sigma = 1 if sigma is None else sigma
        imageData = gaussian_filter(imageData,sigma=sigma)
    if axesObj is None:
        if noErase is False:
            plt.clf()
            axesCont = plt.contour(xs, ys, imageData, levels,colors=color, linewidths=linewidth,
                        linestyles=linestyle)
            
        else:
            axesCont = plt.contour(xs, ys, imageData, levels,colors=color, linewidths=linewidth,
                        linestyles=linestyle)

    else:    # user supplied a matplotlib.axes.Axes object to receive the plotting commands
        axesCont = axesObj.contour(xs, ys, imageData, levels,colors=color, linewidths=linewidth,
                        linestyles=linestyle)
    return axesCont




def add_colorbar( mappable, ax=None,loc="right", size="5%", pad=0.05, label_pad=2,
                    tick_label_size=10 ):
    """
    Function which adds a colorbar to a "mappable" object (e.g., the result of
    calling plt.imshow).

    Example:
        img = plt.imshow(somedata, ...)
        add_colorbar(img, ...)

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

    tick_lable_size : float, optional
        font size for tick labels

    Returns
    -------
    cbar : instance of matplotlib.colorbar.Colorbar
        The generated colorbar
    """
    if loc in ["top", "bottom"]:
        orient = "horizontal"
        if loc == "top":
            tickPos = 'top'
        else:
            tickPos = 'bottom'
    else:
        orient = "vertical"
        if loc == "left":
            tickPos = 'left'
        else:
            tickPos = 'right'
    ax = mappable.axes if ax is None else ax
    fig = ax.figure
    divider = make_axes_locatable(ax)
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