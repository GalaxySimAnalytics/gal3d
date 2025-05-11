from .util import *
import logging


__all__ = ['Grid']

logger = logging.getLogger(__name__)


class Grid:
    """
    A class to represent a 3D grid for fitting galaxy morphologies.

    This class is used to create and manage a 3D grid structure that can be used to fit the morphology of galaxies.
    The grid is constructed based on the positions and parameters of the input data, and can be customized using
    various methods and parameters.

    Parameters
    ----------
    pos : numpy.ndarray
        A 2D array of shape (N, 3) representing the positions of the data points in 3D space.
    parameter : numpy.ndarray
        A 1D array of shape (N,) representing the parameters associated with each data point.
    maxdepth : int, optional
        The maximum depth of the grid. The grid will be recursively subdivided until this depth is reached.
        Default is 14.
    splitpart : int, optional
        The number of parts to split the grid into along each axis at each level of subdivision.
        Default is 128.
    **kwargs : dict, optional
        Additional keyword arguments to customize the grid boundaries and other properties.
        Possible keys include 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax' to set the lower and upper bounds
        of the grid in each dimension.

    Attributes
    ----------
    base_pos : numpy.ndarray
        The positions of the data points that fall within the grid boundaries.
    base_pa : numpy.ndarray
        The parameters associated with the data points that fall within the grid boundaries.
    maxdepth : int
        The maximum depth of the grid.
    splitpart : int
        The number of parts to split the grid into along each axis at each level of subdivision.
    bound_lower : numpy.ndarray
        The lower bounds of the grid in each dimension.
    bound_upper : numpy.ndarray
        The upper bounds of the grid in each dimension.
    grid_pos_l : numpy.ndarray
        The lower bounds of each grid cell.
    grid_pos_u : numpy.ndarray
        The upper bounds of each grid cell.
    grid_depth : numpy.ndarray
        The depth of each grid cell.
    grid_volumn : numpy.ndarray
        The volume of each grid cell.
    grid_sumpa : numpy.ndarray
        The sum of the parameters within each grid cell.
    grid_denpa : numpy.ndarray
        The density of the parameters within each grid cell.
    grid_pos : numpy.ndarray
        The center position of each grid cell.
    grid_nums : numpy.ndarray
        The number of data points within each grid cell.
    """

    def __init__(
        self, pos, parameter, maxdepth: int = 14, splitpart: int = 128, **kwargs
    ):
        """
        Initialize the Grid object.

        Parameters
        ----------
        pos : numpy.ndarray
            A 2D array of shape (N, 3) representing the positions of the data points in 3D space.
        parameter : numpy.ndarray
            A 1D array of shape (N,) representing the parameters associated with each data point.
        maxdepth : int, optional
            The maximum depth of the grid. Default is 14.
        splitpart : int, optional
            The number of parts to split the grid into along each axis at each level of subdivision.
            Default is 128.
        **kwargs : dict, optional
            Additional keyword arguments to customize the grid boundaries and other properties.
        """
        self.set_bound(pos, **kwargs)

        sele = np.sum(pos <= self.bound_upper, axis=1) + np.sum(
            pos >= self.bound_lower, axis=1
        )

        self.base_pos = pos[sele == 6]
        self.base_pa = parameter[sele == 6]

        self.maxdepth = maxdepth
        self.splitpart = splitpart

        self.make_grid(method=kwargs.get('method', 'make_grid_by_num'))

    def set_bound(self, pos, **kwargs):
        """
        Set the lower and upper bounds of the grid.

        Parameters
        ----------
        pos : numpy.ndarray
            A 2D array of shape (N, 3) representing the positions of the data points in 3D space.
        **kwargs : dict, optional
            Additional keyword arguments to customize the grid boundaries.
            Possible keys include 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'.
        """
        bound_lower = np.min(pos, axis=0, keepdims=True)
        bound_upper = np.max(pos, axis=0, keepdims=True)
        posmin = ['xmin', 'ymin', 'zmin']
        posmax = ['xmax', 'ymax', 'zmax']
        for i, j in enumerate(posmin):
            if j in kwargs:
                bound_lower[0][i] = float(kwargs[j])  #  min

        for i, j in enumerate(posmax):
            if j in kwargs:
                bound_upper[0][i] = float(kwargs[j])  #  max
        self.bound_lower = bound_lower
        self.bound_upper = bound_upper

    def make_grid(self, method: str = 'make_grid_by_num') -> None:
        """
        Create the grid using the specified method.

        Parameters
        ----------
        method : str, optional
            The method to use for creating the grid. Possible values are 'make_grid_by_num' and 'make_grid_by_diff'.
            Default is 'make_grid_by_num'.

        Raises
        ------
        ValueError
            If the specified method is not supported.
        """
        splid_method = {
            'make_grid_by_num': make_grid_by_num,
            'make_grid_by_diff': make_grid_by_diff,
        }

        if method not in splid_method:
            logger.error("Unsupported grid creation method: %s", method)
            raise ValueError(f"Unsupported method: {method}")

        try:
            lower_pos, upper_pos, Depth, Nums, Indice = splid_method[method](
                self.base_pos,
                self.maxdepth,
                self.splitpart,
                self.bound_lower,
                self.bound_upper,
            )
            volumn, masses, density = cal_volumn_density(
                lower_pos, upper_pos, self.base_pa, Indice
            )
            grid_pos = (lower_pos + upper_pos) / 2

            self.base_indice = Indice
            self.grid_pos_l = lower_pos[Nums > 0]
            self.grid_pos_u = upper_pos[Nums > 0]
            self.grid_depth = Depth[Nums > 0]
            self.grid_volumn = volumn[Nums > 0]
            self.grid_sumpa = masses[Nums > 0]
            self.grid_denpa = density[Nums > 0]
            self.grid_pos = grid_pos[Nums > 0]
            logger.info("Removed %d void grids", len(Nums[Nums == 0]))
            self.grid_nums = Nums[Nums > 0]
        except Exception as e:
            logger.error("Failed to create grid: %s", e, exc_info=True)
            raise

    def provide_griddata(self) -> dict:
        """
        Provide the grid data as a dictionary.

        Returns
        -------
        dict
            A dictionary containing the grid positions, densities, and volumes.
        """
        retdict = {}
        retdict['pos'] = self.grid_pos
        retdict['density'] = self.grid_denpa
        retdict['volumn'] = self.grid_volumn

        return retdict
