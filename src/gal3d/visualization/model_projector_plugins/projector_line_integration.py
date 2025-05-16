import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate


from gal3d.visualization.model_projector import ModelProjectorBase
from gal3d.util.array_operate import Rotate


class ProjectorLineIntegration(ModelProjectorBase):
    def __init__(self, model, model_cric=None, cache_len=100, **kwargs):

        super().__init__(cache_len=cache_len)
        sel = kwargs.get("sel", None)

        self.model = model
        self.model_cric = model_cric
        try:
            self.model_sel = np.arange(len(self.model['parameter']))
        except KeyError:
            raise KeyError("The model dictionary must contain the key 'parameter'.")

        if (self.model_cric is not None) or (sel is not None):
            if sel is None:
                sel = np.array(self.model.res['fun']) < self.model_cric
            else:
                if self.model_cric is not None:
                    sel = sel & (np.array(self.model.res['fun']) < self.model_cric)

            self.model_sel = self.model_sel[sel]

    def _setup_grid(self, x_range, y_range, nbins):
        """Set up the projection grid and calculate bin centers."""
        deproject_array = np.ones((nbins, nbins), dtype=np.float64)
        cen_indices = np.array(deproject_array.shape) / 2 - 0.5
        indices = np.transpose(np.nonzero(deproject_array))
        pos = np.zeros((nbins * nbins, 2), dtype=np.float64)

        xs = np.linspace(x_range[0], x_range[1], nbins + 1)
        ys = np.linspace(y_range[0], y_range[1], nbins + 1)
        xs = 0.5 * (xs[:-1] + xs[1:])
        ys = 0.5 * (ys[:-1] + ys[1:])

        pos[:, 0] = (indices - cen_indices)[:, 0] * (x_range[1] - x_range[0]) / nbins
        pos[:, 1] = (indices - cen_indices)[:, 1] * (y_range[1] - y_range[0]) / nbins
        
        return indices, pos, xs, ys
    
    def _prepare_sight_lines(self, pos, z_range, rotation):
        """Prepare the sight lines for integration."""
        pos1 = np.zeros((len(pos), 3))
        pos2 = np.zeros((len(pos), 3))

        pos1[:, 0] = pos[:, 0]
        pos1[:, 1] = pos[:, 1]
        pos1[:, 2] = z_range[1]

        pos2[:, 0] = pos[:, 0]
        pos2[:, 1] = pos[:, 1]
        pos2[:, 2] = z_range[0]

        pos1 = Rotate(pos1, rotation)
        pos2 = Rotate(pos2, rotation)
        
        return pos1, pos2
    
    def _calculate_intersections(self, pos1, pos2):
        """Calculate intersections between sight lines and model components."""
        model_sel = self.model_sel
        if len(model_sel) == 0:
            raise ValueError("model_sel is empty. Ensure that the selection criteria result in a non-empty model_sel.")

        project_profile = {}
        
        # Find initial intersections
        alll = self.model[int(model_sel[-1])].quick_line_intersect(pos1=pos1, pos2=pos2)
        ind = np.arange(len(pos1))
        ind_in = ind[(alll[:, 0] > 0.0)]
        ind_total = ind_in.copy()

        # Calculate intersections for each model component
        para = self.model['parameter']
        for i in tqdm(model_sel[::-1], desc="Intersecting"):
            sec = self.model[int(i)].quick_line_intersect(
                pos1=pos1[ind_in], pos2=pos2[ind_in]
            )
            sel = (sec[:, 0] > 0.0)
            tar = ind_in[sel]
            sec = sec[sel]

            for j in range(len(tar)):
                if tar[j] not in project_profile:
                    project_profile[tar[j]] = [[], []]
                project_profile[tar[j]][0].extend([sec[j][0], sec[j][1]])
                project_profile[tar[j]][1].extend([para[i], para[i]])
            ind_in = tar
            
        return project_profile, ind_total
    
    def _integrate_profiles(self, project_profile, ind_total, indices, nbins):
        """Integrate profiles along sight lines to create the final image."""
        deproject_array = np.zeros((nbins, nbins), dtype=np.float64)
        
        for i in tqdm(ind_total, desc="Integrating Profiles"):
           x = np.array(project_profile[i][0])
           y = np.array(project_profile[i][1])
           xsort = np.argsort(x)
           inte = integrate.trapezoid(y[xsort], x[xsort])
           deproject_array[tuple(indices[i])] = inte
           
        return deproject_array
    
    
    def _image(
        self, x_range, y_range, nbins: int = 100, z_range=(-20, 20), rotation=np.eye(3)
    ):
        """
        Generate a 2D projection image by integrating along a specified line of sight.

        Args:
            x_range (tuple): The range of x-coordinates (min, max) for the projection.
            y_range (tuple): The range of y-coordinates (min, max) for the projection.
            nbins (int): The number of bins for the x and y axes (default is 100).
            z_range (tuple): The range of z-coordinates (min, max) for the integration (default is (-20, 20)).
            rotation (numpy.ndarray): A 3x3 rotation matrix to apply to the coordinates (default is identity matrix).

        Returns:
            tuple: A tuple containing:
                - deproject_array.T (numpy.ndarray): The transposed 2D array of integrated values.
                - xs (numpy.ndarray): The x-coordinates of the bin centers.
                - ys (numpy.ndarray): The y-coordinates of the bin centers.
        """
        # Set up projection grid
        indices, pos, xs, ys = self._setup_grid(x_range, y_range, nbins)
        
        # Prepare sight lines
        pos1, pos2 = self._prepare_sight_lines(pos, z_range, rotation)
        
        # Calculate intersections
        project_profile, ind_total = self._calculate_intersections(pos1, pos2)
        
        # Integrate profiles
        deproject_array = self._integrate_profiles(project_profile, ind_total, indices, nbins)
        
        return deproject_array.T, xs, ys
