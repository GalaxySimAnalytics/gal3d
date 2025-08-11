from typing import Sequence, Any, Tuple

import numpy as np
from tqdm import tqdm   # type: ignore

from ...field.spherical_field.spherical_vector import SphVector
from ...util.array_operate import Rotate
from ..hist2d import hist_2d
from ..model_projector import ModelProjectorBase


class ProjectorSphGrid(ModelProjectorBase):
    def __init__(
        self, model, N_ray: int = 6000, num_p: int = 200, cache_len=100, **kwargs
    ):

        inner_model = kwargs.get("inner_model", model[0])
        outer_model = kwargs.get("outer_model", model[-1])

        super().__init__(cache_len=cache_len)
        ray_model = SphVector(N_ray)
        total_outer_r = 1.1 * (1 - outer_model.quick_call_dist(pos=ray_model.pos))
        total_inner_r = 0.9 * (1 - inner_model.quick_call_dist(pos=ray_model.pos))

        points_r = np.geomspace(total_inner_r, total_outer_r, num_p).T

        inner_r = np.array(
            [
                np.convolve(
                    points_r[i],
                    [0.5, 0.5],
                    mode='same',
                )
                for i in range(len(points_r))
            ]
        )
        inner_r[:, 0] = 0
        outer_r = np.roll(inner_r, -1, axis=1)
        outer_r[:, -1] = (points_r[:, -1] * 3 - points_r[:, -2]) / 2

        volumn = np.einsum('ij,i->ij', (outer_r**3 - inner_r**3), ray_model.area / 3)
        volumn = volumn.flatten()

        posall = np.einsum('ij,ik->ijk', points_r, ray_model.pos).reshape(-1, 3)

        self.pos = posall
        self.volumn = volumn

        density = np.zeros(len(self.volumn))
        indices = np.arange(len(self.volumn))

        for i in tqdm(range(len(model))):
            sel = model[i](self.pos[indices]) <= 1
            density[indices[sel]] = model['parameter'][i]

            indices = indices[~sel]

        weight = self.volumn * density

        self.density = density
        self.weight = weight

    def _image(
        self, 
        x_range: Sequence[float], 
        y_range: Sequence[float],
        nbins: int = 100, 
        z_range: Sequence[float] = (-20, 20),
        rotation: np.ndarray = np.eye(3),
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        new_pos = Rotate(self.pos, rotation)
        sel = (new_pos[:, 2] > z_range[0]) & (new_pos[:, 2] < z_range[1])
        model_image, xs, ys = hist_2d(
            new_pos[:, 0][sel],
            new_pos[:, 1][sel],
            weights=self.weight[sel],
            x_range=x_range,
            y_range=y_range,
            density=True,
            nbins=nbins,
        )

        return model_image, xs, ys
