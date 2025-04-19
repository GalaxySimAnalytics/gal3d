import numpy as np
import matplotlib.pyplot as plt
from gal3d.visualization.model_projector import ModelProjector
from gal3d.visualization.data_model_residual import show_image_model_residual


def test_visualize(particle_data,gal_data_for_visualize):
    res = gal_data_for_visualize.fit(r=np.geomspace(3*gal_data_for_visualize.field.iso_pro_r[0],min(gal_data_for_visualize.field.iso_pro_r[-1],15),5))
    model = ModelProjector.get_plugin("ProjectorLineIntegration")(res,0.015)
    model.image(x_range=(-15,15),y_range=(-15,15),nbins=200,)
    model.image_xz(x_range=(-15,15),y_range=(-15,15),nbins=200,)
    model.image_yz(x_range=(-15,15),y_range=(-15,15),nbins=200,)