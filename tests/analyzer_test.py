import os

import numpy as np
import pytest

from gal3d.analyzer import Gal3DAnalyzer, ModelResult
from gal3d.characterization import Characterizer
from gal3d.visualization import ModelProjector, show_image_model_residual


@pytest.fixture
def data():
    ID = 46
    path = os.path.join(os.path.dirname(__file__), "test_data", f"TNG50_ID{ID}_star_particles_float32.npy")
    return np.load(path)

@pytest.fixture
def gal(data) -> Gal3DAnalyzer:
    return Gal3DAnalyzer.analyze(data[:,:3],data[:,3])

@pytest.fixture
def res_ellipsoid_s(gal) -> ModelResult:
    return gal.fit(num_step=50)


def test_characterization(res_ellipsoid_s):
    bar = Characterizer.get_plugin('Bar')
    data = {i: res_ellipsoid_s[i] for i in ['a','eps_ab','eps_bc','ang1','ang2','ang3']}

    data['pa'] = data['ang1']
    res = bar(data).measure()
    assert isinstance(res, dict)
    assert 'flag' in res
    assert 'eps_max' in res
    assert 'R_max' in res
    assert 'R_bar' in res
    
def test_result_generate(res_ellipsoid_s):
    
    res_ellipsoid_s[10].generate_edge2D(n_angle_bins=10,n_r_bins=10)
    res_ellipsoid_s[15].generate_edge3D(n_phi_bins=10,n_theta_bins=5)
    res_ellipsoid_s[15].generate_slice2D(n_bins=8)
    res_ellipsoid_s[20].generate_points(random_np=64)

def test_show_image(gal, res_ellipsoid_s):
    ellipsoid_s_model = ModelProjector.get_plugin('ProjectorLineIntegration')(res_ellipsoid_s)

    box_lh_max = np.max(res_ellipsoid_s['a']) * 1.1
    zoom_lh_max = box_lh_max / 4
    fig = show_image_model_residual(gal.particle, ellipsoid_s_model,
                              large_box_x_range=(-box_lh_max, box_lh_max),
                              large_box_y_range=(-box_lh_max, box_lh_max),
                              zoom_x_range=(-zoom_lh_max, zoom_lh_max),
                              zoom_y_range=(-zoom_lh_max, zoom_lh_max),
                              depth_z_range=(-box_lh_max, box_lh_max),
                              nbins_large=60,
                              nbins_zoom=30,
                              nlevels_large=0,
                              nlevels_zoom=22)

