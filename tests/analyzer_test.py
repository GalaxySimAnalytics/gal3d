import os
import numpy as np
import pytest

from gal3d.analyzer import Gal3DAnalyzer, ModelResult, EmptyModelResult
from gal3d.characterization import Characterizer
from gal3d.visualization import ModelProjector, show_image_model_residual


# --- Fixtures ---
@pytest.fixture
def data():
    """Fixture to load test particle data."""
    ID = 46
    path = os.path.join(os.path.dirname(__file__), "test_data", f"TNG50_ID{ID}_star_particles_float32.npy")
    return np.load(path)

@pytest.fixture
def gal(data) -> Gal3DAnalyzer:
    """Fixture to create a Gal3DAnalyzer instance from test data."""
    return Gal3DAnalyzer.analyze(data[:,:3], data[:,3])

@pytest.fixture
def res_ellipsoid_s(gal) -> ModelResult:
    """Fixture to fit the model and return the ModelResult."""
    return gal.fit(num_step=50)


# --- ModelResult attribute and behavior tests ---
def test_model_result_attributes(res_ellipsoid_s: ModelResult):
    """Test ModelResult attributes and basic access."""
    assert isinstance(res_ellipsoid_s, ModelResult)
    # Check main attributes
    for attr in ["params", "fun", "start_fun", "start_params", "algorithm"]:
        getattr(res_ellipsoid_s, attr)
    for attr in ["success", "message", "status", "n_fun_evals", "n_jac_evals"]:
        getattr(res_ellipsoid_s, attr)
    for attr in ["n_hess_evals", "n_iterations", "jac", "hess", "hess_inv"]:
        getattr(res_ellipsoid_s, attr)
    for attr in ["max_constraint_violation", "history", "algorithm_output", "multistart_info", "x"]:
        getattr(res_ellipsoid_s, attr)
    for attr in ["x0", "nfev", "nit", "njev", "nhev"]:
        getattr(res_ellipsoid_s, attr)
    # Method and magic
    assert callable(res_ellipsoid_s.get)
    assert isinstance(dir(res_ellipsoid_s), list)
    assert isinstance(repr(res_ellipsoid_s), str)
    assert isinstance(len(res_ellipsoid_s), int)

def test_model_result_get(res_ellipsoid_s: ModelResult):
    """Test ModelResult get method."""
    assert res_ellipsoid_s.get("parameter") is not None

def test_model_result_slice_and_index(res_ellipsoid_s: ModelResult):
    """Test ModelResult slicing and indexing."""
    assert isinstance(res_ellipsoid_s[0:10], ModelResult)
    with pytest.raises(IndexError):
        _ = res_ellipsoid_s[200]
    with pytest.raises(KeyError):
        _ = res_ellipsoid_s[0.1]

def test_model_result_invalid_attribute(res_ellipsoid_s: ModelResult):
    """Test ModelResult invalid attribute access."""
    with pytest.raises(AttributeError):
        _ = res_ellipsoid_s.non_existent_attribute
        _ = res_ellipsoid_s.__a__


def test_model_result_addition(res_ellipsoid_s: ModelResult):
    """Test addition of ModelResult objects."""
    other = res_ellipsoid_s[0:5]
    combined = res_ellipsoid_s + other
    assert isinstance(combined, ModelResult)
    assert len(combined) != len(res_ellipsoid_s)

def test_model_result_call(res_ellipsoid_s: ModelResult):
    """Test __call__ method of ModelResult."""
    func = res_ellipsoid_s[5]
    result = func([1.0, 2.0, 3.0])
    assert result is not None


# --- EmptyModelResult tests ---
def test_empty_model_result_behavior(res_ellipsoid_s: ModelResult):
    """Test EmptyModelResult behavior and interaction with ModelResult."""
    result = EmptyModelResult()
    assert isinstance(result, ModelResult)
    h = res_ellipsoid_s + result
    assert h is res_ellipsoid_s
    assert bool(result) is False
    assert bool(res_ellipsoid_s) is True
    with pytest.raises(ValueError):
        _ = result()
    with pytest.raises(ValueError):
        _ = result[0]
    assert isinstance(repr(result), str)


# --- Characterization plugin tests ---
def test_characterizer_bar_plugin(res_ellipsoid_s: ModelResult):
    """Test Characterizer Bar plugin measurement output."""
    bar = Characterizer.get_plugin('Bar')
    res = bar(res_ellipsoid_s).measure()
    assert isinstance(res, dict)
    for key in ['flag', 'eps_max', 'R_max', 'R_bar']:
        assert key in res


# --- ModelResult generation methods ---
def test_model_result_generate_methods(res_ellipsoid_s: ModelResult):
    """Test ModelResult generation methods for edge, slice, and points."""
    res_ellipsoid_s[10].generate_edge2D(n_angle_bins=10, n_r_bins=10)
    res_ellipsoid_s[15].generate_edge3D(n_phi_bins=10, n_theta_bins=5)
    res_ellipsoid_s[15].generate_slice2D(n_bins=8)
    res_ellipsoid_s[20].generate_points(random_np=32)

# --- Visualization and field tests ---
def test_gal3d_field_and_visualization(gal: Gal3DAnalyzer, res_ellipsoid_s: ModelResult):
    """Test Gal3DAnalyzer field methods and visualization."""
    assert isinstance(repr(gal), str)
    assert isinstance(repr(gal.field), str)
    gal.field.pos_ray_n(2)
    gal.field.r_ray_n(4)
    gal.field.mass_ray_n(4)
    gal.field.parameter_ray_n(4)
    gal.field.gradient_ray_n(10)
    gal.field.generate(2.1)
    gal.field.generate_by_f(1e7)
    ellipsoid_s_model = ModelProjector.get_plugin('ProjectorLineIntegration')(res_ellipsoid_s)
    box_lh_max = np.max(res_ellipsoid_s['a']) * 1.1
    zoom_lh_max = box_lh_max / 4
    fig = show_image_model_residual(
        gal.particle, ellipsoid_s_model,
        large_box_x_range=(-box_lh_max, box_lh_max),
        large_box_y_range=(-box_lh_max, box_lh_max),
        zoom_x_range=(-zoom_lh_max, zoom_lh_max),
        zoom_y_range=(-zoom_lh_max, zoom_lh_max),
        depth_z_range=(-box_lh_max, box_lh_max),
        nbins_large=60,
        nbins_zoom=30,
        nlevels_large=0,
        nlevels_zoom=22
    )
    fig = show_image_model_residual(
        gal.particle, ellipsoid_s_model,
        large_box_x_range=(-box_lh_max, box_lh_max),
        large_box_y_range=(-box_lh_max, box_lh_max),
        zoom_x_range=(-zoom_lh_max, zoom_lh_max),
        zoom_y_range=(-zoom_lh_max, zoom_lh_max),
        depth_z_range=(-box_lh_max, box_lh_max),
        nbins_large=60,
        nbins_zoom=30,
        nlevels_large=0,
        nlevels_zoom=22,
        render=False
    )

def test_gal3d_analyzer_fit_single_radius(gal: Gal3DAnalyzer):
    """Test fitting with a single radius value."""
    res = gal._fit(5.0)
    assert isinstance(res, ModelResult)
    res = gal._fit(0.01)
    assert isinstance(res, EmptyModelResult)

