

import matplotlib.pyplot as plt
import numpy as np

import pytest


from gal3d.analyzer import Gal3DAnalyzer
from gal3d.optimization.result import ModelResult,EmptyModelResult
from gal3d.visualization.model_projector import ModelProjector
from gal3d.visualization.data_model_residual import show_image_model_residual
from gal3d.characterization.characterizer import Characterizer
from gal3d.characterization.characterizer_plugins import Bar
from gal3d.characterization.characterizer_plugins import Disk
from gal3d.optimization.model_io import ModelIO
from gal3d.optimization.model_io_plugins import HDF5ModelIO



@pytest.fixture
def gal(tng_galaxy_data) -> Gal3DAnalyzer:
    """Fixture to create a Gal3DAnalyzer instance from test data."""
    pos, mass = tng_galaxy_data
    return Gal3DAnalyzer.analyze(pos, mass,recenter=False)



@pytest.fixture
def res_ellipsoid_s(gal) -> ModelResult:
    """Fixture to fit the model and return the ModelResult."""
    return gal.fit(30)


class TestModelResultBehavior:
    def test_result_attributes(self, res_ellipsoid_s: ModelResult):
        """Test ModelResult attributes and basic access."""
        assert isinstance(res_ellipsoid_s, ModelResult)
        # Check main attributes
        for attr in dir(ModelResult):
            if not attr.startswith("_"):
                getattr(res_ellipsoid_s, attr)
        # Method and magic
        assert callable(res_ellipsoid_s.get)
        assert isinstance(dir(res_ellipsoid_s), list)
        assert isinstance(repr(res_ellipsoid_s), str)
        assert isinstance(len(res_ellipsoid_s), int)
    
    def test_result_getitem(self, res_ellipsoid_s: ModelResult):
        res_1 = res_ellipsoid_s[0,2,3,5]
        assert isinstance(res_1, ModelResult)
        with pytest.raises(TypeError):
            res_ellipsoid_s[1,0.8]

        res_1 = res_ellipsoid_s[[0,2,3,5]]
        assert isinstance(res_1, ModelResult)
        
        res_1 = res_ellipsoid_s["eps_ab"]
        assert isinstance(res_1, np.ndarray)
        
        res_1 = res_ellipsoid_s["eps_ab_err"]
        assert isinstance(res_1, np.ndarray)
        
        with pytest.raises(KeyError):
            res_ellipsoid_s["non_existent_key"]
        
        res_1 = res_ellipsoid_s[5]
        res_1 = res_ellipsoid_s[-10]
        
        res_1 = res_ellipsoid_s[np.ones(len(res_ellipsoid_s), dtype=bool)]
        assert isinstance(res_1, ModelResult)
    
    
    def test_ipython(self, res_ellipsoid_s: ModelResult):
        """Test ModelResult IPython display."""
        html = res_ellipsoid_s._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html and "</table>" in html
        text = res_ellipsoid_s.__repr__()
        assert isinstance(text, str)
        assert "ModelResult" in text
        res_ellipsoid_s._ipython_key_completions_()
    
    def test_attribute_access(self, res_ellipsoid_s: ModelResult):
        res_ellipsoid_s.cost
    
    def test_head_and_tail(self, res_ellipsoid_s: ModelResult):
        head = res_ellipsoid_s.head(5)
        tail = res_ellipsoid_s.tail(5)
        assert isinstance(head, ModelResult)
        assert isinstance(tail, ModelResult)
        assert len(head) == 5
        assert len(tail) == 5
        
        

    def test_result_get(self, res_ellipsoid_s: ModelResult):
        """Test ModelResult get method."""
        assert res_ellipsoid_s.get("parameter") is not None

    def test_result_slice_and_index(self, res_ellipsoid_s: ModelResult):
        """Test ModelResult slicing and indexing."""
        assert isinstance(res_ellipsoid_s[0:10], ModelResult)
        with pytest.raises(IndexError):
            _ = res_ellipsoid_s[200]
        with pytest.raises(KeyError):
            _ = res_ellipsoid_s[0.1]

    def test_result_invalid_attribute(self, res_ellipsoid_s: ModelResult):
        """Test ModelResult invalid attribute access."""
        with pytest.raises(AttributeError):
            _ = res_ellipsoid_s.non_existent_attribute


    def test_result_addition(self, res_ellipsoid_s: ModelResult):
        """Test addition of ModelResult objects."""
        other = res_ellipsoid_s[0:5]
        combined = res_ellipsoid_s + other
        assert isinstance(combined, ModelResult)
        assert len(combined) != len(res_ellipsoid_s)

    def test_result_call(self, res_ellipsoid_s: ModelResult):
        """Test __call__ method of ModelResult."""
        func = res_ellipsoid_s[5]
        result = func([1.0, 2.0, 3.0])
        assert result is not None

    def test_empty_result_behavior(self, res_ellipsoid_s: ModelResult):
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


# --- ModelResult generation methods ---
def test_model_result_generate_methods(res_ellipsoid_s: ModelResult):
    """Test ModelResult generation methods for edge, slice, and points."""
    res_ellipsoid_s[10].generate_edge2D(n_angle_bins=10, n_r_bins=10)
    res_ellipsoid_s[15].generate_edge3D(n_phi_bins=10, n_theta_bins=5)
    res_ellipsoid_s[15].generate_slice2D(n_bins=8)
    res_ellipsoid_s[20].generate_points(32)

# --- Visualization and field tests ---
def test_gal3d_field_and_visualization(gal: Gal3DAnalyzer, res_ellipsoid_s: ModelResult, out_dir):
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
    
    
    assert len(gal.density_source.pos) == len(gal.density_source.mass) == len(gal.density_source.hsm)
    # is finite and non infinite values
    assert np.isfinite(gal.density_source.pos).all()
    assert np.isfinite(gal.density_source.mass).all()
    assert np.isfinite(gal.density_source.hsm).all()
    assert (gal.density_source.mass > 0).all()
    assert (gal.density_source.hsm > 0).all()

    out = out_dir / "test_gal3d_field_and_visualization.png"
    fig = show_image_model_residual(
        gal.density_source, ellipsoid_s_model,
        large_box_x_range=(-box_lh_max, box_lh_max),
        large_box_y_range=(-box_lh_max, box_lh_max),
        zoom_x_range=(-zoom_lh_max, zoom_lh_max),
        zoom_y_range=(-zoom_lh_max, zoom_lh_max),
        depth_z_range=(-box_lh_max, box_lh_max),
        nbins_large=50,
        nbins_zoom=25,
        nlevels_large=0,
        nlevels_zoom=22,
        savefile=out,
        render=True
    )
    assert out.exists()

def test_gal3d_analyzer_fit_single_radius(gal: Gal3DAnalyzer):
    """Test fitting with a single radius value."""
    res = gal.fit(radius=5.0)
    assert isinstance(res, ModelResult)
    res = gal.fit(radius=0.01)
    assert isinstance(res, EmptyModelResult)


def test_bar_characterizer_plugin():
    """Test the Bar characterizer plugin availability."""
    assert Characterizer.get_plugin("Bar") is Bar

def test_bar_measure(res_ellipsoid_s: ModelResult):
    """Test the measure method of the Bar characterizer plugin using ModelResult data."""
    bar = Bar(res_ellipsoid_s)
    result = bar.measure(other_keys=["sa"])
    assert result["flag"] == 1

def test_disk_characterizer_plugin():
    """Test the Disk characterizer plugin availability."""
    assert Characterizer.get_plugin("Disk") is Disk

def test_disk_measure(res_ellipsoid_s: ModelResult):
    """Test the measure method of the Disk characterizer plugin using ModelResult data."""
    disk = Disk(res_ellipsoid_s)
    result = disk.measure(other_keys=["sa"])
    assert result["flag"] == 1
    

def test_hdf5_model_io_plugin():
    """Test the HDF5ModelIO plugin availability."""
    assert ModelIO.get_plugin("HDF5ModelIO") is HDF5ModelIO

def test_model_save_load_hdf5(res_ellipsoid_s: ModelResult, tmp_path):
    """Test saving and loading a ModelResult using HDF5ModelIO."""
    path = tmp_path / "model_result.h5"
    HDF5ModelIO.save(res_ellipsoid_s, filename=str(path), metadata={"test": "value"}, overwrite=True)
    assert path.exists()
    loaded = HDF5ModelIO.load(str(path))
    assert isinstance(loaded, ModelResult)
    assert loaded.structure.is_equal(res_ellipsoid_s.structure)
    assert (loaded["parameter"] == res_ellipsoid_s["parameter"]).all()