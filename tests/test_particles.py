"""Unit tests for the Particles class in gal3d.point."""


import pytest

import numpy as np

from gal3d.point import Particles
from gal3d.visualization.show import show_image


@pytest.fixture
def particles(tng_galaxy_data):
    """Load TNG50 galaxy data and create Particles instance for testing."""
    pos, mass = tng_galaxy_data
    return Particles(pos=pos, mass=mass, recenter=False)

@pytest.fixture
def small_particles(small_random_galaxy_data):
    """Create a smaller Particles instance from random galaxy data for faster tests."""
    pos, mass = small_random_galaxy_data
    return Particles(pos=pos, mass=mass, recenter=False)

def test_particles_init(particles, tng_galaxy_data):
    """Test the initialization of Particles and the properties it computes."""
    pos, _ = tng_galaxy_data
    assert particles.pos.shape[1] == 3
    assert particles.mass.shape[0] == pos.shape[0]

    particles.gradient
    assert particles.pos.dtype == np.float32
    assert particles.mass.dtype == np.float32
    assert particles.hsm.dtype == np.float64        # float64
    assert particles.parameter.dtype == np.float64  # float64
    assert particles.moi.dtype == np.float32
    assert isinstance(particles.as_dict(), dict)


def test_ssc_center(particles):
    """Test the computation of the shape center (SSC) of the particles."""
    ssc = particles.ssc_center
    assert isinstance(ssc, np.ndarray)
    assert ssc.shape == (3,)

def test_abc(particles):
    """Test the computation of the shape parameters (abc) of the particles."""
    abc = particles.abc
    assert isinstance(abc, tuple)
    assert len(abc) == 2
    assert abc[0].shape == (3,)
    assert abc[1].shape == (3,3)

def test_mass_center(particles):
    """Test the computation of the mass center of the particles."""
    mc = particles.mass_center
    assert isinstance(mc, np.ndarray)
    assert mc.shape == (3,)

def test_shape_center(particles):
    """Test the computation of the shape center of the particles."""
    sc = particles.shape_center
    assert isinstance(sc, np.ndarray)
    assert sc.shape == (3,)
    
    
def test_estimate_mass_resolution(particles):
    """Test the estimation of mass resolution from the particles."""
    res = particles.estimate_mass_resolution()
    assert isinstance(res, (float, np.floating))


def test_estimate_spatial_resolution(particles):
    """Test the estimation of spatial resolution from the particles."""
    res = particles.estimate_spatial_resolution()
    assert isinstance(res, (float, np.floating))

def test_rmax_limit(tng_galaxy_data):
    """Test the rmax limit functionality of the Particles class."""
    pos, mass = tng_galaxy_data
    par2 = Particles(pos=pos, mass=mass, rmax=10, recenter=False)
    assert np.max(par2.r) < 10

def test_float64_input(tng_galaxy_data):
    """Test that Particles can be initialized with float64 data and maintains the correct dtypes."""
    pos, mass = tng_galaxy_data
    pos = np.float64(pos)
    mass = np.float64(mass)
    par = Particles(pos=pos, mass=mass, recenter=False)
    par.gradient
    assert par.pos.dtype == np.float64
    assert par.mass.dtype == np.float64
    assert par.hsm.dtype == np.float64
    assert par.parameter.dtype == np.float64
    assert par.moi.dtype == np.float64
    assert isinstance(par.as_dict(), dict)

def test_particles_render_sph(small_particles, out_dir):
    rendered = small_particles.project_2d([-40, 40], [-40, 40], 300)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    show_image(rendered, axesObj=ax)

    out = out_dir / "particles_render_sph.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    assert out.exists()

def test_particles_render_los(small_particles, out_dir):
    rendered = small_particles.project_2d([-40, 40], [-40, 40], 300, method="los")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    show_image(rendered, axesObj=ax)

    out = out_dir / "particles_render_los.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    assert out.exists()


def test_particles_shape(small_particles):
    """Test the shape method of the Particles class."""
    # 1:0.5:0.25
    shape = small_particles.shape_at(30)
    
    assert shape["eps_ab"] == pytest.approx(0.5, abs=0.05)
    assert shape["eps_ac"] == pytest.approx(0.75, abs=0.05)
    

