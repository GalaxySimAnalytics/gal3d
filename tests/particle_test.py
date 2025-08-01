import os

import numpy as np
import pytest

from gal3d.point import Particles


@pytest.fixture
def data():
    ID = 46
    return np.load(f"{os.path.dirname(__file__)}/test_data/TNG50_ID{ID}_star_particles_float32.npy")

@pytest.fixture
def pos_mass(data):
    return data[:,:3], data[:,3]


@pytest.fixture
def particles(pos_mass):
    pos, mass = pos_mass
    return Particles(pos=pos, mass=mass, recenter=False)

def test_particles_init(particles, pos_mass):
    pos, _ = pos_mass
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
    ssc = particles.ssc_center
    assert isinstance(ssc, np.ndarray)
    assert ssc.shape == (3,)

def test_abc(particles):
    abc = particles.abc
    assert isinstance(abc, tuple)
    assert len(abc) == 2
    assert abc[0].shape == (3,)
    assert abc[1].shape == (3,3)

def test_mass_center(particles):
    mc = particles.mass_center
    assert isinstance(mc, np.ndarray)
    assert mc.shape == (3,)

def test_shape_center(particles):
    sc = particles.shape_center
    assert isinstance(sc, np.ndarray)
    assert sc.shape == (3,)
    
    
def test_estimate_mass_resolution(particles):
    res = particles.estimate_mass_resolution()
    assert isinstance(res, (float, np.floating))


def test_estimate_spatial_resolution(particles):
    res = particles.estimate_spatial_resolution()
    assert isinstance(res, (float, np.floating))

def test_rmax_limit(pos_mass):
    pos, mass = pos_mass
    par2 = Particles(pos=pos, mass=mass, rmax=10, recenter=False)
    assert np.max(par2.r) < 10

def test_float64_input(data):
    pos = np.float64(data[:,:3])
    mass = np.float64(data[:,3])
    par = Particles(pos=pos, mass=mass, recenter=False)
    par.gradient
    assert par.pos.dtype == np.float64
    assert par.mass.dtype == np.float64
    assert par.hsm.dtype == np.float64
    assert par.parameter.dtype == np.float64
    assert par.moi.dtype == np.float64
    assert isinstance(par.as_dict(), dict)

