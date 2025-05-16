import numpy as np
import pytest
from gal3d.point import Particles
from gal3d.point.density_estimator import DensityEstimatorBase

def test_particles_initialization():
    # Mock data
    pos = np.random.rand(100, 3)  # 100 particles in 3D space
    mass = np.random.rand(100)    # 100 mass values

    # Initialize Particles
    particles = Particles(pos, mass, rmax=0.5)

    # Check if particles are filtered correctly by rmax
    assert len(particles.pos) <= len(pos)
    assert len(particles.mass) <= len(mass)

def test_parameter_property():
    # Mock data
    pos = np.random.rand(100, 3)
    mass = np.random.rand(100)

    # Initialize Particles
    particles = Particles(pos, mass)

    # Check if parameter property is accessible
    assert particles.parameter is not None

def test_gradient_property():
    # Mock data
    pos = np.random.rand(100, 3)
    mass = np.random.rand(100)

    # Initialize Particles
    particles = Particles(pos, mass)

    # Check if gradient property is accessible
    assert particles.gradient is not None

def test_get_parameter():
    # Mock data
    pos = np.random.rand(100, 3)
    mass = np.random.rand(100)
    target_pos = np.random.rand(10, 3)  # 10 target positions

    # Initialize Particles
    particles = Particles(pos, mass)

    # Test get_parameter method
    parameter_values = particles.get_parameter(target_pos)
    assert len(parameter_values) == len(target_pos)

def test_get_gradient():
    # Mock data
    pos = np.random.rand(100, 3)
    mass = np.random.rand(100)
    target_pos = np.random.rand(10, 3)  # 10 target positions

    # Initialize Particles
    particles = Particles(pos, mass)

    # Test get_gradient method
    gradient = particles.get_gradient(target_pos)
    assert len(gradient) == 2  # Should return two tuples

def test_available_estimator():
    # Check if available_estimator class property is accessible
    assert isinstance(Particles.available_estimator, list)
    for estimator in Particles.available_estimator:
        assert issubclass(estimator, DensityEstimatorBase)