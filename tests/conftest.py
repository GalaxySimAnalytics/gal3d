"""Fixtures for testing gal3d configuration and parameters."""

from pathlib import Path

import numpy as np
import pytest
from gal3d import config


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Ensure test data is available for the whole test session."""
    # Check if the test data file exists
    ID = 46
    path = Path(__file__).parent / "test_data" / f"TNG50_ID{ID}_star_particles_float32.npy"
    if not path.exists():
        pytest.exit(f"Test data file {path} is missing. Please ensure it is available.")
    yield
    # No special teardown needed for test data

@pytest.fixture(scope="session")
def out_dir(pytestconfig) -> Path:
    """Session-wide output directory under the repository root."""
    path = pytestconfig.rootpath / "out" / "tests"
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture(scope="function", autouse=True)
def force_single_thread():
    old_threads = config.general.number_of_threads
    config.general.number_of_threads = 1
    yield
    config.general.number_of_threads = old_threads

@pytest.fixture(scope="function")
def tng_galaxy_data():
    """Load TNG50 galaxy data for testing."""
    ID = 46
    path = Path(__file__).parent / "test_data" / f"TNG50_ID{ID}_star_particles_float32.npy"
    data = np.load(path)
    pos = data[:, :3] # x, y, z positions
    mass = data[:, 3] # masses
    return pos, mass



@pytest.fixture(scope="function")
def random_galaxy_data(num_particles=10000, axis_ratios=(1.0, 0.5, 0.25)):
    """Generate random galaxy data for testing."""
    #gaussian distribution of positions with most lie within 30 kpc, and masses between 0 and 10^10 Msun
    # fix random seed for reproducibility
    np.random.seed(42)
    positions = np.random.randn(num_particles, 3) * 15.0  # scale to typical galaxy size
    masses = np.random.rand(num_particles).astype(np.float32) * 1e10  # random masses up to 10^10 Msun
    positions[:, 1] *= axis_ratios[1]  # flatten y-axis
    positions[:, 2] *= axis_ratios[2]  # flatten z-axis
    return positions, masses


@pytest.fixture(scope="function")
def small_random_galaxy_data(random_galaxy_data):
    """Generate a smaller random galaxy dataset for faster tests."""
    pos, mass = random_galaxy_data
    return pos[:1000], mass[:1000]
