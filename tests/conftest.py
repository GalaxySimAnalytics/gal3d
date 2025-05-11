import os
import itertools
import logging
from typing import Generator

import pytest
import numpy as np

from gal3d.point import GlobalCalculator, Particles
from gal3d.field import SphField
from gal3d.shape import Structure3D
from gal3d.optimization.optimizer import Optimizer
from gal3d.analyzer import Gal3DAnalyzer

logger = logging.getLogger(__name__)

@pytest.fixture(scope='session')
def point_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load point data from a test file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing positions and masses.
    """
    try:
        _current_dir = os.path.dirname(__file__)
        data = np.loadtxt(os.path.join(_current_dir, "test_data/TNG50_Snap99_Subhalo282780_star.txt"))
        pos = data[:, :3]
        mass = data[:, 3]
        return pos, mass
    except Exception as e:
        logger.error("Failed to load point data: %s", e, exc_info=True)
        raise

@pytest.fixture(scope='session')
def global_data(point_data: tuple[np.ndarray, np.ndarray]) -> GlobalCalculator:
    """
    Create a GlobalCalculator instance.

    Parameters
    ----------
    point_data : tuple[np.ndarray, np.ndarray]
        Positions and masses.

    Returns
    -------
    GlobalCalculator
        An instance of GlobalCalculator.
    """
    return GlobalCalculator(*point_data)

@pytest.fixture(scope='session')
def particle_data(point_data: tuple[np.ndarray, np.ndarray]) -> Particles:
    """
    Create a Particles instance.

    Parameters
    ----------
    point_data : tuple[np.ndarray, np.ndarray]
        Positions and masses.

    Returns
    -------
    Particles
        An instance of Particles.
    """
    return Particles(*point_data)

@pytest.fixture(scope='session')
def field_data(particle_data: Particles) -> SphField:
    """
    Create an SphField instance.

    Parameters
    ----------
    particle_data : Particles
        Particle data.

    Returns
    -------
    SphField
        An instance of SphField.
    """
    return SphField(particle_data, num_ray=1024)


@pytest.fixture(scope='session',params=[[0.3,'dist',5e4,'value'],[5,'pct',95,'pct']])
def field_data_with_boundary(field_data: SphField, request) -> Generator[SphField, None, None]:
    """
    Create a field with boundary using different boundary configurations.
    
    Parameters
    ----------
    field_data : SphField
        Base field data without boundaries.
    request : pytest.FixtureRequest
        Fixture request containing parameters.
    
    Returns
    -------
    SphField
        Field with boundaries applied.
    
    Notes
    -----
    Tests two different boundary configurations:
    1. Distance-based inner (0.3) and value-based outer (5e4)
    2. Percentile-based inner (5%) and percentile-based outer (95%)
    """
    try:
        inner, inner_mode, outer, outer_mode = request.param
        logger.debug(f"Building field boundary with inner={inner} ({inner_mode}), "
                    f"outer={outer} ({outer_mode})")
        yield field_data.build_field_boundary(
            inner=inner, inner_mode=inner_mode, 
            outer=outer, outer_mode=outer_mode
        )
    except Exception as e:
        logger.error(f"Failed to build field boundary: {e}", exc_info=True)
        raise

@pytest.fixture(scope='session',params=[[500,'log'],[300,'lin']])
def field_with_sample(field_data_with_boundary, request) -> Generator[SphField, None, None]:
    """
    Create a field with sample profile using different sampling configurations.
    
    Parameters
    ----------
    field_data_with_boundary : SphField
        Field data with boundary already set.
    request : pytest.FixtureRequest
        Fixture request containing parameters.
    
    Returns
    -------
    SphField
        Field with sample profile built.
    
    Notes
    -----
    Tests both logarithmic and linear sampling with different point counts.
    """
    try:
        num_p, step_mode = request.param
        logger.debug(f"Building profile sample with {num_p} points using {step_mode} mode")
        yield field_data_with_boundary.build_profile_sample(num_p, step_mode)
    except Exception as e:
        logger.error(f"Failed to build profile sample: {e}", exc_info=True)
        raise

@pytest.fixture(scope='session',params=['LU','SG'])
def field_with_interpolate(field_with_sample, request) -> Generator[SphField, None, None]:
    """
    Create a field with interpolation using different interpolation methods.
    
    Parameters
    ----------
    field_with_sample : SphField
        Field with sample profile already built.
    request : pytest.FixtureRequest
        Fixture request containing parameters.
    
    Returns
    -------
    SphField
        Field with interpolator built.
    
    Notes
    -----
    Tests both LU and SG interpolation methods.
    """
    try:
        interpolator_method = request.param
        logger.debug(f"Building profile interpolator using {interpolator_method} method")
        yield field_with_sample.build_profile_interpolator(interpolator_method=interpolator_method)
    except Exception as e:
        logger.error(f"Failed to build profile interpolator: {e}", exc_info=True)
        raise

@pytest.fixture(scope='session',params=[True,False])
def field(field_with_interpolate, request) -> Generator[SphField, None, None]:
    """
    Create a field with isodensity profile.
    
    Parameters
    ----------
    field_with_interpolate : SphField
        Field with interpolator already built.
    request : pytest.FixtureRequest
        Fixture request containing parameters.
    
    Returns
    -------
    SphField
        Field with isodensity profile built.
    
    Notes
    -----
    Tests with both from_rays_func=True and False.
    """
    try:
        from_rays_func = request.param
        logger.debug(f"Building isodensity profile with from_rays_func={from_rays_func}")
        yield field_with_interpolate.build_isodensity_profile(from_rays_func=from_rays_func)
    except Exception as e:
        logger.error(f"Failed to build isodensity profile: {e}", exc_info=True)
        raise

@pytest.fixture(scope='session',params=[['OptimizerScipy','Powell'],['OptimizerNLopt','LN_NELDERMEAD']])
def optimizer(request) -> Generator[Optimizer, None, None]:
    """
    Create an optimizer with different algorithms.
    
    Parameters
    ----------
    request : pytest.FixtureRequest
        Fixture request containing parameters.
    
    Returnsfrom typing import Generator
    -------
    Optimizer
        Optimizer instance with specified plugin and algorithm.
    
    Notes
    -----
    Tests both scipy's Powell and NLopt's Nelder-Mead optimization algorithms.
    """
    try:
        plugin, algorithm = request.param
        logger.debug(f"Creating optimizer with plugin={plugin} and algorithm={algorithm}")
        optimizer = Optimizer.get_plugin(plugin=plugin)(algorithm=algorithm)
        yield optimizer
    except Exception as e:
        logger.error(f"Failed to create optimizer with {plugin}/{algorithm}: {e}", exc_info=True)
        raise

@pytest.fixture(scope='session',params= [["Ellipsoid","sums_dev","isodensity_fcall"],["Ellipsoid_S","sums_dev_rscale","isodensity_dcall"]])
def structure(request) -> Generator[Structure3D, None, None]:
    """
    Create a 3D structure with different geometries and error methods.
    
    Parameters
    ----------
    request : pytest.FixtureRequest
        Fixture request containing parameters.
    
    Returns
    -------
    Structure3D
        Structure with specified geometry, error function and method.
    
    Notes
    -----
    Tests both regular Ellipsoid and Ellipsoid_S geometries with appropriate error functions.
    """
    try:
        geometry, error_func, error_method = request.param
        logger.debug(f"Creating structure with geometry={geometry}, error_func={error_func}, error_method={error_method}")
        yield Structure3D(coordinate='EulerShift', geometry=geometry, error_func=error_func, error_method=error_method)
    except Exception as e:
        logger.error(f"Failed to create structure with {geometry}/{error_func}/{error_method}: {e}", exc_info=True)
        raise

@pytest.fixture(scope='session')
def field_once(particle_data: Particles) -> SphField:
    """
    Create a fully configured field with a predefined configuration.
    
    Parameters
    ----------
    particle_data : Particles
        Particle data to use for field creation.
    
    Returns
    -------
    SphField
        Fully configured field with all processing steps completed.
    """
    try:
        logger.info("Creating field with standard configuration")
        field = SphField(particle_data, num_ray=1024)
        field.build_field_boundary(inner=0.6, inner_mode='dist', outer=5e4, outer_mode='value')
        field.build_profile_sample()
        field.build_profile_interpolator().build_profile_interpolator().build_isodensity_profile()
        return field
    except Exception as e:
        logger.error(f"Failed to create standard field: {e}", exc_info=True)
        raise

@pytest.fixture(scope='session')
def gal(particle_data: Particles, field_once: SphField, structure: Structure3D, optimizer: Optimizer) -> Gal3DAnalyzer:
    """
    Create a Gal3DAnalyzer instance.

    Parameters
    ----------
    particle_data : Particles
        Particle data.
    field_once : SphField
        Pre-built field data.
    structure : Structure3D
        Structure data.
    optimizer : Optimizer
        Optimizer instance.

    Returns
    -------
    Gal3DAnalyzer
        An instance of Gal3DAnalyzer.
    """
    try:
        return Gal3DAnalyzer(particle_data, field_once, structure, optimizer)
    except Exception as e:
        logger.error("Failed to create Gal3DAnalyzer: %s", e, exc_info=True)
        raise

@pytest.fixture(scope='session')
def gal_data_for_visualize(particle_data: Particles, field_once: SphField, structure: Structure3D) -> Generator[Gal3DAnalyzer, None, None]:
    """
    Create a Gal3DAnalyzer instance specifically configured for visualization purposes.
    
    This fixture uses a specific optimizer (OptimizerScipy with Powell algorithm),
    which is particularly suitable for visualization tasks.
    
    Parameters
    ----------
    particle_data : Particles
        Particle data to use for analysis.
    field_once : SphField
        Pre-built field data.
    structure : Structure3D
        Structure model configuration.
    
    Returns
    -------
    Gal3DAnalyzer
        An analyzer instance ready for visualization work.
        
    Notes
    -----
    This fixture differs from the standard 'gal' fixture by always using
    the same optimizer regardless of test parameterization.
    """
    try:
        logger.debug("Creating visualization-specific analyzer with OptimizerScipy/Powell")
        optimizer = Optimizer.get_plugin(plugin='OptimizerScipy')(algorithm='Powell')
        yield Gal3DAnalyzer(particle_data, field_once, structure, optimizer)
    except Exception as e:
        logger.error(f"Failed to create analyzer for visualization: {e}", exc_info=True)
        raise
