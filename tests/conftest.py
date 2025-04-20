import os
import itertools

import pytest
import numpy as np

from gal3d.point import GlobalCalculator, Particles
from gal3d.field import SphField
from gal3d.shape import Structure3D
from gal3d.optimization.optimizer import Optimizer
from gal3d.analyzer import Gal3DAnalyzer

@pytest.fixture(scope='session')
def point_data():
    _current_dir = os.path.dirname(__file__)
    data = np.loadtxt(_current_dir+"/test_data/TNG50_Snap99_Subhalo282780_star.txt")
    pos = data[:,:3]
    mass =data[:,3]
    return pos,mass


@pytest.fixture(scope='session')
def global_data(point_data):
    return GlobalCalculator(*point_data)


@pytest.fixture(scope='session')
def particle_data(point_data):
    return Particles(*point_data)

@pytest.fixture(scope='session')
def field_data(particle_data):
    return SphField(particle_data,num_ray=1024)


@pytest.fixture(scope='session',params=[[0.3,'dist',5e4,'value'],[5,'pct',95,'pct']])
def field_data_with_boundary(field_data,request):
    inner,inner_mode,outer,outer_mode = request.param
    yield field_data.build_field_boundary(inner=inner,inner_mode=inner_mode,outer=outer,outer_mode=outer_mode)

@pytest.fixture(scope='session',params=[[500,'log'],[300,'lin']])
def field_with_sample(field_data_with_boundary,request):
    num_p,step_mode = request.param
    yield field_data_with_boundary.build_profile_sample(num_p,step_mode)

@pytest.fixture(scope='session',params=['LU','SG'])
def field_with_interpolate(field_with_sample,request):
    yield field_with_sample.build_profile_interpolator(interpolator_method= request.param)


@pytest.fixture(scope='session',params=[True,False])
def field(field_with_interpolate,request):
    yield field_with_interpolate.build_isodensity_profile(from_rays_func= request.param)


@pytest.fixture(scope='session',params=[['OptimizerScipy','Powell'],['OptimizerNLopt','LN_NELDERMEAD']])
def optimizer(request):
    plugin,algorithm = request.param
    optimizer = Optimizer.get_plugin(plugin = plugin)(algorithm=algorithm)
    yield optimizer

@pytest.fixture(scope='session',params= [["Ellipsoid","sums_dev","isodensity_fcall"],["Ellipsoid_S","sums_dev_rscale","isodensity_dcall"]])
def structure(request):
    geometry,error_func,error_method = request.param
    
    yield Structure3D(coordinate='EulerShift',geometry=geometry,error_func=error_func,error_method=error_method)


@pytest.fixture(scope='session')
def field_once(particle_data):
    field = SphField(particle_data,num_ray=1024)
    field.build_field_boundary(inner=0.6,inner_mode='dist',outer=5e4,outer_mode='value')
    field.build_profile_sample()
    field.build_profile_interpolator().build_profile_interpolator().build_isodensity_profile()
    return field

@pytest.fixture(scope='session')
def gal(particle_data,field_once,structure,optimizer):
    
    yield Gal3DAnalyzer(particle_data,field_once,structure,optimizer)

@pytest.fixture(scope='session')
def gal_data_for_visualize(particle_data,field_once,structure):
    
    optimizer = Optimizer.get_plugin(plugin = 'OptimizerScipy')(algorithm='Powell')
    yield Gal3DAnalyzer(particle_data,field_once,structure,optimizer)
    