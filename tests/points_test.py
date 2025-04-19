import numpy as np








def test_data_size(point_data,global_data):
    assert len(point_data[0])==len(global_data.pos)
    assert len(point_data[0][0])==len(global_data.pos[0])
    assert len(point_data[1]) == len(global_data.mass)
    
def test_sort_r(global_data):
    assert np.all(np.diff(global_data.r)>=0)
    
def test_ssc_center(global_data):
    assert global_data.ssc_center.shape == (3,)
    
def test_ssc_center(global_data):
    assert global_data.mass_center.shape == (3,)
    
def test_ssc_center(global_data):
    assert global_data.shape_center.shape == (3,)

def test_ssc_center(global_data):
    assert global_data.moi.shape == (3,3)
    
    
def test_ssc_center(global_data):
    assert global_data.abc[0].shape == (3,)
    assert global_data.abc[1].shape == (3,3)
    
def test_points_parameter(particle_data):
    assert particle_data.parameter.shape == particle_data.mass.shape