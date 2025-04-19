

import numpy as np

def test_parameter_num(structure):
    assert len(structure.parameters) == len(structure._geometry.PN+structure._coordinate.PN)
    
    
def test_name(structure):
    assert isinstance(structure._geometry_name,str)
    assert isinstance(structure._coordinate_name,str)
    assert isinstance(structure._error_func_name,str)
    assert isinstance(structure._error_method_name,str)
    
def test_call(structure,particle_data):
    assert len(structure(particle_data.pos)) == len(particle_data.mass)
    assert len(structure.f_ray_d(particle_data.pos)) == len(particle_data.mass)
    pos1 = np.array([[-20.,-20,-10.],[19,18.,-20.]])
    pos2 = np.array([[20.,10,10.],[10.,10.,20]])
    assert len(structure.line_intersect(pos1,pos2)) == 2
    
def test_generate(structure):
    assert structure.generate_points(100).shape == (100,3)
    
def test_parameter(structure):
    assert len(structure.init_parameters()) == len(structure.parameters)