
import pytest
import numpy as np

from gal3d.field.spherical_field.spherical_harmonic import spherical_harmonics_dec
from gal3d.util.array_operate import trans_to_Spherical_coordinates

def test_ray_num(field):
    assert field.rays.num == len(field.rays_points_num)
    assert len(field.rays_func) == len(field.rays_points_num)
    


@pytest.mark.parametrize("r,for_fit", [(1,True),(3,False),(np.linspace(1,10,10),True)])
def test_generate(field,r,for_fit):
    assert isinstance(field.generate(r,for_fit),dict)
    
    
def test_ray_point_num(field):
    assert len(field.r_ray_n(2)) == len(field.mass_ray_n(2))
    assert len(field.parameter_ray_n(5)) == len(field.mass_ray_n(5))
    
    
def test_spherical_harmonics_dec(field):
    
    density = field.query_rays_f(2)
    sp = trans_to_Spherical_coordinates(field.rays_vect)
    assert len(density) == len(sp)
    assert isinstance(spherical_harmonics_dec(sp[:,1],sp[:,2],density),dict)