import pytest
import numpy as np
from gal3d.optimization.parameter import Parameter, Parameters


def test_single_parameter():
    param = Parameter(3,lb="0.1")

    # Check parameter properties
    assert isinstance(param, float)
    assert param == 3
    assert param.lb == 0.1
    assert param.ub is np.inf

    # Check assign_bounds
    param.assign_bounds(0.0, 5.0)
    assert param.lb == 0.0
    assert param.ub == 5.0

    # Check direct assignment
    param.lb = 2
    assert param.lb == 2
    assert isinstance(param.lb, float)

    # Check assign_value would not change bounds
    param = param.assign_value(4)
    assert param == 4
    assert param.lb == 2
    assert param.ub == 5.0

def test_parameters_basic():
    
    params = Parameters(a=2,b=3)
    assert isinstance(params['a'], Parameter)
    assert isinstance(params['a'].lb, float)
    
    # test set bounds
    params.set_lb(b=4)
    params.set_ub(a=5)
    assert params['b'].lb == 4
    assert params['a'].ub == 5
    
    # test setitem, would not influence bounds
    params['a'] = 6
    assert params['a'] == 6
    assert params['a'].ub == 5

    # test set_value, would not influence bounds
    params.set_value(a=3,b=7)
    assert params['a'] == 3
    assert params['b'] == 7
    assert params['b'].lb == 4
    assert params['a'].ub == 5
    
    # test some basic use
    params.keys()
    params.scipy_bounds
    params.value
    params.ub
    params.lb

    assert params.values_list() == [3., 7.]
    
    
