"""
Unit tests for gal3d.optimization.parameter.Parameter and Parameters.
Covers bounds, assignment, copying, constraints, derived values, and utility methods.
"""

import pytest
import numpy as np
from gal3d.optimization.parameter import Parameter, Parameters

# -------------------------------
# Parameter basic functionality
# -------------------------------

def test_parameter_bounds_and_hash():
    """Test Parameter's bounds properties and hash consistency"""
    p1 = Parameter(1.0, lb=0.0, ub=2.0)
    p2 = Parameter(1.0, lb=0.0, ub=2.0)
    assert hash(p1) == hash(p2)
    assert p1.lb == 0.0
    assert p1.ub == 2.0

def test_parameter_value_and_bounds_assignment():
    """Test Parameter value and bounds assignment."""
    p = Parameter(5.0)
    p2 = p.assign_value(10.0)
    assert p2 == 10.0
    assert p2.lb == p.lb
    assert p2.ub == p.ub

    p.assign_bounds(-1.0, 1.0)
    assert p.lb == -1.0
    assert p.ub == 1.0
    
def test_parameter_invalid_bounds():
    """Test Parameter raises ValueError for invalid bounds."""
    p = Parameter(2.0, lb=1.0, ub=3.0)
    with pytest.raises(ValueError):
        p.lb = 4
        p.ub = 0

def test_parameter_single_instance_behavior():
    """Test Parameter instance attributes and assignment."""
    param = Parameter(3, lb="0.1")
    hash(param)
    assert isinstance(param, float)
    assert param == 3
    assert param.lb == 0.1
    assert param.ub is np.inf

    param.assign_bounds(0.0, 5.0)
    assert param.lb == 0.0
    assert param.ub == 5.0

    param.lb = 2
    assert param.lb == 2
    assert isinstance(param.lb, float)

    param = param.assign_value(4)
    assert param == 4
    assert param.lb == 2
    assert param.ub == 5.0

# -------------------------------
# Parameters dictionary functionality
# -------------------------------

def test_parameters_set_value_and_bounds():
    """Test Parameters set_value and bounds."""
    pd = Parameters(a=1, b=2)
    pd.set_value(5, 6)
    assert pd['a'] == 5
    assert pd['b'] == 6
    pd.set_value(a=7)
    assert pd['a'] == 7
    with pytest.raises(KeyError):
        pd.set_value(c=1)

    pd['a'].lb = -1
    pd['a'].ub = 1
    with pytest.raises(ValueError):
        pd.set_ub(a=-10)  # ub < lb

def test_parameters_update_and_copy_behavior():
    """Test Parameters update, copy, and deepcopy."""
    pd = Parameters(a=1, b=2)
    pd2 = Parameters(a=3, b=4)
    pd.update(pd2)
    assert pd['a'] == 3
    assert pd['b'] == 4
    pd3 = pd.copy()
    assert pd3['a'] == 3
    pd4 = pd.deepcopy()
    assert pd4['b'] == 4
    # Ensure shallow or deep copies, the Parameter instances should be different
    assert id(pd3['b']) != id(pd['b']) 
    assert id(pd4['b']) != id(pd['b']) 

def test_parameters_basic_usage():
    """Test Parameters basic usage and properties."""
    params = Parameters(a=2, b=3)
    assert isinstance(params['a'], Parameter)
    assert isinstance(params['a'].lb, float)

    with pytest.raises(KeyError):
        params.get_parameter("c")
    params.parameter_keys()
    params.get_rounded()
    with pytest.raises(ValueError):
        params.get_rounded(-1)
    params.copy()
    params.deepcopy()
    params.available_keys()
    repr(params)

    params.set_lb(b=4)
    params.set_ub(a=5)
    assert params['b'].lb == 4
    assert params['a'].ub == 5

    params['a'] = 6
    assert params['a'] == 6
    assert params['a'].ub == 5

    params.set_value(a=3, b=7)
    assert params['a'] == 3
    assert params['b'] == 7
    assert params['b'].lb == 4
    assert params['a'].ub == 5

    params.keys()
    params.scipy_bounds
    params.value
    params.ub
    params.lb
    assert params.values_list() == [3., 7.]

def test_parameters_new_and_add_methods():
    """Test Parameters new and __add__ methods."""
    params = Parameters(a=1, b=2)
    new_params = params.new(a=3, b=4)
    assert new_params['a'] == 3
    assert new_params['b'] == 4
    params2 = Parameters(a=5)
    params3 = params + params2
    assert params3['a'] == 5
    assert params3['b'] == 2

def test_parameters_structure_and_rounded_values():
    """Test Parameters structure_parameters and get_rounded_values_dict methods."""
    params = Parameters(a=1.2345, b=2.3456)
    struct = params.structure_parameters
    assert isinstance(struct, dict)
    rounded = params.get_rounded_values_dict(n=2)
    assert rounded['a'] == pytest.approx(1.23)
    assert rounded['b'] == pytest.approx(2.34)

def test_parameters_repr_and_available_keys():
    """Test Parameters repr and available_keys methods."""
    params = Parameters(a=1, b=2)
    r = repr(params)
    assert "Parameters" in r
    keys = params.available_keys()
    assert set(keys) == {"a", "b"}

def test_parameters_getitem_priority():
    """Test Parameters __getitem__ priority."""
    params = Parameters(a=2, b=1)
    params.add_info(eps_ab=5)
    assert params['eps_ab'] == 5
    params.add_derived("eps_ab", lambda p: 1 - p['b']/p['a'])
    assert params['eps_ab'] == 0.5
    params['eps_ab'] = 0.4
    params['eps_ab'].ub = 1
    params['eps_ab'].lb = 0
    assert params['eps_ab'] == 0.4
    params.add_equal_constraints(eps_ab=lambda k: 0.25)
    assert params['eps_ab'] == 0.25
    params.decorate_func_constraints(lambda p: 1 - p['b']/p['a'])
    params.del_equal_constraints("eps_ab")
    assert params['eps_ab'] == 0.4
    assert params['eps_ab'].lb == 0
    assert params['eps_ab'].ub == 1

# -------------------------------
# Parameters constraints and derived values tests
# -------------------------------

def test_parameters_equal_constraints():
    """Test Parameters equal constraints functionality."""
    params = Parameters(a=2, b=3, c=6)
    params.add_equal_constraints(c=lambda p: p['a'] * p['b'])
    assert params['c'] == 6
    with pytest.raises(ValueError):
        params.add_equal_constraints(d=lambda p: 1)
    with pytest.raises(ValueError):
        params.del_equal_constraints('d')
    params.del_equal_constraints('c')
    assert params['c'] == 6

def test_parameters_decorate_func_constraints():
    """Test Parameters decorate_func_constraints functionality."""
    params = Parameters(a=2, b=4, c=8)
    params.add_equal_constraints(c=lambda p: p['a'] * p['b'])
    def func(p):
        return sum(p)
    wrapped = params.decorate_func_constraints(func)
    result = wrapped([2, 4])
    assert result == 2 + 4 + 8

def test_parameters_derived_and_info():
    """Test Parameters derived values and info functionality."""
    params = Parameters(a=2, b=1)
    params.add_derived("eps_ab", lambda p: 1 - p['b']/p['a'])
    assert params['eps_ab'] == 0.5
    params.add_info(test_info=42)
    assert params['test_info'] == 42
    with pytest.raises(KeyError):
        params.get_derived("not_exist")
    with pytest.raises(KeyError):
        params.get_info("not_exist")

def test_parameters_with_derived_classmethod():
    """Test Parameters with_derived class method."""
    derived = {"eps_ab": lambda p: 1 - p['b']/p['a']}
    params = Parameters.with_derived(derived, a=2, b=1)
    assert params['eps_ab'] == 0.5

# -------------------------------
# Parameters other utility methods tests
# -------------------------------

def test_parameters_get_rounded():
    """Test Parameters get_rounded method various branches."""
    pd = Parameters(a=1.12345, b=2.98765)
    rounded = pd.get_rounded(n=2)
    assert rounded['a'] == pytest.approx(1.12)
    assert rounded['b'] == pytest.approx(2.98)
    rounded_val = pd.get_rounded(n=2, only_value=True)
    assert isinstance(rounded_val, dict)
    with pytest.raises(ValueError):
        pd.get_rounded(-1)

def test_parameters_scipy_bounds_and_values_list():
    """Test Parameters scipy_bounds and values_list methods."""
    pd = Parameters(a=1, b=2)
    bounds = pd.scipy_bounds
    assert np.allclose(bounds.lb, [-np.inf, -np.inf])
    assert np.allclose(bounds.ub, [np.inf, np.inf])
    vals = pd.values_list()
    assert vals == [1., 2.]