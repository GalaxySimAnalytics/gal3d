"""Tests for the Parameter and Parameters classes in gal3d.optimization.parameter."""



import numpy as np
import pytest

from gal3d.optimization.parameter import (
    Parameter,
    Parameters,
    _escape_name,
    _fmt_bounds,
    _fmt_num,
)



class TestParameter:
    def test_bounds_and_hash(self):
        """Test Parameter's bounds properties and hash consistency"""
        p1 = Parameter(1.0, lb=0.0, ub=2.0)
        p2 = Parameter(1.0, lb=0.0, ub=2.0)
        assert hash(p1) == hash(p2)
        assert p1.lb == 0.0
        assert p1.ub == 2.0
    
    def test_value_and_bounds_assignment(self):
        """Test Parameter value and bounds assignment."""
        p = Parameter(5.0)
        p2 = p.assign_value(10.0)
        assert p2 == 10.0
        assert p2.lb == p.lb
        assert p2.ub == p.ub

        p.assign_bounds(-1.0, 1.0)
        assert p.lb == -1.0
        assert p.ub == 1.0
    
    def test_single_instance_behavior(self):
        """Test Parameter instance attributes and assignment."""
        param = Parameter(3, lb="0.1")
        hash(param)
        assert isinstance(param, float)
        assert param == 3
        assert param.lb == 0.1
        assert param.ub == np.inf

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
        
    def test_init_from_parameter_inherits_bounds(self):
        """Parameter constructed from Parameter inherits bounds unless overridden."""
        base = Parameter(2.5, lb=-1.0, ub=3.0, err=0.2)
        cloned = Parameter(base)
        assert cloned == 2.5
        assert cloned.lb == -1.0
        assert cloned.ub == 3.0
        assert cloned.err == 0.2
        # Override lb/ub on clone
        cloned2 = Parameter(base, lb=-2.0, ub=4.0)
        assert cloned2.lb == -2.0
        assert cloned2.ub == 4.0
        assert cloned2.err == 0.2  # err preserved if not overridden


    def test_number_bounds_and_latex_formatting(self):
        """Test number formatting and bounds formatting."""
        assert _fmt_num(np.nan) == "NaN"
        assert _fmt_num(np.nan, latex=True) == r"\mathrm{NaN}"
        assert _fmt_num(np.inf) == "inf"
        assert _fmt_num(-np.inf, latex=True) == r"-\infty"
        assert _fmt_num(12345.0, nd=3) == "1.234e+04"
        assert _fmt_num(0.00001, nd=3) == "1.000e-05"
        assert _fmt_num(1.23456, nd=2) == "1.23"
        assert _fmt_bounds(-1, np.inf, latex=True) == r"[-1.000, \infty]"
        assert _escape_name("eps_ab") == r"eps\_ab"

    def test_repr_hash_assignment_and_latex(self):
        """Test Parameter representation, hash, assignment, and LaTeX formatting."""
        param = Parameter("2.5", lb="0", ub="5", err="0.25")
        assert isinstance(param, float)
        assert param.lb == 0.0
        assert param.ub == 5.0
        assert param.err == 0.25
        assert "Parameter" in repr(param)
        assert r"\pm" in param.to_latex()
        assert r"\pm" in param.to_latex(hide_zero_err=False)
        assert param._repr_latex_().startswith("$")

        param.lb = -1
        param.ub = 10
        param.err = 0
        assert (param.lb, param.ub, param.err) == (-1.0, 10.0, 0.0)
        assert r"\pm" not in param.to_latex()
        assert "in" not in repr(param).lower() or "Parameter" in repr(param)


class TestParameters:
    
    def test_basic_method(self):
        """Test Parameters basic usage and properties."""
        params = Parameters(a=2, b=3)
        # Test that parameters are Parameter instances with bounds
        assert isinstance(params['a'], Parameter)
        assert isinstance(params['a'].lb, float)
        
        params.set_value(5, 6)
        assert params['a'] == 5
        assert params['b'] == 6
        params.set_value(a=7)
        assert params['a'] == 7
        with pytest.raises(KeyError):
            params.set_value(c=1)

        params.add_derived("c", lambda p: p['a'] + p['b'])
        assert params['c'] == 13
        params['a'] = 10
        assert params['c'] == 16

        assert set(params.parameter_keys()) == {"a", "b"}
        assert set(params.keys()) == {"a", "b"}
        assert set(params.available_keys()) == {"a", "b", "c"}
        params.scipy_bounds
        params.value
        params.ub
        params.lb
        repr(params)
        assert params.to_latex_lines()["a"]
        params._repr_latex_()
        params._ipython_key_completions_()
        assert params.values_list() == [10., 6.]
        assert isinstance(params.structure_parameters, dict)

        # Test that accessing a non-existent parameter raises KeyError
        with pytest.raises(KeyError):
            params.get_parameter("c")

    def test_rounded_values(self):
        """Test Parameters structure_parameters and get_rounded_values_dict methods."""
        params = Parameters(a=1.2345, b=2.3456)
        rounded = params.get_rounded_values_dict(n=2)
        assert rounded['a'] == pytest.approx(1.23)
        assert rounded['b'] == pytest.approx(2.34)

        # Test that invalid rounding raises ValueError
        with pytest.raises(ValueError):
            params.get_rounded(-1)

    def test_update_and_copy_behavior(self):
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

    def test_set_value_and_bounds(self):
        params = Parameters(a=2, b=3)

        # Test setting bounds
        params.set_lb(b=4)
        params.set_ub(a=5)
        assert params['b'].lb == 4
        assert params['a'].ub == 5

        
        # Test setting values and that bounds are preserved
        params['a'] = 6
        assert params['a'] == 6
        assert params['a'].ub == 5

        # Test set_value method
        params.set_value(a=3, b=7)
        assert params['a'] == 3
        assert params['b'] == 7
        assert params['b'].lb == 4
        assert params['a'].ub == 5

    def test_new_and_add(self):
        """Test Parameters new and __add__ methods."""
        params = Parameters(a=1, b=2)
        new_params = params.new(a=3, b=4)
        assert new_params['a'] == 3
        assert new_params['b'] == 4
        params2 = Parameters(a=5)
        # Test that addition creates a new instance with updated values
        params3 = params + params2
        assert params3['a'] == 5
        assert params3['b'] == 2

    def test_getitem_priority(self):
        """Test Parameters __getitem__ priority."""
        params = Parameters(a=2, b=1)
        
        # Add info parameter
        params.add_info(eps_ab=5)
        assert params['eps_ab'] == 5
        
        # Add derived parameter, which should take precedence over info
        params.add_derived("eps_ab", lambda p: 1 - p['b']/p['a'])
        assert params['eps_ab'] == 0.5
        
        # add parameter with same name, which should take precedence over derived and info
        params['eps_ab'] = 0.4
        params['eps_ab'].ub = 1
        params['eps_ab'].lb = 0
        assert params['eps_ab'] == 0.4
        
        # Test that equal constraint can be added and takes precedence over parameter, derived, and info
        params.add_equal_constraints(eps_ab=lambda p: 0.25)
        assert params['eps_ab'] == 0.25
         
        wrapped = params.decorate_func_constraints(lambda values: values)
        assert wrapped([1, 3]) == [1, 3, 0.25]  # constraint value should be included in decorated function output
        
        # Test that equal constraint can be removed and parameter value is restored
        params.del_equal_constraints("eps_ab")
        assert params['eps_ab'] == 0.4
        assert params['eps_ab'].lb == 0
        assert params['eps_ab'].ub == 1

        with pytest.raises(KeyError):
            params["missing"]


    # -------------------------------
    # Parameters constraints and derived values tests
    # -------------------------------

    def test_equal_constraints(self):
        """Test Parameters equal constraints functionality."""
        params = Parameters(a=2, b=3, c=6)
        
        with pytest.raises(ValueError, match="must be callable"):
            params.add_equal_constraints(a=3)
        with pytest.raises(ValueError, match="original parameter"):
            params.add_equal_constraints(h=lambda p: 1)
        with pytest.raises(ValueError, match="not found"):
            params.del_equal_constraints("e")
        with pytest.raises(ValueError, match="No constraints"):
            params.del_equal_constraints("a")
        with pytest.raises(KeyError):
            params.get_constraint("a")
        
        
        params.add_equal_constraints(c=lambda p: p['a'] * p['b'])
        assert params['c'] == 6
        with pytest.raises(ValueError):
            params.add_equal_constraints(d=lambda p: 1)
        with pytest.raises(ValueError):
            params.del_equal_constraints('d')
        params.del_equal_constraints('c')
        assert params['c'] == 6

        with pytest.raises(ValueError, match="No constraints"):
            params.del_equal_constraints("a")
        
        assert params.all_parameter_names == ["a", "b", "c"]
        
        params.fix_parameters(b=7)
        assert params["b"] == 7

    def test_decorate_func_constraints(self):
        """Test Parameters decorate_func_constraints functionality."""
        params = Parameters(a=2, b=4, c=8)
        params.add_equal_constraints(c=lambda p: p['a'] * p['b'])
        def func(p):
            return sum(p)
        wrapped = params.decorate_func_constraints(func)
        result = wrapped([2, 4])
        assert result == 2 + 4 + 8

    def test_derived_and_info(self):
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
        with pytest.raises(TypeError):
            params.add_derived("bad")

    def test_with_derived_classmethod(self):
        """Test Parameters with_derived class method."""
        derived = {"eps_ab": lambda p: 1 - p['b']/p['a']}
        params = Parameters.with_derived(derived, a=2, b=1)
        assert params['eps_ab'] == 0.5


    # -------------------------------
    # Parameters other utility methods tests
    # -------------------------------
            
    def test_clip_to_bounds_and_rounding(self):
        """clip_to_bounds clamps values; get_rounded can round value and bounds."""
        pd = Parameters(a=Parameter(-5.0, lb=-1.5, ub=10.0),
                        b=Parameter(20.0, lb=-2.0, ub=3.25))
        pd.clip_to_bounds()
        assert pd['a'] == pytest.approx(-1.5)
        assert pd['b'] == pytest.approx(3.25)
        # Round values only
        r = pd.get_rounded(n=2)
        assert isinstance(r, Parameters.__mro__[-2]) or isinstance(r, dict)  # type fallback
        assert r['a'] == pytest.approx(-1.5)
        # Round bounds too
        pd2 = Parameters(x=Parameter(1.23456, lb=-1.23456, ub=3.45678))
        r2 = pd2.get_rounded(n=3, round_value=True, round_bound=True)
        assert r2['x'] == pytest.approx(1.234)
        assert r2['x'].lb == pytest.approx(-1.234)
        assert r2['x'].ub == pytest.approx(3.456)
        # only_value returns plain dict
        rv = pd2.get_rounded(n=2, only_value=True)
        assert isinstance(rv, dict)
        assert rv['x'] == pytest.approx(1.23)
        with pytest.raises(ValueError):
            pd2.get_rounded(-1)

    def test_scipy_bounds_and_values_list(self):
        """Test Parameters scipy_bounds and values_list methods."""
        pd = Parameters(a=1, b=2)
        bounds = pd.scipy_bounds
        assert np.allclose(bounds.lb, [-np.inf, -np.inf])
        assert np.allclose(bounds.ub, [np.inf, np.inf])
        vals = pd.values_list()
        assert vals == [1., 2.]