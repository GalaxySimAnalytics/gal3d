import importlib

import numpy as np
import pytest

from gal3d.configuration import config
from gal3d.shape.geometry_plugins import Ellipsoid, Ellipsoid_S
import gal3d.shape.geometry_plugins as geometry_plugins
import gal3d.shape.geometry_plugins.ellipsoid as ellipsoid_module
import gal3d.shape.geometry_plugins.ellipsoid_s as ellipsoid_s_module


def surface_points(a=2.0, eps_ab=0.25, eps_bc=1.0 / 3.0):
    b = a * (1.0 - eps_ab)
    c = b * (1.0 - eps_bc)
    return np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])


class TestGeometryPluginPackage:
    def test_exports_and_reload(self):
        assert geometry_plugins.__all__ == ["Ellipsoid", "Ellipsoid_S"]
        assert geometry_plugins.Ellipsoid is Ellipsoid
        assert geometry_plugins.Ellipsoid_S is Ellipsoid_S
        reloaded = importlib.reload(geometry_plugins)
        assert reloaded.Ellipsoid is Ellipsoid


class TestEllipsoidPlugin:
    def test_geometry_methods_quick_methods_and_latex(self):
        geom = Ellipsoid(a=2.0, eps_ab=0.25, eps_bc=1.0 / 3.0)
        points = surface_points()

        np.testing.assert_allclose(geom(points), np.ones(3), atol=1e-12)
        assert len(geom.jacobian(points)) == 6

        ray_points, ray_dist, ray_radius = geom.ray_intersect(points * 2.0)
        np.testing.assert_allclose(ray_points, points, atol=1e-12)
        assert ray_dist.shape == (3,)
        assert ray_radius.shape == (3,)
        np.testing.assert_allclose(geom.f_ray_d(points), np.ones(3), atol=1e-12)
        assert geom.area_factor(points).shape == (3,)

        line_values = geom.line_intersect([[-3.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]])
        assert line_values.shape == (1, 2)
        assert np.any(line_values > 0.0)

        quick_value, quick_radius = Ellipsoid.quick_call(2.0, 0.25, 1.0 / 3.0, points)
        np.testing.assert_allclose(quick_value, np.ones(3), atol=1e-12)
        assert quick_radius.shape == (3,)
        np.testing.assert_allclose(Ellipsoid.quick_f_ray_d(2.0, 0.25, 1.0 / 3.0, points)[0], np.ones(3), atol=1e-12)
        assert Ellipsoid.quick_ray_dist(2.0, 0.25, 1.0 / 3.0, points)[0].shape == (3,)
        assert Ellipsoid.quick_area_factor(2.0, 0.25, 1.0 / 3.0, points).shape == (3,)
        line_start = np.array([[-3.0, 0.0, 0.0]])
        line_end = np.array([[3.0, 0.0, 0.0]])
        assert Ellipsoid.quick_line_intersect(2.0, 0.25, 1.0 / 3.0, line_start, line_end).shape == (1, 2)
        assert len(Ellipsoid.quick_jacobian(2.0, 1.5, 1.0, points)) == 6

        assert Ellipsoid.default_parameters()["a"] == 3.0
        estimated = Ellipsoid.estimate_parameters(points)
        assert estimated["a"] == pytest.approx(1.8)
        assert "epsilon" in Ellipsoid.PNlatex()["eps_ab"]
        assert "frac" in geom._latex_equation
        assert "epsilon" in geom._latex_other

    def test_derived_parameters_and_fallback_branches(self):
        geom = Ellipsoid(a=2.0, eps_ab=0.25, eps_bc=1.0 / 3.0)
        geom.parameters["eps_ab"].err = 0.01
        geom.parameters["eps_bc"].err = 0.02
        assert geom.parameters["b"] == pytest.approx(1.5)
        assert geom.parameters["c"] == pytest.approx(1.0)
        assert geom.parameters["eps_ac"] == pytest.approx(0.5)
        assert geom.parameters["eps_ac_s"] == pytest.approx(0.5)
        assert geom.parameters["T"] == pytest.approx((4.0 - 2.25) / (4.0 - 1.0))
        assert geom.parameters["eps_ac_err"] > 0.0
        assert geom.parameters["T_err"] > 0.0

        assert ellipsoid_module.b({"c": 1.0, "eps_bc": 0.5}) == pytest.approx(2.0)
        assert ellipsoid_module.c({"a": 2.0, "eps_ac": 0.5}) == pytest.approx(1.0)
        assert ellipsoid_module.a({"c": 1.0, "eps_ac": 0.5}) == pytest.approx(2.0)


class TestEllipsoidSPlugin:
    def test_shaped_geometry_methods_and_quick_methods(self):
        geom = Ellipsoid_S(a=2.0, eps_ab=0.25, eps_bc=1.0 / 3.0, sa=1.0, sb=1.0, sc=1.0)
        points = surface_points()

        np.testing.assert_allclose(geom(points), np.ones(3), atol=1e-10)
        assert len(geom.jacobian(points)) == 9
        ray_points, ray_dist, ray_radius = geom.ray_intersect(points * 2.0)
        np.testing.assert_allclose(ray_points, points, atol=1e-10)
        assert ray_dist.shape == (3,)
        assert ray_radius.shape == (3,)
        np.testing.assert_allclose(geom.f_ray_d(points), np.ones(3), atol=1e-10)
        assert geom.area_factor(points).shape == (3,)
        assert geom.line_intersect([[-3.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]).shape == (1, 2)

        quick_value, quick_radius = Ellipsoid_S.quick_call(2.0, 0.25, 1.0 / 3.0, 1.0, 1.0, 1.0, points)
        np.testing.assert_allclose(quick_value, np.ones(3), atol=1e-10)
        assert quick_radius.shape == (3,)
        np.testing.assert_allclose(Ellipsoid_S.quick_f_ray_d(2.0, 0.25, 1.0 / 3.0, 1.0, 1.0, 1.0, points)[0], np.ones(3), atol=1e-10)
        assert Ellipsoid_S.quick_ray_dist(2.0, 0.25, 1.0 / 3.0, 1.0, 1.0, 1.0, points)[0].shape == (3,)
        assert Ellipsoid_S.quick_area_factor(2.0, 0.25, 1.0 / 3.0, 1.0, 1.0, 1.0, points).shape == (3,)
        line_start = np.array([[-3.0, 0.0, 0.0]])
        line_end = np.array([[3.0, 0.0, 0.0]])
        assert Ellipsoid_S.quick_line_intersect(2.0, 0.25, 1.0 / 3.0, 1.0, 1.0, 1.0, line_start, line_end).shape == (1, 2)
        assert len(Ellipsoid_S.quick_jacobian(2.0, 1.5, 1.0, 1.0, 1.0, 1.0, points)) == 9

        assert Ellipsoid_S.default_parameters()["sa"] == 1.0
        estimated = Ellipsoid_S.estimate_parameters(points)
        assert estimated["sa"] == 1.0
        assert estimated["a"] == pytest.approx(1.8)
        assert geom._name == "SuperEllipsoid"
        assert "S_a" in Ellipsoid_S.PNlatex()["sa"]
        assert "frac" in geom._latex_equation
        assert "epsilon" in geom._latex_other

    def test_derived_parameters_cache_and_fallback_branches(self):
        old_n = config.ellipsoid_s.EpsTableN
        ellipsoid_s_module.reset_I_interp_cache()
        config.ellipsoid_s.EpsTableN = 3
        try:
            geom = Ellipsoid_S(a=2.0, eps_ab=0.25, eps_bc=1.0 / 3.0, sa=1.0, sb=1.0, sc=1.0)
            geom.parameters["eps_ab"].err = 0.01
            geom.parameters["eps_bc"].err = 0.02
            assert geom.parameters["b"] == pytest.approx(1.5)
            assert geom.parameters["c"] == pytest.approx(1.0)
            assert geom.parameters["eps_ac"] == pytest.approx(0.5)
            assert geom.parameters["T"] == pytest.approx((4.0 - 2.25) / (4.0 - 1.0))
            assert geom.parameters["eps_ac_err"] > 0.0
            assert geom.parameters["T_err"] > 0.0
            assert 0.0 <= geom.parameters["eps_ac_s"] <= 1.0

            sa_grid, sc_grid, values, interpolator = ellipsoid_s_module._build_I_table(2)
            assert sa_grid.shape == (2,)
            assert sc_grid.shape == (2,)
            assert values.shape == (2, 2)
            assert float(interpolator([[1.0, 1.0]])[0]) > 0.0
            assert ellipsoid_s_module._get_I_interp() is ellipsoid_s_module._I_interp_for_n(3)
        finally:
            config.ellipsoid_s.EpsTableN = old_n
            ellipsoid_s_module.reset_I_interp_cache()

        assert ellipsoid_s_module.b({"c": 1.0, "eps_bc": 0.5}) == pytest.approx(2.0)
        assert ellipsoid_s_module.c({"a": 2.0, "eps_ac": 0.5}) == pytest.approx(1.0)
        assert ellipsoid_s_module.a({"c": 1.0, "eps_ac": 0.5}) == pytest.approx(2.0)