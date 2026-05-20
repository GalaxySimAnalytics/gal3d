import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import gal3d.shape as shape_module
from gal3d.optimization.parameter import Parameters
from gal3d.shape import Structure3D, StructureCore, StructureError, _azimuthal_fourier_residual
from gal3d.shape.coordinate import Coordinate, CoordinateBase
from gal3d.shape.geometry import Geometry, GeometryBase


def _as_pos(pos):
    arr = np.asarray(pos, dtype=np.float64)
    return arr.reshape(1, 3) if arr.ndim == 1 else arr


@pytest.fixture
def fake_shape_plugins():
    saved_coordinates = Coordinate._plugins.copy()
    saved_geometries = Geometry._plugins.copy()
    saved_methods = StructureError._compute_error_method.copy()

    class FakeShiftCoordinate(CoordinateBase):
        PN = ("shift",)
        LB = {"shift": -10.0}
        UB = {"shift": 10.0}

        @classmethod
        def default_parameters(cls):
            return cls.create_parameters(shift=0.0)

        @classmethod
        def estimate_parameters(cls, pos):
            return cls.create_parameters(shift=float(np.mean(_as_pos(pos)[:, 0])))

        def __call__(self, pos):
            return _as_pos(pos) + self["shift"]

        def inverse(self, pos):
            return _as_pos(pos) - self["shift"]

        def jacobian(self, pos):
            return (np.ones_like(_as_pos(pos)),)

        @staticmethod
        def quick_call(*, shift, pos):
            return _as_pos(pos) + shift

        @staticmethod
        def quick_inverse(*, shift, pos):
            return _as_pos(pos) - shift

        @staticmethod
        def quick_jacobian(*, shift, pos):
            return (np.ones_like(_as_pos(pos)),)

    @FakeShiftCoordinate.derived
    def doubled_shift(params):
        return 2.0 * params["shift"]

    class FakeRadiusGeometry(GeometryBase):
        PN = ("scale",)
        LB = {"scale": 0.1}
        UB = {"scale": 10.0}

        @classmethod
        def default_parameters(cls):
            return cls.create_parameters(scale=1.0)

        @classmethod
        def estimate_parameters(cls, pos):
            radius = np.linalg.norm(_as_pos(pos), axis=1)
            return cls.create_parameters(scale=float(np.max(radius)))

        def __call__(self, pos):
            radius = np.linalg.norm(_as_pos(pos), axis=1)
            return radius / self["scale"]

        def jacobian(self, pos):
            radius = np.linalg.norm(_as_pos(pos), axis=1)
            return (-radius / self["scale"] ** 2,)

        def ray_intersect(self, pos):
            pos = _as_pos(pos)
            radius = np.linalg.norm(pos, axis=1)
            safe_radius = np.where(radius == 0.0, 1.0, radius)
            points = pos / safe_radius[:, None] * self["scale"]
            return points, radius - self["scale"], radius

        def line_intersect(self, pos1, pos2):
            return np.tile(np.array([[0.25, 0.75]]), (len(_as_pos(pos1)), 1))

        def f_ray_d(self, pos):
            return self(pos)

        def area_factor(self, pos):
            return np.full(len(_as_pos(pos)), self["scale"])

        @staticmethod
        def quick_call(*, scale, pos):
            radius = np.linalg.norm(_as_pos(pos), axis=1)
            return radius / scale, radius

        @staticmethod
        def quick_f_ray_d(*, scale, pos):
            return FakeRadiusGeometry.quick_call(scale=scale, pos=pos)

        @staticmethod
        def quick_ray_dist(*, scale, pos):
            radius = np.linalg.norm(_as_pos(pos), axis=1)
            return np.abs(radius - scale), radius

        @staticmethod
        def quick_area_factor(*, scale, pos):
            return np.full(len(_as_pos(pos)), scale)

        @staticmethod
        def quick_line_intersect(*, scale, pos1, pos2):
            return np.tile(np.array([[0.2, 0.8]]), (len(_as_pos(pos1)), 1))

        @staticmethod
        def quick_jacobian(*, scale, pos):
            radius = np.linalg.norm(_as_pos(pos), axis=1)
            return (-radius / scale ** 2,)

    @FakeRadiusGeometry.derived
    def doubled_scale(params):
        return 2.0 * params["scale"]

    try:
        yield FakeShiftCoordinate, FakeRadiusGeometry
    finally:
        Coordinate._plugins = saved_coordinates
        Geometry._plugins = saved_geometries
        StructureError._compute_error_method = saved_methods


def demo_error_func(f_call):
    return float(np.sum(np.asarray(f_call) ** 2))


def demo_rscale_error(f_call):
    return float(np.sum(np.asarray(f_call)))


class TestStructureCore:
    def test_options_parameters_copy_repr_latex_and_validation(self, fake_shape_plugins):
        coordinate_cls, geometry_cls = fake_shape_plugins

        core = StructureCore("FakeShiftCoordinate", "FakeRadiusGeometry")
        assert core.coordinate_name == "FakeShiftCoordinate"
        assert core.geometry_name == "FakeRadiusGeometry"
        assert "FakeShiftCoordinate" in StructureCore.available_options()["coordinate"]
        assert "FakeRadiusGeometry" in StructureCore.available_options()["geometry"]
        assert set(core.derived_param_funcs()) == {"doubled_shift", "doubled_scale"}
        assert "Coord" in repr(core)
        assert "Coordinate" in core._repr_latex_()

        by_class = StructureCore(coordinate_cls, geometry_cls)
        by_instance = StructureCore(coordinate_cls(shift=1.0), geometry_cls(scale=2.0))
        assert core.is_equal(by_class)
        assert core.is_equal(by_instance)
        assert not core.is_equal(object())

        defaults = core.create_parameters()
        assert defaults["shift"] == 0.0
        assert defaults["scale"] == 1.0
        from_args = core.create_parameters([2.0, 3.0])
        assert from_args["shift"] == 2.0
        assert from_args["scale"] == 3.0
        from_kwargs = core.create_parameters(shift=4.0, scale=5.0)
        assert from_kwargs["shift"] == 4.0
        assert from_kwargs["scale"] == 5.0

        core.set_parameters(shift=1.5, scale=2.5)
        assert core.parameters["shift"] == 1.5
        assert core.parameters["scale"] == 2.5
        clone = core.clone_with_parameters(shift=0.5, scale=2.5)
        assert clone.parameters["shift"] == 0.5
        assert core.parameters["shift"] == 1.5
        copied = core.copy()
        assert copied.is_equal(core)
        assert copied.parameters is not core.parameters

        assert core._split_quick_parameters([7.0, 8.0]) == ({"shift": 7.0}, {"scale": 8.0})
        assert core._split_quick_parameters(shift=2.0, scale=3.0) == ({"shift": 2.0}, {"scale": 3.0})
        with pytest.raises(KeyError):
            core._split_quick_parameters(shift=2.0)

        estimated = core.estimate_parameters(np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        assert estimated["shift"] == 2.0
        assert estimated["scale"] == pytest.approx(np.sqrt(33.0))

        with pytest.raises(TypeError):
            StructureCore(object(), geometry_cls)
        with pytest.raises(TypeError):
            StructureCore(coordinate_cls, object())

    def test_evaluation_intersections_and_generators(self, fake_shape_plugins):
        coordinate_cls, geometry_cls = fake_shape_plugins
        core = StructureCore(coordinate_cls, geometry_cls)
        core.set_parameters(shift=1.0, scale=2.0)
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        np.testing.assert_allclose(core.transform_pos(pos), pos + 1.0)
        np.testing.assert_allclose(core.inverse_transform(pos + 1.0), pos)
        assert core(pos).shape == (2,)
        assert core.f_ray_d(pos).shape == (2,)
        ray_pos, ray_dist = core.ray_intersect(pos)
        assert ray_pos.shape == (2, 3)
        assert ray_dist.shape == (2,)
        assert core.line_intersect(pos, pos + 1.0).shape == (2, 2)

        values, radius = core.quick_call(1.0, 2.0, pos=pos)
        assert values.shape == (2,)
        assert radius.shape == (2,)
        assert core.quick_f_ray_d(1.0, 2.0, pos=pos)[0].shape == (2,)
        assert core.quick_ray_dist(1.0, 2.0, pos=pos)[0].shape == (2,)
        assert core.quick_area_factor(1.0, 2.0, pos=pos).tolist() == [2.0, 2.0]
        assert core.quick_line_intersect(1.0, 2.0, pos1=pos, pos2=pos + 1.0).shape == (2, 2)

        points = core.generate_points(n_points=5)
        assert points.shape == (5, 3)
        with pytest.raises(ValueError, match="positive"):
            core.generate_points(n_points=0)

        x_slice, y_slice = core.generate_slice2D(n_bins=4)
        assert x_slice.shape == y_slice.shape == (4,)
        x_bins, y_bins = core.generate_slice2D(bins=np.linspace(0.0, np.pi, 3))
        assert x_bins.shape == y_bins.shape == (3,)
        with pytest.raises(ValueError, match="angle_bins"):
            core.generate_slice2D(bins=np.empty((1, 1)))
        with pytest.raises(ValueError, match="rotation"):
            core.generate_slice2D(rotation=np.eye(2))

        x_edge, y_edge = core.generate_edge2D(angle_bins=np.linspace(0.0, np.pi, 4), radius_bins=np.array([0.5, 1.0]))
        assert x_edge.shape == y_edge.shape == (4,)
        rotated_edge = core.generate_edge2D(angle_bins=np.linspace(0.0, np.pi, 3), radius_bins=np.array([1.0]), rotation=np.eye(3))
        assert rotated_edge[0].shape == (3,)
        with pytest.raises(ValueError, match="angle_bins"):
            core.generate_edge2D(angle_bins=np.empty((1, 1)))
        with pytest.raises(ValueError, match="radius_bins"):
            core.generate_edge2D(radius_bins=np.empty((1, 1)))
        with pytest.raises(ValueError, match="rotation"):
            core.generate_edge2D(rotation=np.eye(2))

        x3, y3, z3 = core.generate_edge3D(phi_bins=np.linspace(0.0, np.pi, 3), theta_bins=np.linspace(0.0, np.pi, 4))
        assert x3.shape == y3.shape == z3.shape == (3, 4)
        default_x3, default_y3, default_z3 = core.generate_edge3D(n_phi_bins=2, n_theta_bins=2)
        assert default_x3.shape == default_y3.shape == default_z3.shape == (5, 3)
        with pytest.raises(ValueError, match="phi_bins"):
            core.generate_edge3D(phi_bins=np.empty((1, 1)))
        with pytest.raises(ValueError, match="theta_bins"):
            core.generate_edge3D(theta_bins=np.empty((1, 1)))


class TestStructureError:
    def test_registry_copy_latex_and_validation(self, fake_shape_plugins):
        def demo_error_method(self, params, **kwargs):
            return self._error_func(np.asarray(params))

        StructureError.compute_method_registry("demo_error_method")(demo_error_method)
        error = StructureError(error_func=demo_error_func, error_method="demo_error_method")
        assert "demo_error_method" in StructureError.available_options()["error_method"]
        assert error._error_func_name == "demo_error_func"
        assert error._error_method_name == "demo_error_method"
        assert r"demo\_error\_func" in error._repr_latex_()

        copied = error.copy()
        copied.use_ln_error = True
        assert error.is_equal(copied)
        assert not error.is_equal(object())

        direct = StructureError(error_func=demo_error_func, error_method=demo_error_method)
        assert direct._error_method_name == "demo_error_method"

        @StructureError.compute_method_registry
        def directly_registered(self, params, **kwargs):
            return 0.0

        assert "directly_registered" in StructureError.available_options()["error_method"]
        with pytest.raises(TypeError, match="not callable"):
            StructureError.compute_method_registry("bad_method")(42)
        with pytest.raises(AssertionError):
            StructureError(error_func=42, error_method="demo_error_method")
        with pytest.raises(AssertionError):
            StructureError(error_func=demo_error_func, error_method=42)


class TestStructure3D:
    def test_calculate_error_registered_methods_copy_fit_and_latex(self, fake_shape_plugins):
        coordinate_cls, geometry_cls = fake_shape_plugins
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]])
        structure = Structure3D(coordinate_cls, geometry_cls, demo_error_func, "isodensity_fcall")
        structure.set_parameters(shift=0.0, scale=1.0)

        base_error = structure.calculate_error(pos)
        dict_error = structure.calculate_error(pos, params={"scale": 2.0})
        seq_error = structure.calculate_error(pos, params=[0.0, 2.0])
        assert base_error >= 0.0
        assert dict_error == seq_error
        assert "Error" in repr(structure)
        assert r"demo\_error\_func" in structure._repr_latex_()
        assert "error_func" in Structure3D.available_options()

        copied = structure.copy()
        assert copied.is_equal(structure)
        assert not structure.is_equal(StructureCore(coordinate_cls, geometry_cls))

        structure.use_ln_error = True
        assert structure.calculate_error(pos) >= 0.0

        for method_name in ["isodensity_dcall", "isodensity_dist"]:
            method_structure = Structure3D(coordinate_cls, geometry_cls, demo_error_func, method_name)
            assert method_structure.calculate_error(pos) >= 0.0

        for method_name in ["isodensity_curve_fcall", "isodensity_curve_dcall"]:
            curve_structure = Structure3D(coordinate_cls, geometry_cls, demo_rscale_error, method_name)
            residual = curve_structure.calculate_error(pos)
            assert residual.shape == (3,)

        azimuth_structure = Structure3D(coordinate_cls, geometry_cls, demo_rscale_error, "isodensity_curve_dcall_azimuthal")
        azimuth_residual = azimuth_structure.calculate_error(
            pos,
            azimuth_n_mu_bins=2,
            azimuth_min_count=1,
            azimuth_min_sin_theta=0.0,
            azimuth_m=1,
        )
        assert azimuth_residual.ndim == 1

        class FakeOptimizer:
            def fit(self, method, params, func_kwargs):
                assert isinstance(params, Parameters)
                method(params.values_list(), **func_kwargs)
                return OptimizeResult(params={"shift": 0.25, "scale": 1.25}, cost=0.0)

        result = structure.fit(pos, optimizer=FakeOptimizer(), estimate=True)
        assert result._structure is structure
        assert result._param_sets[0]["shift"] == 0.25
        assert result._param_sets[0]["scale"] == 1.25

    def test_azimuthal_fourier_residual_edge_cases_and_nonempty_case(self):
        pos = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        f = np.array([1.0, 1.1, 0.9, 1.2])
        r = np.ones(4)

        assert _azimuthal_fourier_residual(pos, f, r, azimuth_weight=0.0).size == 0
        assert _azimuthal_fourier_residual(pos, f, r, n_mu_bins=0).size == 0
        assert _azimuthal_fourier_residual(pos, f, r, azimuth_m=0).size == 0
        assert _azimuthal_fourier_residual(pos, f, r, min_sin_theta=2.0).size == 0
        assert _azimuthal_fourier_residual(pos, f, r, min_count=10).size == 0

        residual = _azimuthal_fourier_residual(pos, f, r, n_mu_bins=1, min_count=1, min_sin_theta=0.0, azimuth_m=1)
        assert residual.shape == (2,)
        assert np.all(np.isfinite(residual))

    def test_curve_azimuthal_returns_base_when_fourier_residual_is_empty(self, fake_shape_plugins, monkeypatch):
        coordinate_cls, geometry_cls = fake_shape_plugins
        structure = Structure3D(coordinate_cls, geometry_cls, demo_error_func, "isodensity_curve_dcall_azimuthal")
        monkeypatch.setattr(shape_module, "_azimuthal_fourier_residual", lambda *args, **kwargs: np.empty(0))
        residual = structure.calculate_error(np.array([[1.0, 0.0, 0.0]]))
        assert residual.shape == (1,)
