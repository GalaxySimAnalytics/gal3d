import importlib

import numpy as np
import pytest

from gal3d.shape.coordinate_plugins import EulerShift, RotateOnly, ShiftEuler, ShiftOnly
from gal3d.shape.coordinate_plugins._rotation_eular_util import EulerAngles
import gal3d.shape.coordinate_plugins as coordinate_plugins



class TestCoordinatePluginPackage:
    def test_package_exports_coordinate_plugins(self):
        assert coordinate_plugins.__all__ == ["EulerShift", "ShiftEuler", "ShiftOnly", "RotateOnly"]
        assert coordinate_plugins.EulerShift is EulerShift
        assert coordinate_plugins.RotateOnly is RotateOnly
        reloaded = importlib.reload(coordinate_plugins)
        assert reloaded.ShiftOnly is ShiftOnly


class TestEulerAngles:
    def test_derivative_matrices_and_jacobian_shapes(self):
        rotation = EulerAngles.from_euler("zyx", [0.1, 0.2, 0.3])
        matrices = rotation.d_euler("zyx")
        assert len(matrices) == 3
        assert all(matrix.shape == (3, 3) for matrix in matrices)

        jacobian = rotation.jacobian_euler(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), seq="zyx")
        assert len(jacobian) == 3
        assert all(term.shape == (2, 3) for term in jacobian)


class TestShiftOnly:
    def test_shift_roundtrip_quick_methods_jacobian_and_metadata(self):
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        coord = ShiftOnly(x=0.5, y=-1.0, z=2.0)
        transformed = coord(pos)
        np.testing.assert_allclose(coord.inverse(transformed), pos)
        np.testing.assert_allclose(ShiftOnly.quick_call(0.5, -1.0, 2.0, pos), transformed)
        np.testing.assert_allclose(ShiftOnly.quick_inverse(0.5, -1.0, 2.0, transformed), pos)

        jacobian = coord.jacobian(pos)
        quick_jacobian = ShiftOnly.quick_jacobian(0.5, -1.0, 2.0, pos)
        assert len(jacobian) == 3
        assert jacobian[0].tolist() == [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        np.testing.assert_allclose(quick_jacobian[1], jacobian[1])
        assert ShiftOnly.default_parameters()["x"] == 0.0
        assert ShiftOnly.estimate_parameters(pos) == {"x": 2.5, "y": 3.5, "z": 4.5}
        assert ShiftOnly.PNlatex()["x"] == "x_c"
        assert "mathbf" in coord._latex_equation
        assert "begin" in coord._latex_other


class TestRotateOnly:
    def test_rotation_roundtrip_quick_methods_estimation_and_latex(self):
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coord = RotateOnly(ang1=0.2, ang2=-0.1, ang3=0.3)
        transformed = coord(pos)
        np.testing.assert_allclose(coord.inverse(transformed), pos, atol=1e-12)
        np.testing.assert_allclose(RotateOnly.quick_call(0.2, -0.1, 0.3, pos), transformed)
        np.testing.assert_allclose(RotateOnly.quick_inverse(0.2, -0.1, 0.3, transformed), pos, atol=1e-12)

        jacobian = coord.jacobian(pos)
        quick_jacobian = RotateOnly.quick_jacobian(0.2, -0.1, 0.3, pos)
        assert len(jacobian) == 3
        assert all(term.shape == (2, 3) for term in jacobian)
        np.testing.assert_allclose(quick_jacobian[2], jacobian[2])
        assert RotateOnly.default_parameters()["ang1"] == 0.0
        assert set(RotateOnly.estimate_parameters(np.eye(3))) == {"ang1", "ang2", "ang3"}
        np.testing.assert_allclose(RotateOnly.mat_to_angle(np.eye(3)), np.zeros(3), atol=1e-12)
        assert RotateOnly.PNlatex()["ang1"] == r"\alpha"
        assert "R_{zyx}" in coord._latex_equation
        assert "R_{z}" in coord._latex_other

    def test_estimate_fallback_and_derived_rotation_values(self, monkeypatch):
        monkeypatch.setattr(EulerAngles, "from_matrix", classmethod(lambda cls, matrix: (_ for _ in ()).throw(ValueError("bad matrix"))))
        assert RotateOnly.estimate_parameters(np.eye(3)) == {"ang1": 0.0, "ang2": 0.0, "ang3": 0.0}

        coord = RotateOnly(ang1=0.2, ang2=-0.1, ang3=0.3)
        coord.parameters["ang1"].err = 0.01
        coord.parameters["ang2"].err = 0.02
        coord.parameters["ang3"].err = 0.03
        assert coord.parameters["angle"].tolist() == [0.2, -0.1, 0.3]
        assert coord.parameters["rot_matrix"].shape == (3, 3)
        assert 0.0 <= coord.parameters["x_axis_angle"] <= np.pi / 2
        assert 0.0 <= coord.parameters["z_axis_angle"] <= np.pi / 2
        assert coord.parameters["x_axis_angle_err"] >= 0.0
        assert coord.parameters["z_axis_angle_err"] >= 0.0


class TestEulerShiftAndShiftEuler:
    def test_euler_shift_roundtrip_quick_methods_estimation_and_metadata(self):
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        coord = EulerShift(x=0.5, y=-0.25, z=0.75, ang1=0.1, ang2=0.2, ang3=-0.1)
        transformed = coord(pos)
        np.testing.assert_allclose(coord.inverse(transformed), pos, atol=1e-12)
        np.testing.assert_allclose(EulerShift.quick_call(0.5, -0.25, 0.75, 0.1, 0.2, -0.1, pos), transformed)
        np.testing.assert_allclose(EulerShift.quick_inverse(0.5, -0.25, 0.75, 0.1, 0.2, -0.1, transformed), pos, atol=1e-12)

        jacobian = coord.jacobian(pos)
        quick_jacobian = EulerShift.quick_jacobian(0.5, -0.25, 0.75, 0.1, 0.2, -0.1, pos)
        assert len(jacobian) == 6
        assert all(term.shape == (3, 3) for term in jacobian)
        np.testing.assert_allclose(quick_jacobian[5], jacobian[5])
        assert EulerShift.default_parameters()["ang3"] == 0.0
        assert set(EulerShift.estimate_parameters(pos)) == {"x", "y", "z", "ang1", "ang2", "ang3"}
        np.testing.assert_allclose(EulerShift.mat_to_angle(np.eye(3)), np.zeros(3), atol=1e-12)
        assert EulerShift.PNlatex()["ang2"] == r"\beta"
        assert coord.parameters["pos"].tolist() == [0.5, -0.25, 0.75]
        assert coord.parameters["angle"].tolist() == [0.1, 0.2, -0.1]
        assert coord.parameters["rot_matrix"].shape == (3, 3)
        assert coord.parameters["x_axis_angle"] >= 0.0
        assert coord.parameters["z_axis_angle"] >= 0.0
        assert "mathbf" in coord._latex_equation
        assert "begin" in coord._latex_other

    def test_euler_shift_estimate_fallback(self, monkeypatch):
        monkeypatch.setattr(EulerAngles, "from_matrix", classmethod(lambda cls, matrix: (_ for _ in ()).throw(ValueError("bad matrix"))))
        estimated = EulerShift.estimate_parameters([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        assert estimated == {"x": 2.0, "y": 3.0, "z": 4.0, "ang1": 0.0, "ang2": 0.0, "ang3": 0.0}

    def test_zero_shift_branch_and_shift_euler_roundtrip(self):
        pos = np.array([[1.0, 2.0, 3.0]])
        zero_shift = EulerShift(x=0.0, y=0.0, z=0.0, ang1=0.0, ang2=0.0, ang3=0.0)
        np.testing.assert_allclose(zero_shift(pos), pos)

        zero_shift_euler = ShiftEuler(x=0.0, y=0.0, z=0.0, ang1=0.0, ang2=0.0, ang3=0.0)
        np.testing.assert_allclose(zero_shift_euler(pos), pos)

        coord = ShiftEuler(x=0.5, y=-0.25, z=0.75, ang1=0.1, ang2=0.2, ang3=-0.1)
        transformed = coord(pos)
        np.testing.assert_allclose(coord.inverse(transformed), pos, atol=1e-12)
        assert coord.parameters["angle"].tolist() == [0.1, 0.2, -0.1]
        assert coord.parameters["x_axis_angle_err"] == 0.0
        assert coord.parameters["z_axis_angle_err"] == 0.0
        assert "R_{zyx}" in coord._latex_equation