import numpy as np
import pytest

from gal3d.density import DensitySource
from gal3d.theoretical import (
    CombinedDensityDistribution,
    CoordinateTransform,
    DoubleExponentialDisk,
    FieldCoordinate,
    PlummerSphere,
    PowerLawSphere,
    THEORETICAL_MODELS,
    TheoreticalDensityDistribution,
)
from gal3d.visualization.show import show_image


class ShiftOnlyTransform(CoordinateTransform):
    def __init__(self, shift):
        self.shift = np.asarray(shift, dtype=float)

    def world_to_elliptical(self, pos_world):
        return np.asarray(pos_world, dtype=float) - self.shift


class ConstantDensity(TheoreticalDensityDistribution):
    def __init__(self, value=1.0, coordinate=None):
        super().__init__(coordinate)
        self.value = value

    def _evaluate_density_generic(self, pos_elliptical):
        return np.full(len(pos_elliptical), self.value, dtype=float)


def positions():
    return np.array(
        [
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0],
        ]
    )


class TestFieldCoordinate:
    def test_validation_rotation_and_world_to_elliptical(self):
        """Test that FieldCoordinate validates parameters, computes rotation matrices, and transforms coordinates correctly."""
        with pytest.raises(TypeError):
            CoordinateTransform()
        with pytest.raises(ValueError, match="center_pos"):
            FieldCoordinate(center_pos=[1.0, 2.0])
        with pytest.raises(ValueError, match="scales"):
            FieldCoordinate(scales=[1.0, 0.0, 1.0])

        rotation = FieldCoordinate.rotation_matrix_xyz((0.0, 0.0, np.pi / 2.0))
        np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)

        coordinate = FieldCoordinate(center_pos=[1.0, 2.0, 3.0], angles=(0.0, 0.0, 0.0), scales=[2.0, 4.0, 8.0])
        transformed = coordinate.world_to_elliptical(np.array([[3.0, 6.0, 11.0]]))
        np.testing.assert_allclose(transformed, [[1.0, 1.0, 1.0]])


class TestTheoreticalDensityBase:
    def test_registration_set_coordinate_evaluate_and_addition(self):
        """Test that TheoreticalDensityDistribution correctly registers subclasses, sets coordinates, evaluates density, and supports addition."""

        # Test subclass registration and availability
        assert issubclass(ConstantDensity, TheoreticalDensityDistribution)
        assert "ConstantDensity" in TheoreticalDensityDistribution.available_models()
        assert THEORETICAL_MODELS["ConstantDensity"] is ConstantDensity
        assert isinstance(ConstantDensity(), DensitySource)


        # Test set_coordinate method and evaluation
        model = ConstantDensity(2.5)
        assert model.set_coordinate(center_pos=[1.0, 0.0, 0.0], angles=(0.0, 0.0, 0.0), scales=[2.0, 1.0, 1.0]) is model
        np.testing.assert_allclose(model.coordinate.center_pos, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(model([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), [2.5, 2.5])

        # Test that set_coordinate validates coordinate type
        shifted = ConstantDensity(1.0, coordinate=ShiftOnlyTransform([1.0, 0.0, 0.0]))
        with pytest.raises(TypeError, match="FieldCoordinate"):
            shifted.set_coordinate(center_pos=[0.0, 0.0, 0.0])

        # Test addition of density distributions
        base = TheoreticalDensityDistribution()
        with pytest.raises(NotImplementedError, match="_evaluate_density_generic"):
            base._evaluate_density_generic(np.zeros((1, 3)))

        combo = ConstantDensity(1.0) + ConstantDensity(2.0)
        assert isinstance(combo, CombinedDensityDistribution)
        np.testing.assert_allclose(combo(positions()), np.full(4, 3.0))
        combo2 = ConstantDensity(3.0) + combo
        np.testing.assert_allclose(combo2(positions()), np.full(4, 6.0))
        with pytest.raises(NotImplementedError, match="Addition"):
            ConstantDensity(1.0) + object()


class TestConcreteDensityModels:
    def test_plummer_sphere_matches_formula_and_coordinate_transform(self):
        """Test that PlummerSphere matches the analytical formula and handles coordinate transformations correctly."""
        model = PlummerSphere(total_mass=10.0, scale_radius=2.0)
        pos = positions()
        radius = np.linalg.norm(pos, axis=1)
        expected = (3 * 10.0 * 2.0**2 / (4 * np.pi)) / (radius**2 + 2.0**2) ** 2.5
        np.testing.assert_allclose(model(pos), expected)

        shifted = PlummerSphere(10.0, 2.0, coordinate=ShiftOnlyTransform([1.0, 0.0, 0.0]))
        assert shifted([1.0, 0.0, 0.0]) == pytest.approx(model([0.0, 0.0, 0.0]))

    def test_double_exponential_disk_matches_formula(self):
        """Test that DoubleExponentialDisk matches the analytical formula for density."""
        model = DoubleExponentialDisk(sigma0=5.0, scale_length=2.0, scale_height=0.5)
        pos = positions()
        radius_cyl = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        expected = 5.0 / (2 * 0.5) * np.exp(-radius_cyl / 2.0) * np.exp(-np.abs(pos[:, 2]) / 0.5)
        np.testing.assert_allclose(model(pos), expected)

    def test_power_law_sphere_matches_formula_and_origin_behavior(self):
        """Test that PowerLawSphere matches the analytical formula and handles behavior at the origin correctly."""
        model = PowerLawSphere(total_mass=6.0, scale_radius=2.0, slope=1.0)
        pos = positions()
        radius = np.linalg.norm(pos, axis=1)
        coeff = (3.0 - 1.0) * 6.0 / (4 * np.pi * 2.0**3)
        expected = coeff / ((radius / 2.0) ** 1.0 * (1.0 + radius / 2.0) ** 3.0)
        density = model(pos)
        np.testing.assert_allclose(density[1:], expected[1:])

        cored = PowerLawSphere(total_mass=6.0, scale_radius=2.0, slope=0.0)
        assert np.isfinite(cored(pos)).all()


class TestDensityCombination:
    def test_combined_density_distribution_evaluation(self, out_dir):
        """Test that CombinedDensityDistribution correctly evaluates the sum of multiple density distributions."""
        target_model = (PlummerSphere(1e10, 3.0) + 
                PlummerSphere(5e10, 0.2).set_coordinate(scales=(1,0.4,0.3),angles=(0,0,45/180*3.1416), center_pos=(0, 0, 0)) +
                PlummerSphere(2e10, 1).set_coordinate(scales=(1,0.8,0.6),angles=(0,0,30/180*3.1416), center_pos=(0, 0, 0)))
        
        # dominant component should be the second one, a:b:c = 1:0.4:0.3,
        shape1 = target_model.shape_at(0.1)
        assert shape1["eps_ab"] == pytest.approx(0.6, abs=0.01)
        assert shape1["eps_ac"] == pytest.approx(0.7, abs=0.01)
        
        # dominant component should be the third one, a:b:c = 1:0.8:0.6, 
        shape2 = target_model.shape_at(1.5)
        assert shape2["eps_ab"] == pytest.approx(0.2, abs=0.01)
        assert shape2["eps_ac"] == pytest.approx(0.4, abs=0.01)
        
        
        data = target_model.project_2d(x_range=(-2,2),y_range=(-2,2),resolution=50,z_range=(-4,4))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        im = show_image(data, axesObj=ax)
        out = out_dir / "test_combined_density.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        assert out.exists()
