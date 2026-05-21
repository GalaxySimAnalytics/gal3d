"""Tests for the configuration system in gal3d.configuration."""

import json
import warnings

import pytest

from gal3d.configuration import (
    Config,
    DensitySPHConfig,
    EllipsoidConfig,
    GeneralConfig,
    IterationMethod,
    LOSIntegratorConfig,
    PluginManagerConfig,
    SPHRenderConfig,
)


class TestGeneralConfig:
    def test_invalid_values_are_corrected_with_warnings(self):
        """Test that invalid values in GeneralConfig are corrected and raise warnings."""
        with pytest.warns(UserWarning):
            cfg = GeneralConfig(min_batchsize=0, max_instances=0, use_cython=False, number_of_threads=0)
        assert cfg.min_batchsize == 200000
        assert cfg.max_instances == 20
        assert cfg.use_cython is True
        assert cfg.number_of_threads >= 1

    def test_optimize_and_set_thread_count_delegate(self, monkeypatch):
        """Test that optimize_thread_count and set_optimal_thread_count delegate to the thread optimizer."""
        cfg = GeneralConfig(number_of_threads=1)
        cfg.optimize_thread_count

        seen_iterations = []

        def fake_optimize(set_threads, **kwargs):
            set_threads(7)
            seen_iterations.append(kwargs["iterations"])
            return 7

        monkeypatch.setattr("gal3d.util.thread_optimizer.optimize_thread_count", fake_optimize)
        assert cfg.optimize_thread_count(iterations=3) == 7
        assert cfg.number_of_threads == 7
        assert seen_iterations[-1] == 3

        messages = []

        class Logger:
            def info(self, message, value):
                messages.append((message, value))

        cfg.set_optimal_thread_count(logger=Logger())
        assert messages[-1][1] == 7
        assert seen_iterations[-1] == 100


class TestIndividualConfigs:
    def test_density_sph_and_los_validation(self):
        """Test that invalid values in DensitySPHConfig and LOSIntegratorConfig are corrected and raise warnings/errors."""
        with pytest.warns(UserWarning):
            density = DensitySPHConfig(k_neighbors=0, leafsize=0, workers=0)
        assert density.k_neighbors == 32
        assert density.leafsize == 16
        assert density.workers >= 1

        with pytest.raises(ValueError, match="nz_min"):
            LOSIntegratorConfig(nz_min=2)
        with pytest.raises(ValueError, match="nz_max"):
            LOSIntegratorConfig(nz_min=5, nz_max=3)
        with pytest.raises(ValueError, match="rtol"):
            LOSIntegratorConfig(rtol=-1)
        with pytest.raises(ValueError, match="atol"):
            LOSIntegratorConfig(atol=-1)

        with pytest.warns(UserWarning):
            render = SPHRenderConfig(resolution=0, subsample=0)
        assert render.resolution == 500
        assert render.subsample == 1

    def test_ellipsoid_enum_conversion_and_validation(self):
        """Test that EllipsoidConfig correctly converts iteration method enums and validates values."""
        cfg = EllipsoidConfig(DistIteration=1, LineIteration=IterationMethod.HOUSEHOLDER)
        assert cfg.DistIteration is IterationMethod.NEWTON
        assert cfg.LineIteration is IterationMethod.HOUSEHOLDER
        assert repr(IterationMethod.HALLEY) == "HALLEY(2)"

        with pytest.warns(UserWarning):
            invalid = EllipsoidConfig(MaxIterationDist=0, MaxIterationLine=0, EpsTableN=1)
        assert invalid.MaxIterationDist == 100
        assert invalid.MaxIterationLine == 100
        assert invalid.EpsTableN == 81

        with pytest.raises(ValueError):
            cfg.DistIteration = 99


class TestPluginManagerConfig:
    def test_modules_are_stored_as_set_and_mutated(self):
        """Test that PluginManagerConfig stores modules as a set and allows mutation."""
        cfg = PluginManagerConfig(modules=["a", "a", "b"])
        assert cfg.modules == {"a", "b"}
        assert cfg.to_dict() == {"modules": ["a", "b"]} or cfg.to_dict() == {"modules": ["b", "a"]}

        cfg.add_module("c")
        cfg.remove_module("a")
        cfg.remove_module("missing")
        assert cfg.modules == {"b", "c"}


class TestConfig:
    def test_to_dict_update_reset_and_repr(self):
        """Test the to_dict, update, reset, and repr methods of Config."""
        cfg = Config()
        data = cfg.to_dict()
        assert set(data) >= {"general", "logger", "densitysph", "ellipsoid_s", "plugin_modules"}
        assert "[General]" in repr(cfg)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg.update(
                {
                    "general": {"number_of_threads": 2, "unknown_param": 1},
                    "plugin_modules": {"modules": ["demo.plugins"]},
                    "unknown_section": {},
                    "logger": "not-a-dict",
                }
            )
        assert cfg.general.number_of_threads == 2
        assert cfg.plugin_modules.modules == {"demo.plugins"}
        assert len(caught) == 3

        cfg.reset()
        assert "demo.plugins" not in cfg.plugin_modules.modules

    def test_update_plugin_modules_accepts_sequences_and_warns_on_bad_type(self):
        """Test that updating plugin_modules with sequences works and warns on invalid types."""
        cfg = Config()
        cfg.update({"plugin_modules": ["one", "two"]})
        assert cfg.plugin_modules.modules == {"one", "two"}

        with pytest.warns(UserWarning, match="plugin_modules must"):
            cfg.update({"plugin_modules": object()})

    def test_json_save_load_and_file_errors(self, tmp_path):
        """Test saving and loading Config to JSON, and handling file-related errors."""
        cfg = Config()
        cfg.general.number_of_threads = 3
        cfg.plugin_modules.modules = {"custom.module"}

        path = tmp_path / "gal3d_config.json"
        cfg.save(str(path))
        raw = json.loads(path.read_text())
        assert raw["general"]["number_of_threads"] == 3

        loaded = Config()
        loaded.load(str(path))
        assert loaded.general.number_of_threads == 3
        assert loaded.plugin_modules.modules == {"custom.module"}

        with pytest.raises(FileNotFoundError):
            loaded.load(str(tmp_path / "missing.json"))
        with pytest.raises(ValueError, match="Unsupported"):
            loaded.save(str(tmp_path / "config.txt"))
        unsupported = tmp_path / "config.txt"
        unsupported.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported"):
            loaded.load(str(unsupported))

    def test_yaml_paths_use_yaml_module(self, tmp_path):
        """Test saving and loading Config to YAML, and handling file-related errors."""
        pytest.importorskip("yaml")
        cfg = Config()
        cfg.general.number_of_threads = 4
        path = tmp_path / "gal3d_config.yaml"
        cfg.save(str(path))

        loaded = Config()
        loaded.load(str(path))
        assert loaded.general.number_of_threads == 4
