import inspect
import logging
import time

import numpy as np
import pytest

from gal3d.optimization.util import truncate
from gal3d.util.array_operate import Auto3DShape
from gal3d.util.errors import FitDataError, InsufficientPointsError, PoorUniformityError
from gal3d.util.func_cache import CacheDict
from gal3d.util.func_decorator import DevelopmentWarning, deprecated, development_warning, timer
from gal3d.util.func_signature import MySignature, func_optional_key, func_required_key, update_dict_value
from gal3d.util.string_format import background_color, color, fontformat, foreground_color, string_formatter
from gal3d.util.thread_optimizer import (
    ThreadOptimizer,
    _analyze_results,
    _find_diminishing_returns_point,
    _print_benchmark_results,
    _run_benchmarks,
    _select_result_by_mode,
    _setup_max_threads,
    _setup_test_function,
    _validate_parameters,
    optimize_thread_count,
)


class TestAuto3DShape:
    def test_accepts_n_by_three_and_transposes_three_by_n(self):
        """Test that Auto3DShape.to_3d_array correctly handles (N, 3) and (3, N) arrays."""
        n_by_three = np.arange(12).reshape(4, 3)
        three_by_n = np.arange(12).reshape(3, 4)

        assert np.array_equal(Auto3DShape.to_3d_array(n_by_three), n_by_three)
        assert np.array_equal(Auto3DShape.to_3d_array(three_by_n), three_by_n.T)

    def test_reshapes_flat_and_ambiguous_arrays(self):
        """Test that Auto3DShape.to_3d_array correctly reshapes flat and ambiguous arrays."""
        assert Auto3DShape.to_3d_array([1, 2, 3]).shape == (1, 3)
        assert Auto3DShape.to_3d_array(np.arange(6)).shape == (2, 3)

        ambiguous = np.arange(9).reshape(3, 3)
        assert np.array_equal(Auto3DShape.to_3d_array(ambiguous), ambiguous)

    def test_invalid_size_raises_numpy_reshape_error(self):
        """Test that Auto3DShape.to_3d_array raises an error for invalid input sizes."""
        with pytest.raises(ValueError):
            Auto3DShape.to_3d_array([1, 2, 3, 4])


class TestStringFormat:
    def test_named_integer_tuple_colors_and_styles(self):
        """Test that foreground_color, background_color, color, and fontformat return correct ANSI codes."""
        assert foreground_color("red") == "\033[31m"
        assert foreground_color(123) == "\033[38;5;123m"
        assert foreground_color(1, 2, 3) == "\033[38;2;1;2;3m"
        assert background_color("bright_blue") == "\033[104m"
        assert background_color(5) == "\033[48;5;5m"
        assert background_color(4, 5, 6) == "\033[48;2;4;5;6m"
        assert color(bg_color="black", fg_color="white") == "\033[40m\033[37m"
        assert fontformat(bold=True, thin=True, italics=True, underline=True, strikethrough=True) == (
            "\033[1m\033[2m\033[3m\033[4m\033[9m"
        )

    def test_formatter_wraps_and_invalid_colors_raise(self):
        """Test that string_formatter correctly wraps text and raises errors for invalid colors."""
        formatted = string_formatter("gal3d", fg_color="green", bg_color=(1, 2, 3), bold=True)
        assert formatted.startswith("\033[48;2;1;2;3m\033[32m\033[1m")
        assert formatted.endswith("gal3d\033[0m")

        with pytest.raises(TypeError):
            foreground_color(1, "bad", 3)
        with pytest.raises(TypeError):
            background_color(1, 2)
        with pytest.raises(KeyError):
            foreground_color("not-a-color")


class TestCacheDict:
    def test_lru_eviction_and_access_refresh(self):
        """Test that CacheDict correctly evicts least recently used items and refreshes access order."""
        cache = CacheDict(cache_len=2)
        cache["a"] = 1
        cache["b"] = 2
        assert cache["a"] == 1
        cache["c"] = 3
        assert list(cache.keys()) == ["a", "c"]

    def test_resize_evicts_oldest_and_rejects_invalid_size(self):
        """Test that CacheDict.set_cache_len evicts oldest items and raises errors for invalid sizes."""
        cache = CacheDict({"a": 1, "b": 2, "c": 3}, cache_len=3)
        cache.set_cache_len(1)
        assert list(cache.items()) == [("c", 3)]
        with pytest.raises(AssertionError):
            CacheDict(cache_len=0)
        with pytest.raises(AssertionError):
            cache.set_cache_len(0)


class TestFunctionSignature:
    def test_signature_filters_required_optional_args_and_kwargs(self):
        """Test that MySignature can filter required positional, required keyword, and optional parameters."""
        def sample(a, /, b, c=3, *args, d, e=5, **kwargs):
            return a, b, c, args, d, e, kwargs

        sig = MySignature.from_callable(sample)
        assert sig.args is True
        assert sig.kwargs is True
        assert sig.get_params(positional=2) == {"a": inspect.Parameter.empty}
        assert sig.get_params(keyword=2, empty=1) == {"d": inspect.Parameter.empty}
        assert sig.get_params(keyword=1, empty=2) == {"c": 3, "e": 5}
        assert set(func_required_key(sample)) == {"a", "b", "d"}
        assert func_optional_key(sample) == {"c": 3, "e": 5}

    def test_signature_validation_and_dict_update(self):
        """Test that MySignature validates parameters and update_dict_value correctly updates dictionaries."""
        def sample(a, b=2):
            return a + b

        sig = MySignature.from_callable(sample)
        with pytest.raises(ValueError):
            sig.get_params(positional=99)
        with pytest.raises(ValueError):
            sig.get_params(keyword=99)
        with pytest.raises(ValueError):
            sig.get_params(empty=99)

        origin = {"a": 1, "b": 2}
        assert update_dict_value(origin, {"b": 20, "c": 30}, a=10, d=40) == {"a": 10, "b": 20}
        assert origin == {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            update_dict_value([], {})

    def test_signature_error_paths_return_documented_exceptions(self):
        """Test that MySignature correctly handles error paths and raises documented exceptions."""
        class BrokenSignature(MySignature):
            @property
            def parameters(self):
                raise RuntimeError("broken parameters")

        broken = BrokenSignature()
        with pytest.raises(ValueError, match="Failed to retrieve"):
            broken.params
        assert broken.kwargs is False
        assert broken.args is False

        class ExplodingDict(dict):
            def __iter__(self):
                raise RuntimeError("bad iteration")

        def sample(a):
            return a

        sig = MySignature.from_callable(sample)
        sig.__dict__["params"] = ExplodingDict()
        with pytest.raises(RuntimeError, match="Error filtering"):
            sig.get_params()

        class BadCopyDict(dict):
            def copy(self):
                raise RuntimeError("copy failed")

        with pytest.raises(RuntimeError, match="Failed to update"):
            update_dict_value(BadCopyDict(a=1), {"a": 2})


class TestDecorators:
    def test_timer_logs_success_and_wraps_errors(self, caplog):
        """Test that the timer decorator logs successful execution and wraps exceptions in RuntimeError."""
        logger = logging.getLogger("gal3d.tests.timer")

        @timer(logger)
        def _work(value):
            return value * 2

        @timer(logger)
        def _boom():
            raise ValueError("bad input")

        with caplog.at_level(logging.INFO, logger="gal3d.tests.timer"):
            assert _work(3) == 6
        assert "Work" in caplog.text

        with pytest.raises(RuntimeError, match="bad input"):
            _boom()

    def test_deprecation_and_development_warning_variants(self):
        """Test that deprecated and development_warning decorators correctly issue warnings with custom messages."""
        @deprecated
        def old(value):
            return value + 1

        @deprecated("custom deprecated")
        def old_custom():
            return "ok"

        @development_warning
        def newish():
            return "dev"

        class Demo:
            @development_warning("classmethod warning")
            @classmethod
            def cm(cls):
                return cls.__name__

            @deprecated("staticmethod warning")
            @staticmethod
            def sm():
                return 42

        with pytest.warns(DeprecationWarning, match="old"):
            assert old(2) == 3
        with pytest.warns(DeprecationWarning, match="custom deprecated"):
            assert old_custom() == "ok"
        with pytest.warns(DevelopmentWarning, match="under development"):
            assert newish() == "dev"
        with pytest.warns(DevelopmentWarning, match="classmethod warning"):
            assert Demo.cm() == "Demo"
        with pytest.warns(DeprecationWarning, match="staticmethod warning"):
            assert Demo.sm() == 42


class TestThreadOptimizer:
    def test_validation_helpers_and_analysis_modes(self):
        """Test that _validate_parameters, _setup_max_threads, _setup_test_function, _find_diminishing_returns_point, _analyze_results, and _select_result_by_mode work as expected."""
        _validate_parameters("recommended")
        with pytest.raises(ValueError):
            _validate_parameters("unknown")

        assert _setup_max_threads(4) == 4
        assert _setup_max_threads(None) >= 1
        assert _setup_test_function(lambda: "ok", 1)() == "ok"
        assert _find_diminishing_returns_point([(1, 10.0), (2, 8.0), (3, 7.8)]) == 2
        analysis = _analyze_results([(1, 10.0), (2, 5.0), (3, 4.9)], best_threads=3, real_world_factor=0.5)
        assert analysis == {"fastest": 3, "adjusted": 1, "balanced": 2, "recommended": 1}
        assert _select_result_by_mode(analysis, "balanced", print_result=False) == 2
        assert _select_result_by_mode(analysis, "fastest", print_result=True) == 3

    def test_optimize_thread_count_with_custom_function(self):
        """Test that optimize_thread_count correctly optimizes thread count with a custom function."""
        seen_threads = []

        def set_threads(n):
            seen_threads.append(n)

        calls = {"n": 0}

        def test_function():
            calls["n"] += 1
            return calls["n"]

        result = optimize_thread_count(
            set_threads,
            min_threads=1,
            max_threads=2,
            test_function=test_function,
            iterations=1,
            progress_bar=False,
            print_result=False,
            early_stop=False,
            return_mode="fastest",
        )
        assert result in {1, 2}
        assert seen_threads == [1, 2]

        optimizer = ThreadOptimizer(set_threads)
        assert optimizer.optimize(
            min_threads=1,
            max_threads=1,
            test_function=test_function,
            iterations=1,
            progress_bar=False,
            print_result=False,
        ) == 1

    def test_benchmark_progress_early_stop_and_printing(self, monkeypatch, capsys):
        """Test that _run_benchmarks correctly handles progress bar, early stopping, and result printing."""
        perf_values = iter([0.0, 1.0, 0.0, 2.0])
        monkeypatch.setattr(time, "perf_counter", lambda: next(perf_values))

        seen_threads = []
        results, best_threads, best_time = _run_benchmarks(
            seen_threads.append,
            min_threads=1,
            max_threads=2,
            test_function=lambda: None,
            iterations=1,
            progress_bar=True,
            early_stop=True,
        )
        assert seen_threads == [1, 2]
        assert results == [(1, 1.0)]
        assert best_threads == 1
        assert best_time == 1.0

        analysis = {"fastest": 1, "adjusted": 1, "balanced": 1, "recommended": 1}
        _print_benchmark_results([(1, 1.0), (2, 0.5)], analysis, 0.75)
        _print_benchmark_results([(2, 0.5)], {"fastest": 2, "adjusted": 1, "balanced": 1, "recommended": 1}, 0.75)
        output = capsys.readouterr().out
        assert "Testing" in output
        assert "Benchmark Results" in output


def test_truncate_and_exception_hierarchy():
    """Test that truncate correctly truncates numbers and that custom exceptions are subclasses of FitDataError."""
    assert truncate(1.239, 2) == pytest.approx(1.23)
    assert truncate(-1.239, 2) == pytest.approx(-1.23)
    assert np.isnan(truncate(np.nan, 2))
    assert truncate(np.inf, 2) == np.inf

    assert issubclass(InsufficientPointsError, FitDataError)
    assert issubclass(PoorUniformityError, FitDataError)
    with pytest.raises(FitDataError):
        raise InsufficientPointsError("too few points")
