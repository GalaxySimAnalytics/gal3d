"""
Thread optimization utilities for gal3d.

This module provides functions to determine the optimal number of threads
for OpenMP parallel operations based on benchmarking actual workloads.
"""

import os
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np


def optimize_thread_count(
    set_threads_func: Callable[[int], None],
    benchmark_size: int = 1024,
    min_threads: int = 1,
    max_threads: int | None = None,
    test_function: Callable[[], None] | None = None,
    iterations: int = 100,
    progress_bar: bool = True,
    print_result: bool = True,
    early_stop: bool = True,
    real_world_factor: float = 0.75,
    return_mode: Literal["recommended", "fastest", "adjusted", "balanced"] = "recommended",
) -> int:
    """
    Find the optimal thread count for OpenMP/nogil parallel functions through benchmarking.

    Parameters
    ----------
    set_threads_func : Callable[[int], None]
        Function that sets the thread count for testing, e.g., lambda n: config.general.number_of_threads = n
    benchmark_size : int, optional
        Size of the arrays used for benchmarking, by default 1,024
    min_threads : int, optional
        Minimum number of threads to test, by default 1
    max_threads : int, optional
        Maximum number of threads to test, defaults to CPU count if None
    test_function : Callable[[], None], optional
        Custom function to benchmark. If None, uses a default test
    iterations : int, optional
        Number of iterations for each benchmark, by default 100
    progress_bar : bool, optional
        Whether to display progress bar during benchmarking, by default True
    print_result : bool, optional
        Whether to print benchmark results, by default True
    early_stop : bool, optional
        Whether to stop testing when performance degrades significantly, by default True
    real_world_factor : float, optional
        Factor to apply to raw thread count for real-world workloads, by default 0.75
    return_mode : str, optional
        Which type of optimal thread count to return:
        - "recommended": The overall recommended count (default)
        - "fastest": The raw best performing thread count
        - "adjusted": The raw best adjusted by real_world_factor
        - "balanced": The thread count where diminishing returns start

    Returns
    -------
    int
        Optimal thread count for the current system according to return_mode
    """
    # Validate input parameters
    _validate_parameters(return_mode)

    # Set up test environment
    max_threads = _setup_max_threads(max_threads)
    test_function = _setup_test_function(test_function, benchmark_size)

    # Run benchmarks
    results, best_threads, best_time = _run_benchmarks(
        set_threads_func, min_threads, max_threads, test_function, iterations, progress_bar, early_stop
    )

    # Analyze results
    thread_analysis = _analyze_results(results, best_threads, real_world_factor)

    # Print results if requested
    if print_result:
        _print_benchmark_results(results, thread_analysis, real_world_factor)

    # Return requested thread count
    return _select_result_by_mode(thread_analysis, return_mode, print_result)


def _validate_parameters(return_mode: str) -> None:
    """Validate input parameters."""
    valid_modes = ["recommended", "fastest", "adjusted", "balanced"]
    if return_mode not in valid_modes:
        raise ValueError(f"Invalid return_mode '{return_mode}'. Must be one of {valid_modes}")


def _setup_max_threads(max_threads: int | None) -> int:
    """Determine maximum threads to test."""
    if max_threads is None:
        cpu_count = os.cpu_count() or 16
        return min(cpu_count, 32)  # Cap at 32
    return max_threads


def _setup_test_function(test_function: Callable[[], Any] | None, benchmark_size: int) -> Callable[[], Any]:
    """Set up the test function for benchmarking."""
    if test_function is not None:
        return test_function

    # Create default test function with random data
    rng = np.random.default_rng()
    data = rng.random((benchmark_size, 3)).astype(np.float64)
    rotation_matrix = rng.random((3, 3)).astype(np.float64)
    center = rng.random(3).astype(np.float64)

    from gal3d.shape.geometry_plugins.ellipsoid_s_cy import f_ray_shaped_ellipsoid
    from gal3d.util.array_operate_cy import RotateAndShift

    return lambda: f_ray_shaped_ellipsoid(
        3.0, 2.0, 1.0, 1.5, 1.0, 0.5, RotateAndShift(data, rotation_matrix, center), 100
    )


def _run_benchmarks(
    set_threads_func: Callable[[int], None],
    min_threads: int,
    max_threads: int,
    test_function: Callable[[], Any],
    iterations: int,
    progress_bar: bool,
    early_stop: bool,
) -> tuple[list[tuple[int, float]], int, float]:
    """Run benchmarks for different thread counts."""
    results: list[tuple[int, float]] = []
    raw_times: dict[int, list[float]] = {}
    best_threads = 1
    best_time = float("inf")
    single_thread_total_time = 0.0

    total_threads = max_threads - min_threads + 1
    completed_threads = 0

    # For animated progress indicator
    spinner_chars = ["-", "\\", "|", "/"]
    spinner_idx = 0

    for threads in range(min_threads, max_threads + 1):
        completed_threads += 1

        # Update progress indicator
        if progress_bar:
            _update_progress(completed_threads, total_threads, threads, max_threads, spinner_chars, spinner_idx)
            spinner_idx += 1

        # Set thread count and run benchmark
        set_threads_func(threads)

        # Warm up to ensure JIT compilation is done
        _ = test_function()

        # Run the benchmark
        times = []
        total_time = 0.0
        early_terminated = False

        for _i in range(iterations):
            start_time = time.perf_counter()
            _ = test_function()
            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            times.append(elapsed)

            # Early termination check
            if early_stop and threads != 1 and total_time > single_thread_total_time:
                early_terminated = True
                break

        # Handle early termination
        if early_terminated and progress_bar:
            print(
                f"\n  Thread {threads} terminated after {_i + 1}/{iterations} iterations "
                f"(time: {total_time:.4f}s > single thread: {single_thread_total_time:.4f}s)"
            )
            break

        # Store results
        raw_times[threads] = times
        results.append((threads, total_time))

        if threads == 1:
            single_thread_total_time = total_time

        # Update best time
        if total_time < best_time:
            best_time = total_time
            best_threads = threads

    return results, best_threads, best_time


def _update_progress(
    completed_threads: int,
    total_threads: int,
    threads: int,
    max_threads: int,
    spinner_chars: list[str],
    spinner_idx: int,
) -> None:
    """Update progress bar display."""
    progress_pct = 100 * completed_threads / total_threads
    spinner_char = spinner_chars[spinner_idx % len(spinner_chars)]

    progress_str = "[" + "#" * int(progress_pct / 5) + " " * (20 - int(progress_pct / 5)) + "]"
    status_line = f"\rTesting {spinner_char} Thread {threads:2d}/{max_threads} {progress_str} {progress_pct:.1f}%"
    print(status_line, end="", flush=True)


def _analyze_results(results: list[tuple[int, float]], best_threads: int, real_world_factor: float) -> dict[str, int]:
    """Analyze benchmark results and calculate optimal thread counts."""
    analysis = {
        "fastest": best_threads,
        "adjusted": max(1, int(best_threads * real_world_factor)),
        "balanced": 1,  # Default value
    }

    # Calculate diminishing returns point
    if len(results) > 2:
        diminishing_best = _find_diminishing_returns_point(results)
        analysis["balanced"] = diminishing_best

    # Calculate final recommendation
    analysis["recommended"] = min(analysis["adjusted"], analysis["balanced"])

    return analysis


def _find_diminishing_returns_point(results: list[tuple[int, float]]) -> int:
    """Find the point where adding more threads gives diminishing returns."""
    improvements = []
    for i in range(1, len(results)):
        prev_time = results[i - 1][1]
        curr_time = results[i][1]
        improvement = (prev_time - curr_time) / prev_time  # Relative improvement
        improvements.append((results[i][0], improvement))

    # Find where improvement drops below 5%
    diminishing_best = results[-1][0]  # Default to highest thread count
    for threads, improvement in improvements:
        if improvement < 0.05:  # Less than 5% improvement
            diminishing_best = max(1, threads - 1)  # Use previous thread count
            break
        diminishing_best = threads

    return diminishing_best


def _print_benchmark_results(
    results: list[tuple[int, float]], analysis: dict[str, int], real_world_factor: float
) -> None:
    """Print benchmark results and analysis."""
    print("\n====== Benchmark Results ======")

    # Get baseline (single thread) for calculating speedup
    baseline = results[0][1] if results[0][0] == 1 else None

    # Print individual results
    for threads, time_taken in results:
        speedup = baseline / time_taken if baseline is not None else 0.0
        speedup_str = f"{speedup:.2f}x" if baseline is not None else "N/A"
        efficiency = speedup / threads if threads > 0 and speedup > 0 else 0.0
        efficiency_str = f"{efficiency:.2f}" if baseline is not None and threads > 0 else "N/A"

        markers = []
        if threads == analysis["fastest"]:
            markers.append("FASTEST")
        if threads == analysis["adjusted"]:
            markers.append("ADJUSTED")
        if threads == analysis["balanced"]:
            markers.append("BALANCED")
        if threads == analysis["recommended"]:
            markers.append("RECOMMENDED")

        marker_str = f" ({', '.join(markers)})" if markers else ""
        print(
            f"Threads: {threads:2d}, Time: {time_taken:.4f}s, Speedup: {speedup_str}, "
            f"Efficiency: {efficiency_str}{marker_str}"
        )

    print("\n====== Analysis ======")
    print(f"Raw performance best: {analysis['fastest']} threads (FASTEST)")
    print(f"Adjusted for overhead: {analysis['adjusted']} threads (ADJUSTED, using factor {real_world_factor})")
    print(f"Diminishing returns: {analysis['balanced']} threads (BALANCED)")
    print(f"Overall recommendation: {analysis['recommended']} threads (RECOMMENDED)")


def _select_result_by_mode(analysis: dict[str, int], return_mode: str, print_result: bool) -> int:
    """Return the requested thread count based on return mode."""
    result = analysis[return_mode]

    if print_result:
        print(f"\nReturning {return_mode.upper()} thread count: {result}")

    return result


class ThreadOptimizer:
    """
    Class-based interface for thread count optimization.

    This provides a more object-oriented approach when that's preferred.
    """

    def __init__(self, set_threads_func: Callable[[int], None]):
        """
        Initialize the thread optimizer.

        Parameters
        ----------
        set_threads_func : Callable[[int], None]
            Function that sets the thread count for testing
        """
        self.set_threads_func = set_threads_func

    def optimize(self, **kwargs: Any) -> int:
        """
        Find the optimal thread count using the same parameters as optimize_thread_count.

        Parameters
        ----------
        **kwargs
            All parameters accepted by optimize_thread_count except set_threads_func

        Returns
        -------
        int
            Optimal thread count
        """
        return optimize_thread_count(self.set_threads_func, **kwargs)
