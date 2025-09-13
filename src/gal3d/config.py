"""
Module: gal3d.config

This module defines configuration classes and utilities for the gal3d framework.

Usage Example
-------------
>>> from gal3d.config import config
>>> print(config.general.min_batchsize)
>>> config.general.number_of_threads = 8
>>> config.update({"general": {"number_of_threads": 16}})
>>> config.save("my_config.json")  # Save configuration to file
>>> config.load("my_config.json")  # Load configuration from file

"""

import json
import os
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pprint import pformat
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

if TYPE_CHECKING:
    from logging import Logger

cpu_count: int
try:
    import psutil
    cpu_count = psutil.cpu_count(logical=False)
except ImportError:
    os_cpu = os.cpu_count()
    if os_cpu is not None:
        cpu_count = os_cpu
    else:
        cpu_count = 1

DEFAULT_PLUGIN_MODULES = {
    "gal3d.point.density_estimator",
    "gal3d.shape.geometry",
    "gal3d.shape.coordinate",
    "gal3d.visualization.model_projector",
    "gal3d.optimization.optimizer",
    "gal3d.characterization.characterizer",
    "gal3d.model_workflow.fit_workflow",
    "gal3d.model_workflow.error_workflow"
}
class IterationMethod(IntEnum):
    """
    Iteration methods for ray-ellipsoid intersection.

    Parameters
    ----------
    NEWTON : int
        Newton's method (1st order).
        Iteration formula:
            :math:`x_{n+1} = x_n - f(x_n) / f'(x_n)`
    HALLEY : int
        Halley's method (2nd order).
        Iteration formula:
            :math:`x_{n+1} = x_n - [2 f(x_n) f'(x_n)] / [2 (f'(x_n))^2 - f(x_n) f''(x_n)]`
    HOUSEHOLDER : int
        Householder's method (3rd order).
        Iteration formula:
            :math:`x_{n+1} = x_n - [6 f(x_n) (f'(x_n))^2 - 3 f(x_n)^2 f''(x_n)] /
                            [6 (f'(x_n))^3 - 6 f(x_n) f'(x_n) f''(x_n) + f(x_n)^2 f'''(x_n)]`
    """
    NEWTON = 1
    HALLEY = 2
    HOUSEHOLDER = 3

    @property
    def value(self) -> Literal[1, 2, 3]:
        return super().value

    def __repr__(self) -> str:
        return f"{self.name}({self.value})"

@dataclass
class BaseConfig:
    """Base configuration class with common functionality."""

    def __repr__(self) -> str:
        """Prettier representation of the config object."""
        cls_name = self.__class__.__name__
        attrs = pformat(asdict(self), indent=2)
        return f"{cls_name}:\n{attrs}"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate configuration values. To be implemented by subclasses."""

@dataclass
class GeneralConfig(BaseConfig):
    """
    General configuration parameters.

    Parameters
    ----------
    min_batchsize : int
        Minimum batch size for processing, to prevent memory overflow.
    number_of_threads : int
        Number of threads for parallel processing; -1 means auto-select.
    use_cython : bool
        Use Cython for acceleration; if False and numba is available, use numba.
    max_instances : int
        Maximum number of cached instances.
    """
    min_batchsize: int = 200000         # Minimum batch size for processing
    number_of_threads: int = -1         # Number of threads for parallel processing
    use_cython: bool = True             # Use Cython for acceleration
    max_instances: int = 20              # Maximum number of cached instances

    def validate(self) -> None:
        """Validate and correct configuration values."""
        if self.min_batchsize <= 0:
            self.min_batchsize = 200000
            warnings.warn(f"Invalid min_batchsize corrected to {self.min_batchsize}",stacklevel=2)

        if self.number_of_threads <= 0:
            self.number_of_threads = cpu_count

        if self.max_instances <= 0:
            self.max_instances = 20
            warnings.warn(f"Invalid max_instances corrected to {self.max_instances}",stacklevel=2)

        if not self.use_cython:
            self.use_cython = True
            warnings.warn(
                "Numba support has been disabled. Using Cython as a fallback.",
                UserWarning, stacklevel=2
            )
    def optimize_thread_count(self,
                     benchmark_size: int = 1024,
                     min_threads: int = 1,
                     max_threads: int | None = None,
                     test_function: Callable | None = None,
                     iterations: int = 100,
                     progress_bar: bool = False,
                     print_result: bool = False,
                     early_stop: bool = True,
                     real_world_factor: float = 0.75,
                     return_mode: Literal["recommended", "fastest", "adjusted", "balanced"] = "recommended"
                     ) -> int:
        """
        Find the optimal thread count for OpenMP/nogil parallel functions through benchmarking.

        This is now a wrapper around gal3d.util.thread_optimizer.optimize_thread_count

        Parameters
        ----------
        benchmark_size : int, optional
            Size of the arrays used for benchmarking, by default 1,000,000
        min_threads : int, optional
            Minimum number of threads to test, by default 1
        max_threads : int, optional
            Maximum number of threads to test, defaults to 2x physical cores if None
        test_function : Callable, optional
            Custom function to benchmark. If None, uses RotateAndShift as default test
        iterations : int, optional
            Number of iterations for each benchmark, by default 10
        progress_bar : bool, optional
            Whether to show a progress bar during benchmarking, by default False
        print_result : bool, optional
            Whether to print the result of the benchmarking, by default False
        early_stop : bool, optional
            Whether to stop testing when performance degrades, by default True
        real_world_factor : float, optional
            Factor to apply to raw thread count for real-world workloads, by default 0.75
        return_mode: Literal["recommended", "fastest", "adjusted", "balanced"], optional
            The mode for returning the thread count, by default "recommended".

        Returns
        -------
        int
            Optimal thread count for the current system

        Examples
        --------
        >>> from gal3d.config import config
        >>> # Basic usage
        >>> optimal_threads = config.general.optimize_thread_count()
        >>> config.general.number_of_threads = optimal_threads
        """
        from gal3d.util.thread_optimizer import optimize_thread_count as _optimize

        # Create a function to set thread count
        def set_threads(n):
            self.number_of_threads = n

        # Use the utility function
        return _optimize(
            set_threads,
            benchmark_size=benchmark_size,
            min_threads=min_threads,
            max_threads=max_threads,
            test_function=test_function,
            iterations=iterations,
            progress_bar=progress_bar,
            print_result = print_result,
            early_stop=early_stop,
            real_world_factor=real_world_factor,
            return_mode = return_mode,
        )

    def set_optimal_thread_count(self, logger: Optional["Logger"] = None) -> None:
        """
        Benchmark and set the optimal thread count automatically.

        This is a convenience method that runs optimize_thread_count()
        and automatically updates the configuration with the result.
        """
        optimal_threads = self.optimize_thread_count()
        self.number_of_threads = optimal_threads
        if logger is not None:
            logger.info("Thread count has been set to optimal value: %d", optimal_threads)

@dataclass
class LoggerConfig(BaseConfig):
    """
    Logger configuration parameters.

    Parameters
    ----------
    level : int
        Logging level (10: debug, 20: info, 30: warning, 40: error, 50: critical).
    save_file : bool
        Whether to save logs to a file.
    file_name : str
        Log file name.
    file_level : int
        Log file level.
    stream_level : int
        Console log level.
    """
    level: int = 20                     # Logging level
    save_file: bool = False             # Save logs to a file
    file_name: str = "gal3d.log"        # Log file name
    file_level: int = 20                # Log file level
    stream_level: int = 20              # Console log level


@dataclass
class DensityKNNConfig(BaseConfig):
    """
    DensityKNN configuration parameters.

    Parameters
    ----------
    k_neighbors : int
        Number of neighbors to use for KNN.
    leafsize : int, optional
        Leaf size for KNN tree construction. If None, will be set to max(k_neighbors // 2, 10).
    """
    k_neighbors: int = 32
    leafsize: int | None = None

    def validate(self) -> None:
        """Validate and correct configuration values."""
        if self.k_neighbors <= 0:
            self.k_neighbors = 32
            warnings.warn(f"Invalid k_neighbors corrected to {self.k_neighbors}",stacklevel=2)

        if self.leafsize is None:
            self.leafsize = max(int(self.k_neighbors / 2), 10)
        elif self.leafsize <= 0:
            self.leafsize = max(int(self.k_neighbors / 2), 10)
            warnings.warn(f"Invalid leafsize corrected to {self.leafsize}",stacklevel=2)

@dataclass
class SPHRenderConfig(BaseConfig):
    """
    SPH rendering configuration parameters.

    Parameters
    ----------
    resolution : int
        Resolution of the rendered image.
    render_double : bool
        Whether to use double precision for rendering.
    subsample : int
        Subsampling factor. Particles subsampled by this factor.
    """
    resolution: int = 500
    render_double: bool = False         # Use double precision for rendering
    subsample: int = 1                   # Subsampling factor

    def validate(self) -> None:
        """Validate and correct configuration values."""
        if self.resolution <= 0:
            self.resolution = 500
            warnings.warn(f"Invalid resolution corrected to {self.resolution}", stacklevel=2)

        if self.subsample <= 0:
            self.subsample = 1
            warnings.warn(f"Invalid subsample corrected to {self.subsample}", stacklevel=2)

@dataclass
class EllipsoidConfig(BaseConfig):
    """
    Ellipsoid_S configuration parameters.

    Parameters
    ----------
    DistIteration : IterationMethod
        Distance iteration method.
    LineIteration : IterationMethod
        Line iteration method.
    MaxIterationDist : int
        Maximum iterations for ray distance.
    MaxIterationLine : int
        Maximum iterations for line intersection.

    """
    DistIteration: IterationMethod = IterationMethod.HALLEY   # Distance iteration method
    LineIteration: IterationMethod = IterationMethod.HALLEY    # Line iteration method  #TODO, currently only Newton
    MaxIterationDist: int = 100                          # Maximum iterations for ray distance
    MaxIterationLine: int = 100
    # Maximum iterations for line intersection
    def __setattr__(self, name, value):
        if name in ("DistIteration", "LineIteration"):
            value = IterationMethod(value)
        super().__setattr__(name, value)

    def validate(self) -> None:
        """Validate and correct configuration values."""

        if self.MaxIterationDist <= 0:
            self.MaxIterationDist = 100
            warnings.warn(f"Invalid MaxIterationDist corrected to {self.MaxIterationDist}",stacklevel=2)

        if self.MaxIterationLine <= 0:
            self.MaxIterationLine = 100
            warnings.warn(f"Invalid MaxIterationLine corrected to {self.MaxIterationLine}",stacklevel=2)


@dataclass
class PluginManagerConfig(BaseConfig):
    """
    Plugin Manager configuration.

    Parameters
    ----------
    modules : Set[str]
        Set of module paths to be loaded by the plugin manager.
    """
    section_name: ClassVar[str] = "plugin_manager"
    modules: set[str] = field(default_factory=lambda: DEFAULT_PLUGIN_MODULES.copy())

    def __setattr__(self, name, value):
        if name == "modules":
            value = set(value)
        super().__setattr__(name, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert plugin manager config to dictionary."""
        return {"modules": list(self.modules)}

    def add_module(self, module_path: str) -> None:
        """
        Add a module to the plugin manager.

        Parameters
        ----------
        module_path : str
            Path to the module to add.
        """
        self.modules.add(module_path)

    def remove_module(self, module_path: str) -> None:
        """
        Remove a module from the plugin manager.

        Parameters
        ----------
        module_path : str
            Path to the module to remove.
        """
        if module_path in self.modules:
            self.modules.remove(module_path)
@dataclass
class Config:
    """
    Top-level configuration dataclass.

    Parameters
    ----------
    general : GeneralConfig
        General settings section.
    logger : LoggerConfig
        Logger settings section.
    densityknn : DensityKNNConfig
        Density KNN settings section.
    sph_render : SPHRenderConfig
        SPH rendering settings section.
    ellipsoid_s : EllipsoidConfig
        Ellipsoid_S settings section.
    # Add more sections as needed, e.g. database, simulation, etc.
    """
    general: GeneralConfig = field(default_factory=GeneralConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    densityknn: DensityKNNConfig = field(default_factory=DensityKNNConfig)
    sph_render: SPHRenderConfig = field(default_factory=SPHRenderConfig)
    ellipsoid_s: EllipsoidConfig = field(default_factory=EllipsoidConfig)
    plugin_modules: PluginManagerConfig = field(default_factory=PluginManagerConfig)

    def __post_init__(self):
        """Validate all configuration sections."""
        self.validate()

    def validate(self):
        """Validate all configuration sections."""
        self.general.validate()
        self.logger.validate()
        self.densityknn.validate()
        self.sph_render.validate()
        self.ellipsoid_s.validate()

    def __repr__(self) -> str:
        """Pretty representation of the entire configuration."""
        sections = [
            f"[General]\n{self.general}",
            f"[Logger]\n{self.logger}",
            f"[DensityKNN]\n{self.densityknn}",
            f"[SPHRender]\n{self.sph_render}",
            f"[Ellipsoid_S]\n{self.ellipsoid_s}",
            f"[Plugin Modules]\n{pformat(self.plugin_modules)}"
        ]
        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "general": asdict(self.general),
            "logger": asdict(self.logger),
            "densityknn": asdict(self.densityknn),
            "sph_render": asdict(self.sph_render),
            "ellipsoid_s": {k: (v.value if isinstance(v, IterationMethod) else v)
                           for k, v in asdict(self.ellipsoid_s).items()},
            "plugin_modules": list(self.plugin_modules.modules)
        }
        return config_dict

    def update(self, config_dict: dict[str, Any]) -> None:
        """
        Update configuration with values from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary with configuration values to update.
            The keys should match section names, and values should be
            dictionaries with parameter names and values.
        """
        for section_name, section_value in config_dict.items():
            if not hasattr(self, section_name):
                warnings.warn(f"Unknown configuration section: {section_name}", stacklevel=2)
                continue

            # Special handling for plugin_modules which is a tuple, not a config class
            if section_name == "plugin_modules":
                if isinstance(section_value, list):
                    setattr(self, section_name, tuple(section_value))
                continue

            # For normal config sections (which are objects)
            section = getattr(self, section_name)
            if not isinstance(section_value, dict):
                warnings.warn(f"Section {section_name} value must be a dictionary", stacklevel=2)
                continue

            for param_name, param_value in section_value.items():
                if not hasattr(section, param_name):
                    warnings.warn(f"Unknown parameter {param_name} in section {section_name}", stacklevel=2)
                    continue

                setattr(section, param_name, param_value)

        # Validate after updating
        self.validate()

    def save(self, filepath: str) -> None:
        """
        Save configuration to file.

        Parameters
        ----------
        filepath : str
            Path to save configuration to. Extension determines format (.json or .yaml).
        """
        config_dict = self.to_dict()

        if filepath.lower().endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.lower().endswith((".yaml", ".yml")):
            import yaml  # type: ignore
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

    def load(self, filepath: str) -> None:
        """
        Load configuration from file.

        Parameters
        ----------
        filepath : str
            Path to load configuration from. Extension determines format (.json or .yaml).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        if filepath.lower().endswith(".json"):
            with open(filepath) as f:
                config_dict = json.load(f)
        elif filepath.lower().endswith((".yaml", ".yml")):
            import yaml
            with open(filepath) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

        self.update(config_dict)

    def reset(self) -> "Config":
        """Reset all configuration to default values."""
        # Create a new instance with defaults and copy its attributes
        default_config = Config()
        for attr in self.__dataclass_fields__:
            setattr(self, attr, getattr(default_config, attr))
        self.validate()
        return self

# Instantiate configuration
config: Config = Config()
