
import os
import warnings
from dataclasses import dataclass, field, is_dataclass
from enum import IntEnum
from typing import Literal, Optional

try:
    import numba    # type: ignore
except ImportError:
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


default_thread_count = None
try:
    import psutil   # type: ignore
    default_thread_count = psutil.cpu_count(logical=False)
except ImportError:
    pass

if default_thread_count is None:
    default_thread_count = os.cpu_count()
    if default_thread_count is None:
        default_thread_count = 1
    else:
        default_thread_count = max(default_thread_count // 2, 1)
    


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

@dataclass
class GeneralConfig:
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
    render_double : bool
        Use double precision for rendering.
    """
    min_batchsize: int = 200000         # Minimum batch size for processing
    number_of_threads: int = -1         # Number of threads for parallel processing
    use_cython: bool = True             # Use Cython for acceleration
    render_double: bool = False         # Use double precision for rendering

    def __post_init__(self):
        if self.min_batchsize <= 0:
            self.min_batchsize = 200000
        if self.number_of_threads <= 0:
            self.number_of_threads = default_thread_count

        if  not NUMBA_AVAILABLE and not self.use_cython:
            self.use_cython = True
            warnings.warn(
                "Numba is not available. Using Cython as a fallback.",
                UserWarning
            )

@dataclass
class LoggerConfig:
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
class DensityKNNConfig:
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
    leafsize: Optional[int] = None
    
    def __post_init__(self):
        if self.leafsize is None:
            self.leafsize = max(int(self.k_neighbors / 2), 10)

@dataclass
class EllipsoidConfig:
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

PLUGIN_MANAGER_MODULES = {
    "gal3d.point.density_estimator",
    "gal3d.shape.geometry",
    "gal3d.shape.coordinate",
    "gal3d.visualization.model_projector",
    "gal3d.optimization.optimizer",
    "gal3d.characterization.characterizer",
    "gal3d.fit_workflow.fit_workflow"
}

@dataclass(frozen=True)
class Config:
    """
    Top-level configuration dataclass.

    Parameters
    ----------
    general : GeneralConfig
        General settings section.
    logger : LoggerConfig
        Logger settings section.
    ellipsoid_s : EllipsoidConfig
        Ellipsoid S settings section.
    # Add more sections as needed, e.g. database, simulation, etc.
    """
    general: GeneralConfig = field(default_factory=GeneralConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    densityknn: DensityKNNConfig = field(default_factory=DensityKNNConfig)
    ellipsoid_s: EllipsoidConfig = field(default_factory=EllipsoidConfig)
    plugin_manager_modules: tuple = field(default_factory=lambda: tuple(PLUGIN_MANAGER_MODULES))
    
        
# Instantiate configuration
config: Config = Config()