"""
Module for applying optimization, including various optimization algorithms, parameter definitions, result handling, and more.
"""

from typing import TYPE_CHECKING

from .optimizer import Optimizer
from .parameter import Parameters

__all__ = ["Optimizer", "Parameters", "ModelIO"]


# Lazy import to avoid circular import with gal3d.shape
def __getattr__(name: str):  # type: ignore
    if name == "ModelIO":
        from .model_io import ModelIO  # local import to break cycle

        return ModelIO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    # For type checkers only; does not run at runtime
    from .model_io import ModelIO
