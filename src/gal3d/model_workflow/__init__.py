"""
Model workflows for fitting and error estimation.

This module provides functionality for model fitting and error estimation
through a plugin-based workflow system.
"""

from .error_workflow import ErrorWorkflow
from .fit_workflow import FitWorkflow

__all__ = ["FitWorkflow", "ErrorWorkflow"]
