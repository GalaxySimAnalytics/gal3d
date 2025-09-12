"""
Custom exceptions for the gal3d framework.
"""

class FitDataError(ValueError):
    """Raised when data issues prevent fitting, such as insufficient points or poor uniformity."""
