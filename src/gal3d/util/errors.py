"""
Custom exceptions for the gal3d framework.
"""

class FitDataError(ValueError):
    """Raised when data issues prevent fitting, such as insufficient points or poor uniformity."""


class InsufficientPointsError(FitDataError):
    """Insufficient points for fitting: < 12"""

class PoorUniformityError(FitDataError):
    """Poor point distribution uniformity"""
