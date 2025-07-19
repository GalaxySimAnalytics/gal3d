
import typing
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# from optimagic
@dataclass(frozen=True)
class InternalOptimizeResult:
    """Internal representation of the result of an optimization problem.

    Args:
        x: The optimal parameters.
        fun: The function value at the optimal parameters.
        success: Whether the optimization was successful.
        message: A message from the optimizer.
        status: The status of the optimization.
        n_fun_evals: The number of function evaluations.
        n_jac_evals: The number of gradient or jacobian evaluations.
        n_hess_evals: The number of Hessian evaluations.
        n_iterations: The number of iterations.
        jac: The Jacobian of the objective function at the optimal parameters.
        hess: The Hessian of the objective function at the optimal parameters.
        hess_inv: The inverse of the Hessian of the objective function at the optimal
            parameters.
        max_constraint_violation: The maximum constraint violation.
        info: Additional information from the optimizer.

    """

    x: NDArray[np.float64]
    fun: float | NDArray[np.float64]
    success: bool | None = None
    message: str | None = None
    status: int | None = None
    n_fun_evals: int | None = None
    n_jac_evals: int | None = None
    n_hess_evals: int | None = None
    n_iterations: int | None = None
    jac: NDArray[np.float64] | None = None
    hess: NDArray[np.float64] | None = None
    hess_inv: NDArray[np.float64] | None = None
    max_constraint_violation: float | None = None
    info: dict[str, typing.Any] | None = None
    multistart_info: dict[str, typing.Any] | None = None

    def __post_init__(self) -> None:
        report: list[str] = []
        if not isinstance(self.x, np.ndarray):
            report.append("x must be a numpy array")

        if not (isinstance(self.fun, np.ndarray) or np.isscalar(self.fun)):
            report.append("fun must be a numpy array or scalar")

        if self.success is not None and not isinstance(self.success, bool):
            report.append("success must be a bool or None")

        if self.message is not None and not isinstance(self.message, str):
            report.append("message must be a string or None")

        if self.n_fun_evals is not None and not isinstance(self.n_fun_evals, int):
            report.append("n_fun_evals must be an int or None")

        if self.n_jac_evals is not None and not isinstance(self.n_jac_evals, int):
            report.append("n_jac_evals must be an int or None")

        if self.n_hess_evals is not None and not isinstance(self.n_hess_evals, int):
            report.append("n_hess_evals must be an int or None")

        if self.n_iterations is not None and not isinstance(self.n_iterations, int):
            report.append("n_iterations must be an int or None")

        if self.jac is not None and not isinstance(self.jac, np.ndarray):
            report.append("jac must be a numpy array or None")

        if self.hess is not None and not isinstance(self.hess, np.ndarray):
            report.append("hess must be a numpy array or None")

        if self.hess_inv is not None and not isinstance(self.hess_inv, np.ndarray):
            report.append("hess_inv must be a numpy array or None")

        if self.max_constraint_violation is not None and not np.isscalar(
            self.max_constraint_violation
        ):
            report.append("max_constraint_violation must be a scalar or None")

        if self.info is not None and not isinstance(self.info, dict):
            report.append("info must be a dictionary or None")

        if self.status is not None and not isinstance(self.status, int):
            report.append("status must be an int or None")

        if self.max_constraint_violation and not isinstance(
            self.max_constraint_violation, float
        ):
            report.append("max_constraint_violation must be a float or None")

        if report:
            msg = (
                "The following arguments to InternalOptimizeResult are invalid:\n"
                + "\n".join(report)
            )
            raise TypeError(msg)