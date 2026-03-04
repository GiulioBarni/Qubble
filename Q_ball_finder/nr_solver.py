"""
Generic Newton–Raphson utilities used by the 2D bounce solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

ArrayLike = np.ndarray | Sequence[float]


class NewtonConvergenceError(RuntimeError):
    """
    Exception raised when Newton-Raphson iteration fails to converge.
    
    This is raised when the residual norm exceeds a threshold (typically 10^10),
    indicating that the solution is diverging and further iterations are unlikely
    to succeed.
    """
    pass


@dataclass
class NewtonResult:
    x: np.ndarray
    success: bool
    iterations: int
    residual_norm: float
    history: List[float] = field(default_factory=list)


LinearSolver = Callable[[ArrayLike, np.ndarray], np.ndarray]


def _default_linear_solver(A: ArrayLike, b: np.ndarray) -> np.ndarray:
    if sp.issparse(A):
        return spsolve(sp.csc_matrix(A), b)
    A_arr = np.asarray(A)
    return np.linalg.solve(A_arr, b)


def newton_solve(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], ArrayLike],
    x0: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 25,
    damping: float = 1.0,
    line_search: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None,
    norm: Callable[[np.ndarray], float] = lambda v: np.linalg.norm(v, ord=2),
    solve_linear: Optional[LinearSolver] = None,
    verbose: bool = False,
    callback: Optional[Callable[[int, np.ndarray, np.ndarray, float], None]] = None,
) -> NewtonResult:
    """
    Solve F(x) = 0 using Newton–Raphson with optional damping/line search.
    """
    x = np.array(x0, dtype=complex if np.iscomplexobj(x0) else float)
    if solve_linear is None:
        solve_linear = _default_linear_solver

    history: List[float] = []
    success = False
    
    # Threshold for divergence detection
    DIVERGENCE_THRESHOLD = 1e20

    for it in range(1, max_iter + 1):
        F = residual(x)
        res_norm = norm(F)
        history.append(res_norm)
        
        # Check for divergence: if residual norm exceeds threshold, solution is not converging
        if res_norm > DIVERGENCE_THRESHOLD:
            raise NewtonConvergenceError(
                f"Solution does not converge: residual norm ||F|| = {res_norm:.6e} "
                f"exceeds threshold {DIVERGENCE_THRESHOLD:.0e} at iteration {it}"
            )
        
        if callback is not None:
            callback(it, x, F, res_norm)
        if res_norm < tol:
            success = True
            break

        J = jacobian(x)
        delta = solve_linear(J, -F)

        step = damping
        if line_search is not None:
            step = line_search(x, delta, F)

        x = x + step * delta

    return NewtonResult(
        x=x,
        success=success,
        iterations=len(history),
        residual_norm=history[-1],
        history=history,
    )


__all__ = ["NewtonResult", "newton_solve", "NewtonConvergenceError"]

