"""
Profiles and unstable mode computation for Q-balls/Q-clouds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from . import bounce_solver, potentials
from .bounce_solver import BounceSolution

ArrayLike = np.ndarray | float


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility helper: use np.trapezoid when available, otherwise np.trapz."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class QBallProfile:
    """Radial Q-ball profile χ(r) obtained from the 1D bounce solver."""

    solution: BounceSolution
    omega: float
    params: potentials.LogisticPotentialParams

    @property
    def r(self) -> np.ndarray:
        return self.solution.r

    @property
    def chi(self) -> np.ndarray:
        return self.solution.phi

    @property
    def dchi_dr(self) -> np.ndarray:
        return self.solution.phip

    @property
    def phi_abs(self) -> np.ndarray:
        """Physical |φ(r)| used in the 2D construction (χ = √2 |φ|)."""
        return self.solution.phi / np.sqrt(2.0)

    @property
    def rho(self) -> np.ndarray:
        """Field invariant s = |φ|²."""
        return 0.5 * self.solution.phi**2


@dataclass
class UnstableMode:
    """Unstable eigenmode ξ₋(r) around the Q-cloud background."""

    r: np.ndarray
    xi_real: np.ndarray
    xi_imag: np.ndarray
    gamma: float
    gamma_grid: Optional[np.ndarray] = None
    Dp_vals: Optional[np.ndarray] = None

    @property
    def xi_complex(self) -> np.ndarray:
        return self.xi_real + 1j * self.xi_imag


def solve_qball_profile(
    params: potentials.LogisticPotentialParams,
    omega: float,
    *,
    x_min: float = -1.0,
    x_max: float = 6.0,
    phi0_cap: float = 10.0,
    prefer_side: str = "+",
    r0: float = 1e-6,
    rmax: float = 200.0,
    max_step: float = 0.1,
    verbose: bool = False,
) -> QBallProfile:
    """
    Compute the radial Q-ball profile χ(r) by solving the 1D bounce for the
    effective potential V̂(χ) = V(χ) - ½ ω² χ².
    """
    V_hat, dV_hat = potentials.effective_qball_potential(params, omega)
    solution = bounce_solver.solve_bounce(
        V_hat,
        dV=dV_hat,
        d=3,
        r0=r0,
        rmax=rmax,
        max_step=max_step,
        x_min=x_min,
        x_max=x_max,
        tol=1e-8,
        verbose=verbose,
        allow_unbounded=True,
        phi0_cap=phi0_cap,
        prefer_side=prefer_side,
    )
    return QBallProfile(solution=solution, omega=float(omega), params=params)


def _logistic_V_derivatives(
    params: potentials.LogisticPotentialParams, s: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return V′(s) and V″(s) for s = |φ|² using the logistic potential parameters.
    """
    m, v, b = params.m, params.v, params.b
    s_arr = np.asarray(s, dtype=float)
    s_arr = np.maximum(s_arr, 0.0)

    c = np.exp(-b) / (1.0 + np.exp(-b))
    logc = np.log(c)
    t = (b * s_arr / (v * v)) + logc
    t_clip = np.clip(t, -50.0, 50.0)

    exp_t = np.exp(t_clip)
    denom = 1.0 + exp_t

    V_prime = (m * m) / denom
    sigma = 1.0 / denom
    V_second = - (m * m) * (b / (v * v)) * sigma * (1.0 - sigma)
    return V_prime, V_second


def compute_unstable_mode(
    profile: QBallProfile,
    *,
    gamma_scan: Tuple[float, float] = (0.01, 0.2),
    n_scan: int = 60,
) -> UnstableMode:
    """Unstable mode around the Q-cloud profile (shooting method)."""
    omega = profile.omega
    r_bg = profile.r
    rho_bg = profile.rho

    V1_bg, V2_bg = _logistic_V_derivatives(profile.params, rho_bg)

    # Interpolants over the background grid
    V1 = interp1d(r_bg, V1_bg, kind="cubic", fill_value="extrapolate")
    V2 = interp1d(r_bg, V2_bg, kind="cubic", fill_value="extrapolate")
    rho_interp = interp1d(r_bg, rho_bg, kind="cubic", fill_value="extrapolate")

    def background_coeffs(r: float) -> Tuple[float, float, float]:
        return float(V1(r)), float(V2(r)), float(rho_interp(r))

    def xi_ode(r: float, y: np.ndarray, gamma: float) -> np.ndarray:
        xi_R, dxi_R, xi_I, dxi_I = y
        V1_val, V2_val, rho_val = background_coeffs(r)
        coeff = (gamma * gamma - omega * omega) + V1_val
        extra = 2.0 * V2_val * rho_val
        xi_R2 = (coeff + extra) * xi_R - 2.0 * omega * gamma * xi_I
        xi_I2 = coeff * xi_I + 2.0 * omega * gamma * xi_R
        return np.array([dxi_R, xi_R2, dxi_I, xi_I2], dtype=float)

    r_min, r_max = float(r_bg[0]), float(r_bg[-1])
    t_eval = r_bg[::-1]

    def solve_basis(gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y0_A = np.array([1.0, 0.0, 0.0, 0.0])
        sol_A = solve_ivp(
            lambda r, y: xi_ode(r, y, gamma),
            t_span=(r_max, r_min),
            y0=y0_A,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )

        y0_B = np.array([0.0, 0.0, 1.0, 0.0])
        sol_B = solve_ivp(
            lambda r, y: xi_ode(r, y, gamma),
            t_span=(r_max, r_min),
            y0=y0_B,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )

        if not (sol_A.success and sol_B.success):
            raise RuntimeError(f"ODE solver failed for gamma = {gamma}")

        xi_R_A = sol_A.y[0, ::-1]
        xi_I_A = sol_A.y[2, ::-1]
        xi_R_B = sol_B.y[0, ::-1]
        xi_I_B = sol_B.y[2, ::-1]
        return xi_R_A, xi_I_A, xi_R_B, xi_I_B

    def Dp_of_gamma(gamma: float) -> float:
        xi_R_A, xi_I_A, xi_R_B, xi_I_B = solve_basis(gamma)
        xi_R_A0 = xi_R_A[0]
        xi_I_A0 = xi_I_A[0]
        xi_R_B0 = xi_R_B[0]
        xi_I_B0 = xi_I_B[0]
        return xi_R_A0 * xi_I_B0 - xi_R_B0 * xi_I_A0

    gamma_grid = np.linspace(gamma_scan[0], gamma_scan[1], n_scan)
    D_vals = np.array([Dp_of_gamma(g) for g in gamma_grid])

    bracket = None
    for i in range(len(gamma_grid) - 1):
        if np.sign(D_vals[i]) != np.sign(D_vals[i + 1]):
            bracket = (gamma_grid[i], gamma_grid[i + 1])
            break
    if bracket is None:
        raise RuntimeError("Could not bracket negative mode eigenvalue; try adjusting scan range.")

    root = root_scalar(Dp_of_gamma, bracket=bracket, method="brentq", xtol=1e-8, rtol=1e-8)
    if not root.converged:
        raise RuntimeError("Root finding for gamma did not converge.")
    gamma_star = float(root.root)

    xi_R_A, xi_I_A, xi_R_B, xi_I_B = solve_basis(gamma_star)
    xi_R_A0 = xi_R_A[0]
    xi_R_B0 = xi_R_B[0]
    c_A = xi_R_B0
    c_B = -xi_R_A0

    xi_R = c_A * xi_R_A + c_B * xi_R_B
    xi_I = c_A * xi_I_A + c_B * xi_I_B

    norm_integrand = r_bg**2 * (xi_R**2 + xi_I**2)
    norm = np.sqrt(_trapz(norm_integrand, r_bg))
    xi_R /= norm
    xi_I /= norm

    return UnstableMode(
        r=r_bg.copy(), 
        xi_real=xi_R, 
        xi_imag=xi_I, 
        gamma=gamma_star,
        gamma_grid=gamma_grid.copy(),
        Dp_vals=D_vals.copy()
    )


__all__ = ["QBallProfile", "UnstableMode", "solve_qball_profile", "compute_unstable_mode"]

