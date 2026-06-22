"""Observables at fixed τ, including ghost reconstruction at τ = 0."""


import numpy as np
from .grid import RadialTimeGrid
from .potentials import LogisticPotentialParams, logistic_potential_rho


def compute_energy_tau0_ghost(
    y: np.ndarray,
    ybar: np.ndarray,
    grid: RadialTimeGrid,
    omega: float,
    eta0: float,
    potential_params: LogisticPotentialParams,
) -> float:
    """
    Compute E(τ=0) by ghost reconstruction.
    
    Uses paper-style energy functional (Eq. 2.4) with V(0)=0 shift,
    and tau-derivatives consistent with solver ghost rules (Eq. 4.2).
    
    The grid is staggered: τ_i = -(i+1/2)Δτ, so τ=0 is NOT a grid point.
    The slice closest to τ=0 is i=0: τ[0] = -Δτ/2.
    
    To evaluate at τ=0, we reconstruct the fields using the ghost rules:
    - At τ = +Δτ/2 (ghost point): y_plus = ybar[:, 0], ybar_plus = y[:, 0]
    - At τ = -Δτ/2 (grid point): y_minus = y[:, 0], ybar_minus = ybar[:, 0]
    - At τ = 0 (midpoint): y0 = 0.5*(y_plus + y_minus), ybar0 = 0.5*(ybar_plus + ybar_minus)
    - Derivative at τ=0: y_t0 = (y_plus - y_minus)/dt, ybar_t0 = (ybar_plus - ybar_minus)/dt
    
    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau)
    grid
        RadialTimeGrid instance
    omega
        Q-ball frequency used in the transformation y = r e^{-ωτ} φ
    eta0
        The eta used in tau-ghost BC (not used for i=0, but kept for consistency)
    potential_params
        LogisticPotentialParams instance defining the potential
    
    Returns
    -------
    float
        Energy E at τ=0
    """
    dt = grid.dtau
    r = grid.r
    
    # Get potential function and shift so V(0)=0
    V_rho, _, _ = logistic_potential_rho(potential_params)
    V0 = float(V_rho(0.0))
    def V_of_s(s):
        return V_rho(s) - V0
    
    # Radial derivative function (same stencil as compute_energy)
    def radial_derivative(f_i):
        df = np.empty_like(f_i, dtype=complex)
        df[0]  = (f_i[1] - f_i[0]) / (r[1] - r[0])
        df[-1] = (f_i[-1] - f_i[-2]) / (r[-1] - r[-2])
        df[1:-1] = (f_i[2:] - f_i[:-2]) / (r[2:] - r[:-2])
        return df
    
    # Ghost reconstruction at τ=0
    # At τ = -Δτ/2 (grid point i=0):
    y_minus = y[:, 0]
    ybar_minus = ybar[:, 0]
    
    # At τ = +Δτ/2 (ghost point, using ghost rules for i=0):
    # From Eq. 4.2: y_{-1} = ybar_0, ybar_{-1} = y_0
    y_plus = ybar[:, 0]
    ybar_plus = y[:, 0]
    
    # Midpoint values at τ=0:
    y0 = 0.5 * (y_plus + y_minus)
    ybar0 = 0.5 * (ybar_plus + ybar_minus)
    
    # Derivatives at τ=0 (central difference between ±Δτ/2):
    y_t0 = (y_plus - y_minus) / dt
    ybar_t0 = (ybar_plus - ybar_minus) / dt
    
    # Reconstruct fields at τ=0:
    # At τ=0, e^{ω*0} = 1, so φ(0) = y0/r, φ̄(0) = ybar0/r
    phi0 = y0 / r
    phibar0 = ybar0 / r
    
    # Reconstruct tau-derivatives of fields at τ=0:
    # φ_τ(0) = (y_t0 + omega*y0)/r, φ̄_τ(0) = (ybar_t0 - omega*ybar0)/r
    # At τ = 0 the rotation factor is unity
    phi_tau0 = (y_t0 + omega * y0) / r
    phibar_tau0 = (ybar_t0 - omega * ybar0) / r
    
    # Radial derivatives at τ=0:
    phi_r0 = radial_derivative(phi0)
    phibar_r0 = radial_derivative(phibar0)
    
    # Potential:
    s = (phi0 * phibar0).real
    s = np.maximum(s, 0.0)
    V = V_of_s(s)
    
    # Energy density (Eq. 2.4):
    # Sign matches compute_energy (Euclidean metric)
    dens = -(phi_tau0 * phibar_tau0) + (phi_r0 * phibar_r0) + V
    
    # Integrate:
    E0 = float(4.0 * np.pi * np.trapz((r**2) * dens.real, r))
    
    return E0
