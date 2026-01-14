"""
Computation of Euclidean action and suppression exponent for semiclassical decay rate.

This module provides functions to compute the Euclidean action S_E and the suppression
exponent F_{Q,beta} used in the semiclassical decay rate Γ ~ exp(-F_{Q,beta}).

The 2D solver works on the half Euclidean-time interval: -beta/2 < tau < 0 with a
staggered grid. The full action S_E[phi_cl] over (-beta/2, beta/2) is obtained by
doubling the half-interval result using time-reflection symmetry.

Two methods are provided for computing F_{Q,beta}:
1. Legacy method: F = S_E[phi_cl] - beta*(E_Q - omega*Q) + eta0*Q
   (suffers from large cancellations and discretization mismatches)
2. Direct difference method: F = 2*∫[L_E(phi_cl) - L_E(phi_Q)] + eta0*Q
   (numerically stable, avoids cancellations by computing difference on same grid)
"""

import numpy as np
from typing import Callable
from .grid import RadialTimeGrid
from .potentials import LogisticPotentialParams, logistic_potential_rho


def _radial_derivative_stencil(f_i: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute radial derivative using the standard stencil:
    - Central differences in bulk: df[i] = (f[i+1] - f[i-1]) / (r[i+1] - r[i-1])
    - One-sided at boundaries
    
    This is the same stencil used in compute_energy and compute_euclidean_action_half.
    """
    df = np.empty_like(f_i, dtype=complex)
    df[0]  = (f_i[1] - f_i[0]) / (r[1] - r[0])
    df[-1] = (f_i[-1] - f_i[-2]) / (r[-1] - r[-2])
    df[1:-1] = (f_i[2:] - f_i[:-2]) / (r[2:] - r[:-2])
    return df


def compute_euclidean_action_half(
    y: np.ndarray,
    ybar: np.ndarray,
    grid: RadialTimeGrid,
    omega: float,
    eta0: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Compute Euclidean action S_E over the HALF interval (-beta/2, 0).
    
    Formula:
      S_half = ∫_{-beta/2}^{0} dτ 4π ∫ dr r² [ (∂τ φ)(∂τ φ̄) + (∂r φ)(∂r φ̄) + V(φ φ̄) ]
    
    Uses the EXACT SAME tau ghost rules as the solver residual/jacobian (Eq. 4.2).
    Integrates in r using np.trapz with 4π r², and in tau using simple Riemann sum
    (rectangle rule) with constant dtau.
    
    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau) representing the fields in y-variables
    grid
        RadialTimeGrid instance
    omega
        Q-ball frequency used in the transformation y = r e^{-ωτ} φ
    eta0
        The eta used in tau-ghost BC at i=Nt-1
    V_of_s
        Potential function V(s) where s = |φ|², already shifted so V(0)=0
    
    Returns
    -------
    float
        Euclidean action S_E over the half interval (-beta/2, 0)
    """
    Nr, Nt = grid.Nr, grid.Ntau
    dt = grid.dtau
    r = grid.r
    tau = grid.tau
    
    # Use shared radial derivative stencil
    def radial_derivative(f_i):
        return _radial_derivative_stencil(f_i, r)
    
    # Storage for action density at each tau slice
    S_density_tau = np.zeros(Nt, dtype=float)
    
    for i in range(Nt):
        # Reconstruct phi, phibar from y, ybar
        phi_i    = np.exp(omega * tau[i])  * (y[:, i]    / r)
        phibar_i = np.exp(-omega * tau[i]) * (ybar[:, i] / r)
        
        # Compute tau derivatives using EXACT SAME ghost rules as compute_energy
        if i == 0:
            y_im1  = ybar[:, 0]
            y_ip1  = y[:, 1]
            yb_im1 = y[:, 0]
            yb_ip1 = ybar[:, 1]
        elif i == Nt - 1:
            y_im1  = y[:, Nt - 2]
            y_ip1  = np.exp(-eta0) * ybar[:, Nt - 1]
            yb_im1 = ybar[:, Nt - 2]
            yb_ip1 = np.exp(+eta0) * y[:, Nt - 1]
        else:
            y_im1  = y[:, i - 1]
            y_ip1  = y[:, i + 1]
            yb_im1 = ybar[:, i - 1]
            yb_ip1 = ybar[:, i + 1]
        
        # Central differences for y_t, ybar_t
        y_t  = (y_im1  - y_ip1)  / (2.0 * dt)
        yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)
        
        # Convert to phi_tau, phibar_tau
        phi_tau_i    = (np.exp(omega * tau[i])  / r) * (y_t  + omega * y[:, i])
        phibar_tau_i = (np.exp(-omega * tau[i]) / r) * (yb_t - omega * ybar[:, i])
        
        # Radial derivatives
        phi_r_i    = radial_derivative(phi_i)
        phibar_r_i = radial_derivative(phibar_i)
        
        # Potential
        s = (phi_i * phibar_i).real
        s = np.maximum(s, 0.0)
        V = V_of_s(s)
        
        # Action density (note: positive sign for tau term, unlike energy)
        # S_density = (∂τ φ)(∂τ φ̄) + (∂r φ)(∂r φ̄) + V(φ φ̄)
        dens = (phi_tau_i * phibar_tau_i) + (phi_r_i * phibar_r_i) + V
        
        # Integrate in r: 4π ∫ dr r² * dens
        S_density_tau[i] = float(4.0 * np.pi * np.trapz((r**2) * dens.real, r))
    
    # Integrate in tau: simple Riemann sum (rectangle rule)
    S_half = float(np.sum(S_density_tau) * dt)
    
    return S_half


def compute_euclidean_action_full(
    y: np.ndarray,
    ybar: np.ndarray,
    grid: RadialTimeGrid,
    omega: float,
    eta0: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Compute Euclidean action S_E over the FULL interval (-beta/2, beta/2).
    
    Uses time-reflection symmetry: S_full = 2 * S_half, since the solver only
    stores the half interval tau < 0.
    
    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau)
    grid
        RadialTimeGrid instance
    omega
        Q-ball frequency
    eta0
        The eta used in tau-ghost BC
    V_of_s
        Potential function V(s) where s = |φ|², already shifted so V(0)=0
    
    Returns
    -------
    float
        Euclidean action S_E over the full interval (-beta/2, beta/2)
    """
    S_half = compute_euclidean_action_half(y, ybar, grid, omega, eta0, V_of_s)
    return 2.0 * S_half


def compute_qball_action(
    beta: float,
    E_Q: float,
    Q: float,
    omega: float,
) -> float:
    """
    Compute Euclidean action of the Q-ball configuration.
    
    Formula: S_E[phi_Q] = β * (E_Q - ω*Q)
    
    Parameters
    ----------
    beta
        Inverse temperature
    E_Q
        Q-ball energy
    Q
        Q-ball charge
    omega
        Q-ball frequency
    
    Returns
    -------
    float
        Euclidean action of the Q-ball configuration
    """
    return beta * (E_Q - omega * Q)


def compute_suppression_exponent(
    y: np.ndarray,
    ybar: np.ndarray,
    grid: RadialTimeGrid,
    omega: float,
    eta0: float,
    beta: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
    E_Q: float,
    Q: float,
) -> float:
    """
    Compute suppression exponent F_{Q,beta} for the semiclassical decay rate.
    
    Formula: F_{Q,beta} = S_E[phi_cl] - S_E[phi_Q] + η0*Q
                    = S_E_full - β*(E_Q - ω*Q) + η0*Q
    
    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau) representing the 2D bounce solution
    grid
        RadialTimeGrid instance
    omega
        Q-ball frequency
    eta0
        The eta used in tau-ghost BC (from solution settings)
    beta
        Inverse temperature (from solution settings)
    V_of_s
        Potential function V(s) where s = |φ|², already shifted so V(0)=0
    E_Q
        Q-ball energy
    Q
        Q-ball charge (should match target charge)
    
    Returns
    -------
    float
        Suppression exponent F_{Q,beta}
    """
    S_full = compute_euclidean_action_full(y, ybar, grid, omega, eta0, V_of_s)
    S_Q = compute_qball_action(beta, E_Q, Q, omega)
    return S_full - S_Q + eta0 * Q


def compute_activation_exponent(
    beta: float,
    E_cloud: float,
    E_Q: float,
) -> float:
    """
    Compute activation/Q-cloud exponent used in Fig. 7(b).
    
    Formula: F_cloud = β * (E_cloud - E_Q)
    
    Parameters
    ----------
    beta
        Inverse temperature
    E_cloud
        Q-cloud energy at the same conserved charge Q
    E_Q
        Q-ball energy
    
    Returns
    -------
    float
        Activation exponent F_cloud
    """
    return beta * (E_cloud - E_Q)


def compute_suppression_exponent_direct_difference(
    y: np.ndarray,
    ybar: np.ndarray,
    grid: RadialTimeGrid,
    omega: float,
    eta0: float,
    beta: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
    qball_profile,
    Q_target: float,
    *,
    debug_check_legacy: bool = False,
) -> float:
    """
    Compute suppression exponent F_{Q,beta} using direct difference of Lagrangian densities.
    
    This method is numerically stable and avoids large cancellations by computing the
    difference L_E(phi_cl) - L_E(phi_Q) on the SAME grid with the SAME stencils, subtracting
    the Q-ball reference BEFORE integrating in tau.
    
    Formula:
      F_{Q,beta} = 2 * ∫_{-beta/2}^0 dτ 4π ∫ dr r² [ L_E(phi_cl) - L_E(phi_Q) ] + η0*Q_target
    
    where:
      L_E(phi) = (∂τ φ)(∂τ φ̄) + (∂r φ)(∂r φ̄) + V(|φ|²)
    
    The Q-ball reference is constructed from the 1D profile phi_abs_Q(r) = |φ(r)|:
      phi_Q(r, τ)    = exp(omega * τ) * phi_abs_Q(r)
      phibar_Q(r, τ) = exp(-omega * τ) * phi_abs_Q(r)
    
    For the stationary Q-ball, the tau-derivative term is:
      (∂τ φ_Q)(∂τ φ̄_Q) = -omega² * phi_abs_Q² = -omega² * s_Q
    and the radial derivative term simplifies (exp factors cancel in product).
    
    The Q-ball Lagrangian density is tau-independent and computed ONCE, then subtracted
    from each bounce slice before integrating in tau.
    
    For phi_cl: uses EXACT SAME tau-ghost rules as solver residual/jacobian.
    Both use the SAME radial derivative stencil.
    
    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau) representing the 2D bounce solution
    grid
        RadialTimeGrid instance
    omega
        Q-ball frequency
    eta0
        The eta used in tau-ghost BC at i=Nt-1
    beta
        Inverse temperature
    V_of_s
        Potential function V(s) where s = |φ|², already shifted so V(0)=0
    qball_profile
        QBallProfile instance. Uses qball_profile.phi_abs (physical |φ|) if available,
        otherwise falls back to qball_profile.chi / sqrt(2).
    Q_target
        Target conserved charge (used in the eta scan)
    debug_check_legacy
        Reserved for future use. Currently unused (debug checks should be done at call site).
    
    Returns
    -------
    float
        Suppression exponent F_{Q,beta} computed via direct difference method.
        No median/trim is used; returns the actual computed value.
    """
    Nr, Nt = grid.Nr, grid.Ntau
    dt = grid.dtau
    r = grid.r
    tau = grid.tau
    
    # Get phi_abs from qball_profile (physical 2D field amplitude |φ|)
    if hasattr(qball_profile, 'phi_abs'):
        phi_abs_Q_1d = qball_profile.phi_abs
    else:
        # Fallback: chi / sqrt(2) if phi_abs not available
        phi_abs_Q_1d = qball_profile.chi / np.sqrt(2.0)
    
    # Interpolate Q-ball profile onto grid.r, zero outside support
    phi_abs_Q = np.interp(r, qball_profile.r, phi_abs_Q_1d, left=0.0, right=0.0)  # (Nr,)
    
    # ===== Compute Q-ball reference slice density (tau-independent) =====
    s_Q = phi_abs_Q**2  # |φ|²
    
    # Tau-derivative term: (∂τ φ_Q)(∂τ φ̄_Q) = -omega² * s_Q
    tau_term_Q = -omega**2 * s_Q
    
    # Radial derivative: ∂r phi_abs_Q (real, since phi_abs_Q is real)
    phi_abs_Q_r = _radial_derivative_stencil(phi_abs_Q.astype(complex), r).real
    # For stationary Q-ball: (∂r φ_Q)(∂r φ̄_Q) = (∂r phi_abs_Q)² (exp factors cancel)
    radial_term_Q = phi_abs_Q_r**2
    
    # Potential
    V_Q = V_of_s(s_Q)
    
    # Q-ball Lagrangian density (tau-independent)
    L_Q = tau_term_Q + radial_term_Q + V_Q
    
    # Integrate Q-ball slice density in r
    Lslice_Q = float(4.0 * np.pi * np.trapz((r**2) * L_Q, r))
    
    # ===== Compute bounce slice densities and subtract Q-ball reference =====
    Lslice_diff = np.zeros(Nt, dtype=float)
    
    for i in range(Nt):
        # Reconstruct phi_cl, phibar_cl from y, ybar (EXACT SAME as compute_euclidean_action_half)
        phi_cl_i    = np.exp(omega * tau[i])  * (y[:, i]    / r)
        phibar_cl_i = np.exp(-omega * tau[i]) * (ybar[:, i] / r)
        
        # Compute tau derivatives using EXACT SAME ghost rules as solver
        if i == 0:
            y_im1  = ybar[:, 0]
            y_ip1  = y[:, 1]
            yb_im1 = y[:, 0]
            yb_ip1 = ybar[:, 1]
        elif i == Nt - 1:
            y_im1  = y[:, Nt - 2]
            y_ip1  = np.exp(-eta0) * ybar[:, Nt - 1]
            yb_im1 = ybar[:, Nt - 2]
            yb_ip1 = np.exp(+eta0) * y[:, Nt - 1]
        else:
            y_im1  = y[:, i - 1]
            y_ip1  = y[:, i + 1]
            yb_im1 = ybar[:, i - 1]
            yb_ip1 = ybar[:, i + 1]
        
        # Central differences for y_t, ybar_t
        y_t  = (y_im1  - y_ip1)  / (2.0 * dt)
        yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)
        
        # Convert to phi_tau, phibar_tau
        phi_tau_cl_i    = (np.exp(omega * tau[i])  / r) * (y_t  + omega * y[:, i])
        phibar_tau_cl_i = (np.exp(-omega * tau[i]) / r) * (yb_t - omega * ybar[:, i])
        
        # Radial derivatives
        phi_r_cl_i    = _radial_derivative_stencil(phi_cl_i, r)
        phibar_r_cl_i = _radial_derivative_stencil(phibar_cl_i, r)
        
        # Potential: s = |φ|²
        s_cl = (phi_cl_i * phibar_cl_i).real
        s_cl = np.maximum(s_cl, 0.0)
        V_cl = V_of_s(s_cl)
        
        # Lagrangian density for phi_cl
        L_E_cl = (phi_tau_cl_i * phibar_tau_cl_i) + (phi_r_cl_i * phibar_r_cl_i) + V_cl
        
        # Integrate in r to get slice density
        Lslice_cl = float(4.0 * np.pi * np.trapz((r**2) * L_E_cl.real, r))
        
        # Subtract Q-ball reference (computed once, tau-independent)
        Lslice_diff[i] = Lslice_cl - Lslice_Q
    
    # Integrate in tau: simple Riemann sum (rectangle rule) over half interval
    F_half = float(np.sum(Lslice_diff) * dt)
    
    # Multiply by 2 (time reflection) and add eta0 * Q_target
    F_full = 2.0 * F_half + eta0 * Q_target
    
    return F_full


def make_V_of_s_from_params(
    params: LogisticPotentialParams,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a V_of_s function from LogisticPotentialParams, shifted so V(0)=0.
    
    This is a convenience wrapper for notebook use.
    
    Parameters
    ----------
    params
        LogisticPotentialParams instance
    
    Returns
    -------
    Callable
        Function V_of_s(s) where s = |φ|², with V(0)=0
    """
    V_rho, _, _ = logistic_potential_rho(params)
    V0 = float(V_rho(0.0))
    def V_of_s(s):
        return V_rho(s) - V0
    return V_of_s
