"""Euclidean action and fixed-charge suppression exponent F_{Q,β}.

Half-interval action is doubled by τ-reflection symmetry. Reference state: homogeneous
charged background with Legendre term η₀ Q.
"""


from __future__ import annotations

from typing import Callable

import numpy as np


def _radial_derivative_stencil(f_i: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Radial derivative with standard stencil: central in bulk, one-sided at boundaries.
    Same convention as in observables_2d and the solver.
    """
    f_i = np.asarray(f_i)
    r = np.asarray(r, dtype=float).flatten()
    df = np.empty_like(f_i, dtype=np.result_type(f_i, np.complex128))
    if r.size < 2:
        df[:] = 0.0
        return df
    df[0] = (f_i[1] - f_i[0]) / (r[1] - r[0])
    df[-1] = (f_i[-1] - f_i[-2]) / (r[-1] - r[-2])
    if f_i.size > 2:
        df[1:-1] = (f_i[2:] - f_i[:-2]) / (r[2:] - r[:-2])
    return df


def compute_euclidean_action_half(
    y: np.ndarray,
    ybar: np.ndarray,
    grid,
    omega: float,
    eta0: float,
    rho0: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Compute Euclidean action S_E over the HALF interval (-β/2, 0).

    Formula:
      S_half = ∫_{-β/2}^{0} dτ  4π ∫ dr r² [ (∂τ φ)(∂τ φ̄) + (∂r φ)(∂r φ̄) + V(φ φ̄) ]

    Uses the same tau ghost rules as the bubble solver (reflection at τ=0, twist at τ=-β/2).
    Field reconstruction: φ = exp(+ω τ)(y/r + ρ₀), φ̄ = exp(-ω τ)(ȳ/r + ρ₀), with y_tot = y + r ρ₀
    for the twist at the last tau index.

    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau): fluctuation fields (solver representation).
    grid
        Object with .r, .tau, .Nr, .Ntau, .dtau (e.g. solver.grid).
    omega
        Phase frequency (chemical potential) of the homogeneous state.
    eta0
        Twist parameter used in tau-ghost BC at i = Ntau-1.
    rho0
        Homogeneous background ρ₀ (solver.rho0).
    V_of_s
        Potential V(s) with s = φ φ̄ (real), typically shifted so V(0)=0.

    Returns
    -------
    float
        S_E over the half interval (-β/2, 0).
    """
    Nr = int(getattr(grid, "Nr", grid.Ntau if hasattr(grid, "Ntau") else 0))
    Nt = int(getattr(grid, "Ntau", getattr(grid, "Nt", 0)))
    dt = float(getattr(grid, "dtau", getattr(grid, "dt", None)))
    r = np.asarray(grid.r, dtype=float).flatten()
    tau = np.asarray(grid.tau, dtype=float).flatten()
    if r.size != Nr or tau.size != Nt or dt <= 0:
        raise ValueError("rate_exponent: invalid grid (r, tau, dtau).")
    if y.shape != (Nr, Nt) or ybar.shape != (Nr, Nt):
        raise ValueError("rate_exponent: y, ybar must have shape (Nr, Ntau).")

    S_density_tau = np.zeros(Nt, dtype=float)

    for i in range(Nt):
        # Reconstruct phi, phibar: phi = exp(+ωτ)(y/r + rho0), phibar = exp(-ωτ)(ybar/r + rho0)
        inv_r = np.where(r > 0, 1.0 / r, 0.0)
        phi_i = np.exp(omega * tau[i]) * (y[:, i] * inv_r + rho0)
        phibar_i = np.exp(-omega * tau[i]) * (ybar[:, i] * inv_r + rho0)

        # Tau derivatives: same ghost rules as bounce2d._tau_neighbors (twisted)
        if i == 0:
            y_im1 = ybar[:, 0]
            y_ip1 = y[:, 1]
            yb_im1 = y[:, 0]
            yb_ip1 = ybar[:, 1]
        elif i == Nt - 1:
            y_tot_n = y[:, Nt - 1] + r * rho0
            yb_tot_n = ybar[:, Nt - 1] + r * rho0
            y_im1 = y[:, Nt - 2]
            yb_im1 = ybar[:, Nt - 2]
            y_ip1 = np.exp(-eta0) * yb_tot_n - r * rho0
            yb_ip1 = np.exp(eta0) * y_tot_n - r * rho0
        else:
            y_im1 = y[:, i - 1]
            y_ip1 = y[:, i + 1]
            yb_im1 = ybar[:, i - 1]
            yb_ip1 = ybar[:, i + 1]

        # Central d/dτ: grid tau decreases with index, so y_t ~ (y_im1 - y_ip1)/(2 dt)
        y_t = (y_im1 - y_ip1) / (2.0 * dt)
        yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)

        # φ_τ, φ̄_τ from y_t and chain rule
        phi_tau_i = (np.exp(omega * tau[i]) * inv_r) * (y_t + omega * (y[:, i] + r * rho0))
        phibar_tau_i = (np.exp(-omega * tau[i]) * inv_r) * (yb_t - omega * (ybar[:, i] + r * rho0))

        phi_r_i = _radial_derivative_stencil(phi_i, r)
        phibar_r_i = _radial_derivative_stencil(phibar_i, r)

        s = (phi_i * phibar_i).real
        s = np.maximum(s, 0.0)
        V = V_of_s(s)

        # Euclidean Lagrangian density: (∂τ φ)(∂τ φ̄) + (∂r φ)(∂r φ̄) + V
        dens = (phi_tau_i * phibar_tau_i) + (phi_r_i * phibar_r_i) + V
        S_density_tau[i] = float(4.0 * np.pi * np.trapz((r ** 2) * dens.real, r))

    S_half = float(np.sum(S_density_tau) * dt)
    return S_half


def compute_euclidean_action_half_breakdown(
    y: np.ndarray,
    ybar: np.ndarray,
    grid,
    omega: float,
    eta0: float,
    rho0: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
):
    """
    Same as compute_euclidean_action_half but returns a dict with each term separately:
      S_kin_tau: ∫ dτ 4π ∫ dr r² (∂τ φ)(∂τ φ̄)
      S_kin_r:   ∫ dτ 4π ∫ dr r² (∂r φ)(∂r φ̄)
      S_pot:     ∫ dτ 4π ∫ dr r² V(φφ̄)
      S_half:    S_kin_tau + S_kin_r + S_pot
    Plus per-slice arrays: S_kin_tau_slice, S_kin_r_slice, S_pot_slice (before * dt).
    """
    Nr = int(getattr(grid, "Nr", grid.Ntau if hasattr(grid, "Ntau") else 0))
    Nt = int(getattr(grid, "Ntau", getattr(grid, "Nt", 0)))
    dt = float(getattr(grid, "dtau", getattr(grid, "dt", None)))
    r = np.asarray(grid.r, dtype=float).flatten()
    tau = np.asarray(grid.tau, dtype=float).flatten()
    if r.size != Nr or tau.size != Nt or dt <= 0:
        raise ValueError("rate_exponent: invalid grid (r, tau, dtau).")
    if y.shape != (Nr, Nt) or ybar.shape != (Nr, Nt):
        raise ValueError("rate_exponent: y, ybar must have shape (Nr, Ntau).")

    S_kin_tau_slice = np.zeros(Nt, dtype=float)
    S_kin_r_slice = np.zeros(Nt, dtype=float)
    S_pot_slice = np.zeros(Nt, dtype=float)

    for i in range(Nt):
        inv_r = np.where(r > 0, 1.0 / r, 0.0)
        phi_i = np.exp(omega * tau[i]) * (y[:, i] * inv_r + rho0)
        phibar_i = np.exp(-omega * tau[i]) * (ybar[:, i] * inv_r + rho0)

        if i == 0:
            y_im1 = ybar[:, 0]
            y_ip1 = y[:, 1]
            yb_im1 = y[:, 0]
            yb_ip1 = ybar[:, 1]
        elif i == Nt - 1:
            y_tot_n = y[:, Nt - 1] + r * rho0
            yb_tot_n = ybar[:, Nt - 1] + r * rho0
            y_im1 = y[:, Nt - 2]
            yb_im1 = ybar[:, Nt - 2]
            y_ip1 = np.exp(-eta0) * yb_tot_n - r * rho0
            yb_ip1 = np.exp(eta0) * y_tot_n - r * rho0
        else:
            y_im1 = y[:, i - 1]
            y_ip1 = y[:, i + 1]
            yb_im1 = ybar[:, i - 1]
            yb_ip1 = ybar[:, i + 1]

        y_t = (y_im1 - y_ip1) / (2.0 * dt)
        yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)

        phi_tau_i = (np.exp(omega * tau[i]) * inv_r) * (y_t + omega * (y[:, i] + r * rho0))
        phibar_tau_i = (np.exp(-omega * tau[i]) * inv_r) * (yb_t - omega * (ybar[:, i] + r * rho0))

        phi_r_i = _radial_derivative_stencil(phi_i, r)
        phibar_r_i = _radial_derivative_stencil(phibar_i, r)

        s = (phi_i * phibar_i).real
        s = np.maximum(s, 0.0)
        V = V_of_s(s)

        kin_tau = (phi_tau_i * phibar_tau_i).real
        kin_r = (phi_r_i * phibar_r_i).real
        S_kin_tau_slice[i] = float(4.0 * np.pi * np.trapz((r ** 2) * kin_tau, r))
        S_kin_r_slice[i] = float(4.0 * np.pi * np.trapz((r ** 2) * kin_r, r))
        S_pot_slice[i] = float(4.0 * np.pi * np.trapz((r ** 2) * V.real, r))

    dt_arr = dt
    S_kin_tau = float(np.sum(S_kin_tau_slice) * dt_arr)
    S_kin_r = float(np.sum(S_kin_r_slice) * dt_arr)
    S_pot = float(np.sum(S_pot_slice) * dt_arr)
    S_half = S_kin_tau + S_kin_r + S_pot

    return {
        "S_kin_tau": S_kin_tau,
        "S_kin_r": S_kin_r,
        "S_pot": S_pot,
        "S_half": S_half,
        "S_kin_tau_slice": S_kin_tau_slice,
        "S_kin_r_slice": S_kin_r_slice,
        "S_pot_slice": S_pot_slice,
        "tau": tau,
        "dt": dt,
    }


def compute_euclidean_action_full(
    y: np.ndarray,
    ybar: np.ndarray,
    grid,
    omega: float,
    eta0: float,
    rho0: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Euclidean action S_E over the full interval (-β/2, β/2).

    S_full = 2 * S_half by time-reflection symmetry (solver stores only τ < 0).
    """
    S_half = compute_euclidean_action_half(y, ybar, grid, omega, eta0, rho0, V_of_s)
    return 2.0 * S_half


def compute_homogeneous_action(
    beta: float,
    V_ball: float,
    omega: float,
    rho0: float,
    V_of_s: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Euclidean action of the homogeneous reference configuration.

    S_E[φ_hom] = β V_ball (V(s0) − ω² s0), with s0 = |φ|² = ρ₀²/2.

    Parameters
    ----------
    beta
        Inverse temperature.
    V_ball
        Spatial volume of the ball, 4π ∫ r² dr (same as in the bounce grid).
    omega
        Phase frequency (chemical potential) of the homogeneous state.
    rho0
        Homogeneous background ρ₀.
    V_of_s
        Potential V(s) with s = φφ̄ (real); typically shifted so V(0)=0.

    Returns
    -------
    float
        S_E[φ_hom].
    """
    s0 = max(0.0, float(rho0**2))
    V_at_rho0 = float(np.asarray(V_of_s(np.array([s0])), dtype=float).flat[0])
    return beta * V_ball * (V_at_rho0 - (omega**2) * (rho0**2))


def volume_from_grid(grid) -> float:
    """
    Ball volume consistent with the action integral: 4π ∫ r² dr over the grid r.

    Use this to get V_ball for compute_homogeneous_action when using the same grid as the bounce.
    """
    r = np.asarray(grid.r, dtype=float).flatten()
    if r.size < 2:
        return 0.0
    return float(4.0 * np.pi * np.trapz(r**2, r))


def compute_suppression_exponent_bubble(
    S_bounce_full: float,
    S_hom: float,
    eta0: float,
    Q: float,
) -> float:
    """
    Suppression exponent for bubble nucleation at fixed charge.

    F^{bounce}_{Q,β} = S_E[φ_b] - S_E[φ_hom] + η₀ Q.

    Parameters
    ----------
    S_bounce_full
        Full Euclidean action of the bounce (over (-β/2, β/2)).
    S_hom
        Euclidean action of the homogeneous reference, S_E[φ_hom] = β V_ball (V(s0)−ω² s0),
        with s0=|φ|²=ρ₀²/2.
    eta0
        Twist parameter (from the fixed-Q saddle).
    Q
        Conserved charge.

    Returns
    -------
    float
        F^{bounce}_{Q,β}.
    """
    return S_bounce_full - S_hom + eta0 * Q


def compute_activation_exponent_bubble(
    beta: float,
    E_crit: float,
    E_hom: float,
) -> float:
    """
    Activation exponent (thermal channel): F_act = β (E_crit − E_hom).

    E_crit is the energy of the critical bubble at fixed Q; E_hom is the energy of the
    homogeneous metastable state. Both must be computed without background subtraction
    (raw energies). At finite temperature the dominant decay is min(F^{bounce}, F_act).

    Parameters
    ----------
    beta
        Inverse temperature.
    E_crit
        Energy of the critical bubble, no background subtraction (e.g. 1D E_bubble).
    E_hom
        Energy of the homogeneous configuration, same volume, no background subtraction.

    Returns
    -------
    float
        F_act = β ΔE_act.
    """
    return beta * (E_crit - E_hom)


def make_V_of_s_from_U(U: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build V_of_s(s) from U(ρ) with s = |phi|² = ρ²/2 (so ρ = sqrt(2*s)).

    V_of_s(s) = U(sqrt(2*max(s,0))). No shift to V(0)=0; apply at call site if needed.
    """
    def V_of_s(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        rho = np.sqrt(2.0 * np.maximum(s, 0.0))
        return U(rho)
    return V_of_s
