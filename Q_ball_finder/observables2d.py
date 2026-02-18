"""
2D observables computation: charge and energy following paper conventions.

All functions compute values on a single tau slice (default: i=0, closest to τ=0).
"""

import numpy as np
from .grid import RadialTimeGrid
from .potentials import LogisticPotentialParams, logistic_potential_rho


def compute_charge(
    y: np.ndarray,
    ybar: np.ndarray,
    grid: RadialTimeGrid,
    omega: float,
    eta0: float,
    *,
    index_tau: int = 0,
    return_profile: bool = False,
):
    """
    Charge Q from Eq. (3.9):
      Q = ∫ d^3x [ phibar ∂τ phi - phi ∂τ phibar ].
    
    Uses SAME tau ghost rules as residual/jacobian (paper Eq. 4.2).
    
    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau)
    grid
        RadialTimeGrid instance
    omega
        Q-ball frequency used in the transformation y = r e^{-ωτ} φ
    eta0
        The eta used in tau-ghost BC at i=Nt-1
    index_tau
        Index of the tau slice to evaluate (default 0 = τ≈0, since τ0=-Δτ/2).
        Can use negative indexing (e.g., -1 for last slice).
    return_profile
        If True, return (Q_at_index, Q_tau_profile); otherwise return Q_at_index only.
    
    Returns
    -------
    If return_profile=False:
        float: Charge Q at the specified tau slice
    If return_profile=True:
        tuple: (Q_at_index, Q_tau_profile) where Q_tau_profile is array of shape (Ntau,)
    """
    Nr, Nt = grid.Nr, grid.Ntau
    dt = grid.dtau
    r = grid.r
    tau = grid.tau

    exp_wt = np.exp(omega * tau)[None, :]
    exp_mwt = np.exp(-omega * tau)[None, :]
    phi = exp_wt * (y / r[:, None])
    phibar = exp_mwt * (ybar / r[:, None])

    Q_tau = np.zeros(Nt, dtype=float)

    for i in range(Nt):
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

        y_t  = (y_im1  - y_ip1)  / (2.0 * dt)
        yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)

        dphi  = (np.exp(omega * tau[i])  / r) * (y_t  + omega * y[:, i])
        dphib = (np.exp(-omega * tau[i]) / r) * (yb_t - omega * ybar[:, i])

        density = phibar[:, i] * dphi - phi[:, i] * dphib
        Q_tau[i] = 4.0 * np.pi * np.trapz((r**2) * density.real, r)

    # Return chosen slice only (no median)
    i0 = index_tau if index_tau >= 0 else Nt + index_tau
    Q0 = float(Q_tau[i0])
    return (Q0, Q_tau) if return_profile else Q0


def compute_energy(
    y: np.ndarray,
    ybar: np.ndarray,
    grid: RadialTimeGrid,
    omega: float,
    eta0: float,
    potential_params: LogisticPotentialParams,
    *,
    index_tau: int = 0,
    return_profile: bool = False,
):
    """
    Energy from Eq. (2.4) evaluated on a tau slice. Density (same convention as paper):
      dens = -(∂τ φ)(∂τ φ̄) + (∂r φ)(∂r φ̄) + V(φ φ̄)
      E(τ_i) = 4π ∫ dr r^2 dens.

    The MINUS sign in front of the time-derivative product is the correct convention;
    2D observables must match this exactly (not a bug).

    Implementation uses the same tau ghost rules (Eq. 4.2) to compute tau-derivatives.
    Potential is shifted so V(0)=0, consistent with the legacy energy code.
    
    Parameters
    ----------
    y, ybar
        Complex arrays of shape (Nr, Ntau)
    grid
        RadialTimeGrid instance
    omega
        Q-ball frequency used in the transformation y = r e^{-ωτ} φ
    eta0
        The eta used in tau-ghost BC at i=Nt-1
    potential_params
        LogisticPotentialParams instance defining the potential
    index_tau
        Index of the tau slice to evaluate (default 0 = τ≈0, since τ0=-Δτ/2).
        Can use negative indexing (e.g., -1 for last slice).
    return_profile
        If True, return (E_at_index, E_tau_profile); otherwise return E_at_index only.
    
    Returns
    -------
    If return_profile=False:
        float: Energy E at the specified tau slice
    If return_profile=True:
        tuple: (E_at_index, E_tau_profile) where E_tau_profile is array of shape (Ntau,)
    """
    Nr, Nt = grid.Nr, grid.Ntau
    dt = grid.dtau
    r = grid.r
    tau = grid.tau

    V_rho, _, _ = logistic_potential_rho(potential_params)
    V0 = float(V_rho(0.0))
    def V_of_s(s):
        return V_rho(s) - V0

    def radial_derivative(f_i):
        df = np.empty_like(f_i, dtype=complex)
        df[0]  = (f_i[1] - f_i[0]) / (r[1] - r[0])
        df[-1] = (f_i[-1] - f_i[-2]) / (r[-1] - r[-2])
        df[1:-1] = (f_i[2:] - f_i[:-2]) / (r[2:] - r[:-2])
        return df

    E_tau = np.zeros(Nt, dtype=float)

    for i in range(Nt):
        phi_i    = np.exp(omega * tau[i])  * (y[:, i]    / r)
        phibar_i = np.exp(-omega * tau[i]) * (ybar[:, i] / r)

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

        y_t  = (y_im1  - y_ip1)  / (2.0 * dt)
        yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)

        phi_tau_i    = (np.exp(omega * tau[i])  / r) * (y_t  + omega * y[:, i])
        phibar_tau_i = (np.exp(-omega * tau[i]) / r) * (yb_t - omega * ybar[:, i])

        phi_r_i    = radial_derivative(phi_i)
        phibar_r_i = radial_derivative(phibar_i)

        s = (phi_i * phibar_i).real
        s = np.maximum(s, 0.0)
        V = V_of_s(s)

        dens = -(phi_tau_i * phibar_tau_i) + (phi_r_i * phibar_r_i) + V
        E_tau[i] = float(4.0 * np.pi * np.trapz((r**2) * dens.real, r))

    i0 = index_tau if index_tau >= 0 else Nt + index_tau
    E0 = float(E_tau[i0])
    return (E0, E_tau) if return_profile else E0



