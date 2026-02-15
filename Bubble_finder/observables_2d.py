"""
observables_2d.py — Charge and energy for 1D and 2D bubble.

Purpose:
- 1D: centralised re-export from bounce_1d + helpers (homogeneous Q, volume-corrected).
- 2D: charge/energy on τ slices with the same conventions as 1D.

CHARGE CONVENTION (IMPORTANT)
-----------------------------
We use
    q(τ,r) = 1/2 * Re( phibar * ∂τ phi  -  phi * ∂τ phibar )
    Q(τ)   = 4π ∫_0^{rmax} dr r^2 q(τ,r)

This choice is consistent with the 1D helpers:
    Q_hom = 4π ω ρ0^2 (rmax^3/3)

If you see a factor-of-2 mismatch somewhere in the 2D solver, the first thing
to check is whether that 1/2 is missing or duplicated.
"""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
from scipy.integrate import simpson

# -----------------------------------------------------------------------------
# 1D observables (re-export from bounce_1d for centralised API)
# -----------------------------------------------------------------------------

from .bounce_1d import (
    compute_charge as compute_charge_1d,
    compute_charge_density as compute_charge_density_1d,
    compute_energy as compute_energy_1d,
    compute_energy_density as compute_energy_density_1d,
)

# Notebook compatibility (1D bounce)
compute_energy = compute_energy_1d
compute_charge = compute_charge_1d
compute_energy_density = compute_energy_density_1d
compute_charge_density = compute_charge_density_1d


# -----------------------------------------------------------------------------
# 1D helpers
# -----------------------------------------------------------------------------

def Q_homogeneous_ball(omega: float, phi_false: float, r_max: float) -> float:
    """
    Charge of the homogeneous configuration φ = φ_false in the ball [0, r_max]:
        Q_hom = 4π ω φ_false² (r_max³/3).
    """
    return float(4.0 * np.pi * omega * (phi_false**2) * (r_max**3 / 3.0))


def compute_charge_like_2d_from_1d(
    r_1d: np.ndarray,
    phi_rot_1d: np.ndarray,
    r_2d: np.ndarray,
    omega: float,
    rho0: float,
    *,
    subtract_background: bool = True,
) -> float:
    """
    Compute charge from 1D profile using the exact 2D definition and discretization.

    For a tau-independent config at tau=0: phi = phi_rot, phibar = phi_rot, and
    q = (1/2)*Re(phibar*phi_tau - phi*phibar_tau) = omega * phi_rot^2.
    So Q = 4π ω ∫ r² φ_rot² dr.

    Uses the 2D r grid and Simpson integration so the result is directly
    comparable to compute_charge_tau0_ghost_2d on an embedded 1D config.
    """
    r_1d = np.asarray(r_1d, dtype=float)
    phi_rot_1d = np.asarray(phi_rot_1d, dtype=float)
    r_2d = np.asarray(r_2d, dtype=float)
    order = np.argsort(r_1d)
    r_1d = r_1d[order]
    phi_rot_1d = phi_rot_1d[order]

    # Interpolate 1D profile onto 2D r grid (same as embedding)
    phi_on_r2d = np.interp(r_2d, r_1d, phi_rot_1d, left=phi_rot_1d[0], right=phi_rot_1d[-1])
    qdens = omega * (phi_on_r2d**2)
    Q = float(4.0 * np.pi * simpson(r_2d**2 * qdens, x=r_2d))

    if subtract_background:
        r_max = float(r_2d[-1])
        Q -= Q_homogeneous_ball(omega=omega, phi_false=rho0, r_max=r_max)
    return Q


def compute_charge_1d_volume_corrected(
    r: np.ndarray,
    phi: np.ndarray,
    omega: float,
    r_max_ref: float,
    phi_false_tail: float,
) -> float:
    """
    1D charge on the same volume [0, r_max_ref]: integrate the bounce from 0 to r_max_bubble,
    then extend with φ = phi_false_tail from r_max_bubble to r_max_ref.

    Convention consistent with Q_homogeneous_ball:
        Q = 4π ω ∫ r² φ² dr.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if r.ndim != 1 or phi.shape != r.shape:
        raise ValueError("compute_charge_1d_volume_corrected: r and phi must be 1D arrays with same shape.")
    if len(r) < 2:
        return 0.0

    r_max_bubble = float(r[-1])

    # Bubble part
    Q_bubble_part = float(simpson(r**2 * phi**2, x=r))

    if r_max_bubble >= r_max_ref:
        # Truncate/interpolate up to r_max_ref
        mask = r <= r_max_ref
        r_trunc = r[mask]
        phi_trunc = phi[mask]
        if len(r_trunc) == 0:
            # All beyond r_max_ref: interpolate a single point
            phi_at_rmax = float(np.interp(r_max_ref, r, phi))
            Q_bubble_part = float(simpson(np.array([0.0, r_max_ref])**2 * np.array([phi_trunc[0], phi_at_rmax])**2,
                                          x=np.array([0.0, r_max_ref])))
        else:
            if r_trunc[-1] < r_max_ref - 1e-12:
                r_ext = np.append(r_trunc, r_max_ref)
                phi_ext = np.append(phi_trunc, np.interp(r_max_ref, r, phi))
                Q_bubble_part = float(simpson(r_ext**2 * phi_ext**2, x=r_ext))
            else:
                Q_bubble_part = float(simpson(r_trunc**2 * phi_trunc**2, x=r_trunc))
        Q_tail = 0.0
    else:
        # Add homogeneous tail up to r_max_ref
        Q_tail = (phi_false_tail**2) * (r_max_ref**3 - r_max_bubble**3) / 3.0

    return float(4.0 * np.pi * omega * (Q_bubble_part + Q_tail))


# -----------------------------------------------------------------------------
# 2D observables helpers
# -----------------------------------------------------------------------------

def _safe_divide_by_r(num: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    num / r with robust handling of r=0 (returns 0 at r=0).
    num may be complex.
    """
    num = np.asarray(num)
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(num, dtype=np.result_type(num, np.complex128))
    np.divide(num, r, out=out, where=(r != 0.0))
    return out


def _dr1_radial(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """First radial derivative (centered in the interior, one-sided at boundaries). Supports complex."""
    r = np.asarray(r, dtype=float)
    f = np.asarray(f)
    df = np.empty_like(f, dtype=np.result_type(f, np.complex128))
    n = len(r)
    if n >= 3:
        df[1:-1] = (f[2:] - f[:-2]) / (r[2:] - r[:-2])
        df[0] = (f[1] - f[0]) / (r[1] - r[0])
        df[-1] = (f[-1] - f[-2]) / (r[-1] - r[-2])
    elif n == 2:
        df[:] = (f[1] - f[0]) / (r[1] - r[0])
    else:
        df[:] = 0.0
    return df


def _tau_derivative_centered(
    tau: np.ndarray,
    dt: float,
    y_im1: np.ndarray,
    y_ip1: np.ndarray,
) -> np.ndarray:
    """
    Centered derivative d/dτ using two neighbors that are already BC-aware.

    If tau increases with index:  y_t ~ (y_{i+1} - y_{i-1})/(2dt)
    If tau decreases with index:  y_t ~ (y_{i-1} - y_{i+1})/(2dt)
    """
    tau = np.asarray(tau, dtype=float)
    if len(tau) >= 2 and (tau[1] > tau[0]):
        return (y_ip1 - y_im1) / (2.0 * dt)
    return (y_im1 - y_ip1) / (2.0 * dt)


# -----------------------------------------------------------------------------
# 2D observables (grid + ghost reconstruction at τ=0)
# -----------------------------------------------------------------------------

def compute_charge_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    index_tau: int = 0,
    *,
    return_profile: bool = False,
    subtract_background: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Charge Q on a τ slice of the 2D field (y, ybar), with the solver's BC (twist/ghost).

    Convention (consistent with 1D):
        q_phys = 1/2 * Re(phibar ∂τ phi - phi ∂τ phibar)
        Q      = 4π ∫ r² q_phys dr

    NOTE:
    - We do not assume solver.compute_charge has the correct normalisation here;
      we implement directly using solver._tau_neighbors and solver.phi.
    """
    y = np.asarray(y)
    ybar = np.asarray(ybar)
    if y.shape != ybar.shape:
        raise ValueError("compute_charge_2d: y and ybar must have the same shape (Nr, Nt).")

    r = np.asarray(solver.grid.r, dtype=float)
    tau = np.asarray(solver.grid.tau, dtype=float)
    Nt = int(getattr(solver, "Nt", y.shape[1]))
    dt = float(getattr(solver, "dt", getattr(solver.grid, "dtau", None)))
    if dt is None:
        raise ValueError("compute_charge_2d: could not determine dt (solver.dt or solver.grid.dtau).")

    omega = float(getattr(solver, "omega"))
    rho0 = float(getattr(solver, "rho0"))

    phi, phibar = solver.phi(y, ybar)  # (Nr, Nt), complex

    Q_tau = np.zeros(Nt, dtype=float)

    for i in range(Nt):
        if not hasattr(solver, "_tau_neighbors"):
            raise ValueError("compute_charge_2d: solver must provide _tau_neighbors(y,ybar,i).")
        y_im1, y_ip1, yb_im1, yb_ip1 = solver._tau_neighbors(y, ybar, i)

        y_t = _tau_derivative_centered(tau, dt, y_im1, y_ip1)
        yb_t = _tau_derivative_centered(tau, dt, yb_im1, yb_ip1)

        # y_tot = y + r*rho0   (these are the 'rotated' numerators)
        y_tot = y[:, i] + r * rho0
        yb_tot = ybar[:, i] + r * rho0

        inv_r = np.zeros_like(r, dtype=float)
        inv_r[r != 0.0] = 1.0 / r[r != 0.0]

        exp_p = np.exp(+omega * tau[i]) * inv_r
        exp_m = np.exp(-omega * tau[i]) * inv_r

        phi_tau = exp_p * (y_t + omega * y_tot)
        phibar_tau = exp_m * (yb_t - omega * yb_tot)

        j_tau = phibar[:, i] * phi_tau - phi[:, i] * phibar_tau
        qdens = 0.5 * j_tau.real

        Q_tau[i] = float(4.0 * np.pi * simpson(r**2 * qdens, x=r))

    i0 = index_tau if index_tau >= 0 else Nt + index_tau
    Q0 = float(Q_tau[i0])

    if subtract_background:
        Q_bg = Q_homogeneous_ball(omega=omega, phi_false=rho0, r_max=float(r[-1]))
        Q0 -= Q_bg
        Q_tau = Q_tau - Q_bg

    return (Q0, Q_tau) if return_profile else Q0


def compute_energy_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    index_tau: int = 0,
    *,
    return_profile: bool = False,
    subtract_background: bool = True,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Energy E on a τ slice:
        dens = -(phi_tau*phibar_tau) + (phi_r*phibar_r) + V

    For V we use ρ reconstructed from the "rotated" fields (without exp(±ωτ)), as in the solver.
    subtract_background: subtracts only U(ρ0).
    """
    y = np.asarray(y)
    ybar = np.asarray(ybar)
    if y.shape != ybar.shape:
        raise ValueError("compute_energy_2d: y and ybar must have the same shape (Nr, Nt).")

    r = np.asarray(solver.grid.r, dtype=float)
    tau = np.asarray(solver.grid.tau, dtype=float)
    Nt = int(getattr(solver, "Nt", y.shape[1]))
    dt = float(getattr(solver, "dt", getattr(solver.grid, "dtau", None)))
    if dt is None:
        raise ValueError("compute_energy_2d: could not determine dt (solver.dt or solver.grid.dtau).")

    omega = float(getattr(solver, "omega"))
    rho_bg = float(getattr(solver, "rho0"))
    rho_eps = float(getattr(getattr(solver, "settings", object()), "rho_eps", 0.0))

    phi, phibar = solver.phi(y, ybar)

    # Rotated fields for potential: prefer solver.phi_rot if available
    if hasattr(solver, "phi_rot"):
        phi_rot, phibar_rot = solver.phi_rot(y, ybar)
    else:
        # Fallback: remove exp factors analytically at each τ
        # phi_rot = (y + r*rho0)/r, phibar_rot = (ybar + r*rho0)/r
        rr = r[:, None]
        phi_rot = _safe_divide_by_r(y + rr * rho_bg, rr)
        phibar_rot = _safe_divide_by_r(ybar + rr * rho_bg, rr)

    E_tau = np.zeros(Nt, dtype=float)

    for i in range(Nt):
        if not hasattr(solver, "_tau_neighbors"):
            raise ValueError("compute_energy_2d: solver must provide _tau_neighbors(y,ybar,i).")
        y_im1, y_ip1, yb_im1, yb_ip1 = solver._tau_neighbors(y, ybar, i)

        y_t = _tau_derivative_centered(tau, dt, y_im1, y_ip1)
        yb_t = _tau_derivative_centered(tau, dt, yb_im1, yb_ip1)

        y_tot = y[:, i] + r * rho_bg
        yb_tot = ybar[:, i] + r * rho_bg

        inv_r = np.zeros_like(r, dtype=float)
        inv_r[r != 0.0] = 1.0 / r[r != 0.0]

        exp_p = np.exp(+omega * tau[i]) * inv_r
        exp_m = np.exp(-omega * tau[i]) * inv_r

        phi_tau = exp_p * (y_t + omega * y_tot)
        phibar_tau = exp_m * (yb_t - omega * yb_tot)

        phi_r = _dr1_radial(r, phi[:, i])
        phibar_r = _dr1_radial(r, phibar[:, i])

        u = (phi_rot[:, i] * phibar_rot[:, i]).real
        if hasattr(solver, "_smooth_pos"):
            u_pos, _ = solver._smooth_pos(u)
        else:
            u_pos = np.maximum(u, 0.0)

        rho = np.sqrt(u_pos + rho_eps)
        V = solver.U(rho)
        if subtract_background:
            V = V - solver.U(np.full_like(rho, rho_bg))

        dens = -(phi_tau * phibar_tau) + (phi_r * phibar_r) + V
        E_tau[i] = float(4.0 * np.pi * simpson(r**2 * dens.real, x=r))

    i0 = index_tau if index_tau >= 0 else Nt + index_tau
    E0 = float(E_tau[i0])
    return (E0, E_tau) if return_profile else E0


def compute_charge_tau0_ghost_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    *,
    subtract_background: bool = True,
) -> float:
    """
    Charge Q at τ=0 with ghost reconstruction (half-box):
      y_plus  = ybar[:,0],  y_minus  = y[:,0]
      ybar_plus = y[:,0],   ybar_minus = ybar[:,0]
      midpoint for (y0, ybar0) and central derivative.

    Convention consistent with 1D:
      q = 1/2 Re(phibar φ_τ - φ phibar_τ).
    """
    y = np.asarray(y)
    ybar = np.asarray(ybar)
    r = np.asarray(solver.grid.r, dtype=float)

    dt = float(getattr(solver, "dt", getattr(solver.grid, "dtau", None)))
    if dt is None:
        raise ValueError("compute_charge_tau0_ghost_2d: could not determine dt.")

    omega = float(getattr(solver, "omega"))
    rho0 = float(getattr(solver, "rho0"))

    y_minus, ybar_minus = y[:, 0], ybar[:, 0]
    y_plus, ybar_plus = ybar[:, 0], y[:, 0]   # Ghost via swap at τ=0

    y0 = 0.5 * (y_plus + y_minus)
    ybar0 = 0.5 * (ybar_plus + ybar_minus)

    y_t0 = (y_plus - y_minus) / dt
    ybar_t0 = (ybar_plus - ybar_minus) / dt

    # At τ=0 the exp factors are 1
    phi0 = rho0 + _safe_divide_by_r(y0, r)
    phibar0 = rho0 + _safe_divide_by_r(ybar0, r)

    inv_r = np.zeros_like(r, dtype=float)
    inv_r[r != 0.0] = 1.0 / r[r != 0.0]

    phi_tau0 = inv_r * (y_t0 + omega * (y0 + r * rho0))
    phibar_tau0 = inv_r * (ybar_t0 - omega * (ybar0 + r * rho0))

    j = phibar0 * phi_tau0 - phi0 * phibar_tau0
    qdens = 0.5 * j.real
    Q = float(4.0 * np.pi * simpson(r**2 * qdens, x=r))

    if subtract_background:
        Q -= Q_homogeneous_ball(omega=omega, phi_false=rho0, r_max=float(r[-1]))

    return Q


def compute_energy_tau0_ghost_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    *,
    subtract_background: bool = True,
) -> float:
    """
    Energy E(τ=0) with ghost reconstruction (half-box) and midpoint.
    """
    y = np.asarray(y)
    ybar = np.asarray(ybar)
    r = np.asarray(solver.grid.r, dtype=float)

    dt = float(getattr(solver, "dt", getattr(solver.grid, "dtau", None)))
    if dt is None:
        raise ValueError("compute_energy_tau0_ghost_2d: could not determine dt.")

    omega = float(getattr(solver, "omega"))
    rho0_bg = float(getattr(solver, "rho0"))
    rho_eps = float(getattr(getattr(solver, "settings", object()), "rho_eps", 0.0))

    y_minus, ybar_minus = y[:, 0], ybar[:, 0]
    y_plus, ybar_plus = ybar[:, 0], y[:, 0]   # Ghost via swap

    y0 = 0.5 * (y_plus + y_minus)
    ybar0 = 0.5 * (ybar_plus + ybar_minus)

    y_t0 = (y_plus - y_minus) / dt
    ybar_t0 = (ybar_plus - ybar_minus) / dt

    phi0 = rho0_bg + _safe_divide_by_r(y0, r)
    phibar0 = rho0_bg + _safe_divide_by_r(ybar0, r)

    inv_r = np.zeros_like(r, dtype=float)
    inv_r[r != 0.0] = 1.0 / r[r != 0.0]

    phi_tau0 = inv_r * (y_t0 + omega * (y0 + r * rho0_bg))
    phibar_tau0 = inv_r * (ybar_t0 - omega * (ybar0 + r * rho0_bg))

    phi_r0 = _dr1_radial(r, phi0)
    phibar_r0 = _dr1_radial(r, phibar0)

    u = (phi0 * phibar0).real
    u_pos = np.maximum(u, 0.0)
    rho = np.sqrt(u_pos + rho_eps)

    V = solver.U(rho)
    if subtract_background:
        V = V - solver.U(np.full_like(rho, rho0_bg))

    dens = -(phi_tau0 * phibar_tau0) + (phi_r0 * phibar_r0) + V
    return float(4.0 * np.pi * simpson(r**2 * dens.real, x=r))


__all__ = [
    # 1D (same names as bounce_1d)
    "compute_energy",
    "compute_charge",
    "compute_energy_density",
    "compute_charge_density",
    # 1D explicit
    "compute_energy_1d",
    "compute_charge_1d",
    "compute_energy_density_1d",
    "compute_charge_density_1d",
    # 1D volume-corrected and homogeneous
    "Q_homogeneous_ball",
    "compute_charge_1d_volume_corrected",
    "compute_charge_like_2d_from_1d",
    # 2D
    "compute_charge_2d",
    "compute_energy_2d",
    "compute_charge_tau0_ghost_2d",
    "compute_energy_tau0_ghost_2d",
]
