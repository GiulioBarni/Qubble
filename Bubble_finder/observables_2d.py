"""
observables_2d.py — 2D charge and energy (τ=0 ghost reconstruction).

Purpose: 2D observables on the solver grid with twisted BC and ghost reconstruction at τ=0.
1D helpers (Q_homogeneous_ball, compute_charge_1d, etc.) live in observables_1d.py.

CHARGE CONVENTION (IMPORTANT)
-----------------------------
We use
    q(τ,r) = 1/2 * Re( phibar * ∂τ phi  -  phi * ∂τ phibar )
    Q(τ)   = 4π ∫_0^{rmax} dr r^2 q(τ,r)

This choice is consistent with observables_1d: Q_hom = 4π ω ρ0^2 (rmax^3/3).
"""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
from scipy.integrate import simpson

from .observables_1d import (
    Q_homogeneous_ball,
    compute_charge,
    compute_charge_1d_volume_corrected,
    compute_charge_density,
    compute_energy,
    compute_energy_density,
)

# Aliases for notebook compatibility
compute_charge_1d = compute_charge
compute_energy_1d = compute_energy
compute_charge_density_1d = compute_charge_density
compute_energy_density_1d = compute_energy_density


# -----------------------------------------------------------------------------
# 2D-from-1D bridge (uses 2D r grid; homogeneous target from observables_1d)
# -----------------------------------------------------------------------------

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
# 2D energy: homogeneous background density (for clean subtraction)
# -----------------------------------------------------------------------------

def _homogeneous_energy_density_2d(
    solver: Any,
    r: np.ndarray,
    omega: float,
    rho0_bg: float,
) -> np.ndarray:
    """
    Energy density of the homogeneous reference state (phi = phibar = rho0_bg constant)
    with the SAME convention as the field density:
      dens = -(phi_tau*phibar_tau) + (phi_r*phibar_r) + V.
    For homogeneous: phi_r = phibar_r = 0; phi_tau*phibar_tau = -omega^2*rho0_bg^2
    => dens_bg = +omega^2*rho0_bg^2 + V(rho0_bg).
    Returns array of shape r for pointwise subtraction.
    """
    r = np.asarray(r, dtype=float)
    V_bg = solver.U(np.full_like(r, rho0_bg, dtype=float))
    return (omega**2 * rho0_bg**2) + V_bg


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
    Energy E on a τ slice. Density (must match Q-ball convention exactly):
        dens = -(phi_tau*phibar_tau) + (phi_r*phibar_r) + V
    The MINUS sign in front of the time-derivative product is correct (not a bug).

    subtract_background: if True, subtract the homogeneous reference density pointwise:
        dens_sub = dens_field - dens_bg, with dens_bg = omega^2*rho0^2 + V(rho0).
    This yields E ≈ 0 for the homogeneous configuration (no ad-hoc patches).
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
        V_full = solver.U(rho)
        dens_field = -(phi_tau * phibar_tau) + (phi_r * phibar_r) + V_full
        if subtract_background:
            dens_bg = _homogeneous_energy_density_2d(solver, r, omega, rho_bg)
            dens_field = dens_field - dens_bg
        E_tau[i] = float(4.0 * np.pi * simpson(r**2 * dens_field.real, x=r))

    i0 = index_tau if index_tau >= 0 else Nt + index_tau
    E0 = float(E_tau[i0])
    return (E0, E_tau) if return_profile else E0


def compute_charge_tau0_ghost_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    *,
    subtract_background: bool = True,
    return_profile: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
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

    if return_profile:
        return (Q, np.asarray(qdens, dtype=float))
    return Q


def compute_energy_tau0_ghost_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    *,
    subtract_background: bool = True,
    return_profile: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Energy E(τ=0) with ghost reconstruction (half-box) and midpoint.
    Density: dens = -(phi_tau*phibar_tau) + (phi_r*phibar_r) + V (minus sign is correct).
    subtract_background: pointwise subtract dens_bg = omega^2*rho0^2 + V(rho0) so E_hom ≈ 0.
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

    V_full = solver.U(rho)
    dens_field = -(phi_tau0 * phibar_tau0) + (phi_r0 * phibar_r0) + V_full
    if subtract_background:
        dens_bg = _homogeneous_energy_density_2d(solver, r, omega, rho0_bg)
        dens_field = dens_field - dens_bg
    e_dens = np.asarray(dens_field.real, dtype=float)
    E = float(4.0 * np.pi * simpson(r**2 * e_dens, x=r))

    if return_profile:
        return (E, e_dens)
    return E


def compute_charge_density_tau0_ghost_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    *,
    subtract_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge density q(r) at τ=0 with ghost reconstruction (twisted BC).
    Returns (r_grid, q_r) so that Q = 4π ∫ r² q(r) dr with the same discretization as totals.
    """
    _, q_r = compute_charge_tau0_ghost_2d(
        solver, y, ybar, subtract_background=subtract_background, return_profile=True
    )
    r = np.asarray(solver.grid.r, dtype=float)
    return (r, np.asarray(q_r, dtype=float))


def compute_energy_density_tau0_ghost_2d(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
    *,
    subtract_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Energy density e(r) at τ=0 with ghost reconstruction (twisted BC).
    Returns (r_grid, e_r) so that E = 4π ∫ r² e(r) dr with the same discretization as totals.
    """
    _, e_r = compute_energy_tau0_ghost_2d(
        solver, y, ybar, subtract_background=subtract_background, return_profile=True
    )
    r = np.asarray(solver.grid.r, dtype=float)
    return (r, np.asarray(e_r, dtype=float))


def compute_observables_tau0_ghost(
    solver: Any,
    x: np.ndarray,
    *,
    subtract_background_charge: bool = False,
    subtract_background_energy: bool = True,
) -> dict:
    """
    Single convenience wrapper for τ=0 ghost observables used everywhere in logs.
    Unpacks x -> (y, ybar), computes Q_ghost, E_ghost, q_r, e_r and summary stats.
    Returns dict with: Q, E, r, q_r, e_r, q_max, r_at_q_max, e_max, r_at_e_max.

    In scans we typically want: Q = total charge (subtract_background_charge=False),
    E = energy relative to homogeneous (subtract_background_energy=True) so E_hom ≈ 0.
    """
    y, ybar = solver.unpack(x)
    r = np.asarray(solver.grid.r, dtype=float)

    Q = float(
        compute_charge_tau0_ghost_2d(
            solver, y, ybar, subtract_background=subtract_background_charge, return_profile=False
        )
    )
    E = float(
        compute_energy_tau0_ghost_2d(
            solver, y, ybar, subtract_background=subtract_background_energy, return_profile=False
        )
    )
    _, q_r = compute_charge_tau0_ghost_2d(
        solver, y, ybar, subtract_background=subtract_background_charge, return_profile=True
    )
    _, e_r = compute_energy_tau0_ghost_2d(
        solver, y, ybar, subtract_background=subtract_background_energy, return_profile=True
    )
    q_r = np.asarray(q_r, dtype=float)
    e_r = np.asarray(e_r, dtype=float)

    q_max = float(np.max(np.abs(q_r))) if q_r.size else 0.0
    e_max = float(np.max(np.abs(e_r))) if e_r.size else 0.0
    r_at_q_max = float(r[int(np.argmax(np.abs(q_r)))]) if q_r.size else 0.0
    r_at_e_max = float(r[int(np.argmax(np.abs(e_r)))]) if e_r.size else 0.0

    # Volume and densities (same convention as 1D: V = (4/3)*pi*Lr^3)
    # Use true Lr = dr*Nr to match notebook (e.g. Lr_diag = sol_eta.grid.dr * sol_eta.grid.Nr)
    if hasattr(solver.grid, "dr") and hasattr(solver.grid, "Nr"):
        Lr = float(solver.grid.dr * solver.grid.Nr)
    else:
        Lr = float(getattr(solver.grid, "Lr", r[-1] if r.size else 0.0))
    V = (4.0 / 3.0) * np.pi * (Lr**3)
    rho_Q = Q / V if V > 0 else 0.0
    rho_E = E / V if V > 0 else 0.0

    return {
        "Q": Q,
        "E": E,
        "r": r,
        "q_r": q_r,
        "e_r": e_r,
        "q_max": q_max,
        "e_max": e_max,
        "r_at_q_max": r_at_q_max,
        "r_at_e_max": r_at_e_max,
        "V": float(V),
        "rho_Q": float(rho_Q),
        "rho_E": float(rho_E),
    }


def assert_E_hom_near_zero(
    solver: Any,
    *,
    tol: float = 1e-3,
    message: str = "E (subtracted) for homogeneous config should be ≈ 0",
) -> float:
    """
    Self-check: build homogeneous field (y=ybar=0), compute E with subtract_background_energy=True,
    and assert |E| < tol. Returns E so callers can log it. Use to prevent regressions.
    tol can be relaxed for coarse grids (e.g. 1e-6 to 1e-3 depending on resolution).
    """
    x_bg = solver._zero_vec()
    y, ybar = solver.unpack(x_bg)
    E = float(
        compute_energy_tau0_ghost_2d(
            solver, y, ybar, subtract_background=True, return_profile=False
        )
    )
    if abs(E) >= tol:
        raise AssertionError(f"{message}: got E = {E:.6e}, tol = {tol}")
    return E


def compute_targets_tau0_ghost(
    solver: Any,
    *,
    subtract_background_charge: bool = False,
    subtract_background_energy: bool = True,
) -> dict:
    """
    Compute TARGET charge/energy and densities by applying the same τ=0 ghost
    observable definitions to the homogeneous background configuration (y=ybar=0)
    on the same grid. Used for ratio_Q = Q/Q_target, ratio_E = E/E_target.

    Defaults: subtract_background_charge=False (Q_target = total charge of hom),
    subtract_background_energy=True (E_target ≈ 0 for homogeneous; use for diagnostics).
    Do NOT use E_target = omega*Q_target; E comes from the same ghost definition.
    """
    x_bg = solver._zero_vec()
    return compute_observables_tau0_ghost(
        solver,
        x_bg,
        subtract_background_charge=subtract_background_charge,
        subtract_background_energy=subtract_background_energy,
    )


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
    "compute_charge_density_tau0_ghost_2d",
    "compute_energy_density_tau0_ghost_2d",
    "compute_observables_tau0_ghost",
    "assert_E_hom_near_zero",
    "compute_targets_tau0_ghost",
]
