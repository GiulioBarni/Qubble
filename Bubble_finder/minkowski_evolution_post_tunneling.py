"""
Minkowski post-tunneling evolution from 2D Euclidean solution at τ=0.

Initial data: φ(t=0,r)=φ_E(τ=0,r), φ̄(t=0,r)=φ̄_E(τ=0,r),
              ∂_t φ(t=0,r) = i (∂_τ φ_E)|_{τ=0},  ∂_t φ̄ = i (∂_τ φ̄_E)|_{τ=0}
(analytic continuation t = -i τ ⇒ ∂_t = i ∂_τ).

EOM (spherical 1D): ∂_t² φ = ∂_r² φ + (2/r) ∂_r φ − ∂V/∂φ̄,  and conjugate.
Energy: E = 4π ∫ dr r² [ |∂_t φ|² + (∂_r φ)(∂_r φ̄) + V(ρ) ].
Charge: j⁰ = i(φ̄ ∂_t φ − ∂_t φ̄ φ),  Q = 4π ∫ dr r² (1/2)Re(j⁰).

All comments and docstrings in English.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np


def extract_initial_data_from_euclidean(
    phi: np.ndarray,
    phi_bar: np.ndarray,
    tau_grid: np.ndarray,
    r_grid: np.ndarray,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Minkowski initial data from Euclidean solution at τ=0.

    Assumes phi, phi_bar have shape (Nr, Nt) with r along rows and τ along columns.
    tau_grid shape (Nt,), r_grid shape (Nr,).

    Returns:
        r_grid: same as input (1D)
        phi0, phi_bar0: φ(τ=0,r), φ̄(τ=0,r)
        dotphi0, dotphi_bar0: ∂_t φ and ∂_t φ̄ at t=0, with ∂_t = i ∂_τ.
    """
    phi = np.asarray(phi)
    phi_bar = np.asarray(phi_bar)
    tau_grid = np.asarray(tau_grid, dtype=float).flatten()
    r_grid = np.asarray(r_grid, dtype=float).flatten()

    Nr, Nt = phi.shape
    if phi_bar.shape != (Nr, Nt) or len(tau_grid) != Nt or len(r_grid) != Nr:
        raise ValueError("Shape mismatch: phi/phi_bar (Nr,Nt), tau_grid (Nt,), r_grid (Nr).")

    i0 = int(np.argmin(np.abs(tau_grid)))
    tau0 = float(tau_grid[i0])

    phi0 = np.asarray(phi[:, i0], dtype=np.complex128)
    phi_bar0 = np.asarray(phi_bar[:, i0], dtype=np.complex128)

    # ∂_τ φ at index i0: central if possible, else forward/backward
    if i0 >= 1 and i0 < Nt - 1:
        dtau = tau_grid[i0 + 1] - tau_grid[i0 - 1]
        dtau_phi0 = (phi[:, i0 + 1] - phi[:, i0 - 1]) / dtau
        dtau_phi_bar0 = (phi_bar[:, i0 + 1] - phi_bar[:, i0 - 1]) / dtau
    elif i0 == 0 and Nt >= 2:
        dtau = tau_grid[1] - tau_grid[0]
        dtau_phi0 = (phi[:, 1] - phi[:, 0]) / dtau
        dtau_phi_bar0 = (phi_bar[:, 1] - phi_bar[:, 0]) / dtau
    elif i0 == Nt - 1 and Nt >= 2:
        dtau = tau_grid[-1] - tau_grid[-2]
        dtau_phi0 = (phi[:, -1] - phi[:, -2]) / dtau
        dtau_phi_bar0 = (phi_bar[:, -1] - phi_bar[:, -2]) / dtau
    else:
        dtau_phi0 = np.zeros_like(phi0)
        dtau_phi_bar0 = np.zeros_like(phi_bar0)

    # Minkowski: ∂_t = i ∂_τ  ⇒  dotphi = i ∂_τ φ
    dotphi0 = 1j * np.asarray(dtau_phi0, dtype=np.complex128)
    dotphi_bar0 = 1j * np.asarray(dtau_phi_bar0, dtype=np.complex128)

    if verbose:
        diff_conj = np.max(np.abs(phi_bar0 - np.conj(phi0)))
        print(f"[extract_initial_data] tau index i0={i0}, tau0={tau0:.6g}; max|φ̄0 − conj(φ0)| = {diff_conj:.3e}")

    return r_grid, phi0, phi_bar0, dotphi0, dotphi_bar0


def _radial_laplacian(r: np.ndarray, f: np.ndarray, dr: float) -> np.ndarray:
    """Spherical Laplacian (1/r²)d/dr(r² df/dr). Regularity at r=0: (d/dr(r² df/dr))|0 = 0."""
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=np.complex128)
    n = len(r)
    lap = np.empty_like(f)
    if n < 2:
        lap[:] = 0.0
        return lap
    # Interior: (1/r²) * (r² f')' ≈ (1/r_i^2) * (r_{i+1/2}^2 (f_{i+1}-f_i) - r_{i-1/2}^2 (f_i-f_{i-1})) / dr^2
    for i in range(1, n - 1):
        ri = r[i]
        r_plus = 0.5 * (r[i + 1] + ri)
        r_minus = 0.5 * (ri + r[i - 1])
        lap[i] = (r_plus**2 * (f[i + 1] - f[i]) - r_minus**2 * (f[i] - f[i - 1])) / (dr * dr * ri * ri)
    # At r=0 (index 0): regularity => 3 f''(0), use 6*(f_1 - f_0)/dr^2
    if r[0] > 0:
        lap[0] = 6.0 * (f[1] - f[0]) / (dr * dr)
    else:
        lap[0] = 0.0
    # Outer boundary: same stencil as interior
    if n >= 3 and r[-1] > 0:
        r_plus = 0.5 * (r[-1] + r[-2])
        r_minus = 0.5 * (r[-2] + r[-3])
        lap[-1] = (r_plus**2 * (f[-1] - f[-2]) - r_minus**2 * (f[-2] - f[-3])) / (dr * dr * r[-1] * r[-1])
    else:
        lap[-1] = 0.0
    return lap


def _dr1_radial(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """First radial derivative; regularity at r=0 (forward difference)."""
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=np.complex128)
    n = len(r)
    df = np.empty_like(f)
    if n >= 2:
        df[0] = (f[1] - f[0]) / (r[1] - r[0]) if r[1] != r[0] else 0.0
        if n >= 3:
            df[1:-1] = (f[2:] - f[:-2]) / (r[2:] - r[:-2])
        df[-1] = (f[-1] - f[-2]) / (r[-1] - r[-2]) if r[-1] != r[-2] else 0.0
    else:
        df[:] = 0.0
    return df


def _rho_from_phi_phibar(phi: np.ndarray, phi_bar: np.ndarray, rho_eps: float = 1e-14) -> np.ndarray:
    """ρ = sqrt(Re(φ φ̄))_+ + small epsilon, for U(ρ)."""
    u = (phi * phi_bar).real
    u_pos = np.maximum(u, 0.0)
    return np.sqrt(u_pos + rho_eps)


def minkowski_acceleration(
    phi: np.ndarray,
    phi_bar: np.ndarray,
    r_grid: np.ndarray,
    U: Callable[[np.ndarray], np.ndarray],
    dU: Callable[[np.ndarray], np.ndarray],
    rho_eps: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂_t² φ and ∂_t² φ̄ from EOM:
    ∂_t² φ = ∂_r² φ + (2/r) ∂_r φ − (dU/dρ)/(2ρ) φ  (and conjugate for φ̄).
    """
    r = np.asarray(r_grid, dtype=float).flatten()
    dr = float(r[1] - r[0]) if len(r) >= 2 else 1.0

    lap_phi = _radial_laplacian(r, phi, dr)
    lap_phibar = _radial_laplacian(r, phi_bar, dr)

    rho = _rho_from_phi_phibar(phi, phi_bar, rho_eps)
    dU_rho = np.asarray(dU(rho), dtype=np.complex128)
    coeff = dU_rho / (2.0 * rho)
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)

    accel_phi = lap_phi - coeff * phi
    accel_phibar = lap_phibar - coeff * phi_bar

    return accel_phi, accel_phibar


def velocity_verlet_step(
    phi: np.ndarray,
    phi_bar: np.ndarray,
    dotphi: np.ndarray,
    dotphi_bar: np.ndarray,
    r_grid: np.ndarray,
    dt: float,
    U: Callable[[np.ndarray], np.ndarray],
    dU: Callable[[np.ndarray], np.ndarray],
    rho_eps: float = 1e-14,
    sponge_fraction: float = 0.0,
    sponge_gamma: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One time step: velocity Verlet. Optionally apply sponge damping in the last
    sponge_fraction of the grid (sponge_gamma = damping factor per step).
    """
    acc_phi, acc_phibar = minkowski_acceleration(phi, phi_bar, r_grid, U, dU, rho_eps)

    phi_new = phi + dt * dotphi + 0.5 * dt * dt * acc_phi
    phi_bar_new = phi_bar + dt * dotphi_bar + 0.5 * dt * dt * acc_phibar

    acc_new_phi, acc_new_phibar = minkowski_acceleration(phi_new, phi_bar_new, r_grid, U, dU, rho_eps)

    dotphi_new = dotphi + 0.5 * dt * (acc_phi + acc_new_phi)
    dotphi_bar_new = dotphi_bar + 0.5 * dt * (acc_phibar + acc_new_phibar)

    if sponge_fraction > 0 and sponge_gamma > 0:
        n = len(r_grid)
        n_sponge = max(1, int(n * sponge_fraction))
        for i in range(n - n_sponge, n):
            fac = 1.0 - sponge_gamma
            dotphi_new[i] *= fac
            dotphi_bar_new[i] *= fac

    return phi_new, phi_bar_new, dotphi_new, dotphi_bar_new


def compute_energy(
    phi: np.ndarray,
    phi_bar: np.ndarray,
    dotphi: np.ndarray,
    dotphi_bar: np.ndarray,
    r_grid: np.ndarray,
    U: Callable[[np.ndarray], np.ndarray],
    rho_eps: float = 1e-14,
) -> float:
    """
    Minkowski energy E = 4π ∫ dr r² [ T + G + V ],
    T = dotphi*dotphi_bar,  G = (∂_r φ)(∂_r φ̄),  V = U(ρ).
    """
    r = np.asarray(r_grid, dtype=float).flatten()
    T = (dotphi * dotphi_bar).real
    phi_r = _dr1_radial(r, phi)
    phibar_r = _dr1_radial(r, phi_bar)
    G = (phi_r * phibar_r).real
    rho = _rho_from_phi_phibar(phi, phi_bar, rho_eps)
    V = np.asarray(U(rho), dtype=float).flatten()
    integrand = (T + G + V) * r * r
    return float(4.0 * np.pi * np.trapz(integrand, r))


def compute_charge(
    phi: np.ndarray,
    phi_bar: np.ndarray,
    dotphi: np.ndarray,
    dotphi_bar: np.ndarray,
    r_grid: np.ndarray,
) -> float:
    """
    U(1) charge Q = 4π ∫ dr r² j⁰ with j⁰ = i(φ̄ ∂_t φ − ∂_t φ̄ φ).
    Same convention as observables_2d: q = (1/2) Re(j⁰),  Q = 4π ∫ r² q dr.
    """
    r = np.asarray(r_grid, dtype=float).flatten()
    j0 = 1j * (phi_bar * dotphi - dotphi_bar * phi)
    q = 0.5 * j0.real
    return float(4.0 * np.pi * np.trapz(r * r * q, r))


def run_minkowski_evolution(
    r_grid: np.ndarray,
    phi0: np.ndarray,
    phi_bar0: np.ndarray,
    dotphi0: np.ndarray,
    dotphi_bar0: np.ndarray,
    U: Callable[[np.ndarray], np.ndarray],
    dU: Callable[[np.ndarray], np.ndarray],
    t_max: float,
    dt: Optional[float] = None,
    rho_eps: float = 1e-14,
    sponge_fraction: float = 0.1,
    sponge_gamma: float = 0.02,
    save_every_n: int = 10,
    snapshot_times: Optional[Tuple[float, ...]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Evolve initial data in Minkowski time. Returns:
    t_arr, E_arr, Q_arr, r_grid, and list of (t_snap, phi_snap, phi_bar_snap) for snapshot_times.
    """
    r = np.asarray(r_grid, dtype=float).flatten()
    if dt is None:
        dr = float(r[1] - r[0]) if len(r) >= 2 else 0.1
        dt = 0.2 * dr
        if verbose:
            print(f"[run_minkowski_evolution] dt = 0.2*dr = {dt:.4g} (dr={dr:.4g})")

    phi = np.asarray(phi0, dtype=np.complex128).copy()
    phi_bar = np.asarray(phi_bar0, dtype=np.complex128).copy()
    dotphi = np.asarray(dotphi0, dtype=np.complex128).copy()
    dotphi_bar = np.asarray(dotphi_bar0, dtype=np.complex128).copy()

    n_steps = int(round(t_max / dt))
    t_arr = np.arange(n_steps + 1, dtype=float) * dt
    E_arr = np.zeros(n_steps + 1)
    Q_arr = np.zeros(n_steps + 1)
    E_arr[0] = compute_energy(phi, phi_bar, dotphi, dotphi_bar, r, U, rho_eps)
    Q_arr[0] = compute_charge(phi, phi_bar, dotphi, dotphi_bar, r)

    snapshots = []
    if snapshot_times is None:
        snapshot_times = ()
    # Include t=0 snapshot if requested
    if snapshot_times and snapshot_times[0] <= 0:
        snapshots.append((0.0, phi.copy(), phi_bar.copy()))

    isnap = 1 if (snapshot_times and snapshot_times[0] <= 0) else 0
    for step in range(1, n_steps + 1):
        phi, phi_bar, dotphi, dotphi_bar = velocity_verlet_step(
            phi, phi_bar, dotphi, dotphi_bar, r, dt, U, dU, rho_eps, sponge_fraction, sponge_gamma
        )
        t = step * dt
        E_arr[step] = compute_energy(phi, phi_bar, dotphi, dotphi_bar, r, U, rho_eps)
        Q_arr[step] = compute_charge(phi, phi_bar, dotphi, dotphi_bar, r)

        while isnap < len(snapshot_times) and t >= snapshot_times[isnap]:
            snapshots.append((t, phi.copy(), phi_bar.copy()))
            isnap += 1

    if verbose:
        E0, Q0 = E_arr[0], Q_arr[0]
        drift_E_abs = np.max(np.abs(E_arr - E0))
        drift_E_rel = drift_E_abs / (abs(E0) + 1e-30)
        drift_Q_abs = np.max(np.abs(Q_arr - Q0))
        drift_Q_rel = drift_Q_abs / (abs(Q0) + 1e-30)
        print(f"[run_minkowski_evolution] E: max|ΔE|={drift_E_abs:.3e}, max|ΔE/E0|={drift_E_rel:.3e}")
        print(f"[run_minkowski_evolution] Q: max|ΔQ|={drift_Q_abs:.3e}, max|ΔQ/Q0|={drift_Q_rel:.3e}")

    return t_arr, E_arr, Q_arr, r_grid, snapshots
