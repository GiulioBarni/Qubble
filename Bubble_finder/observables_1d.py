"""
observables_1d.py — 1D-only charge and energy for O(3) bounce.

Single source of truth for 1D observables: Q, E_M (Minkowski energy), F_ω (grand-canonical
functional), charge/energy density profiles, homogeneous ball Q/E in finite volume.

Conventions:
- Minkowski energy density: 𝓔_M = 1/2(∂r ρ)² + 1/2 ω² ρ² + V(ρ)  =>  E_M = 4π ∫ r² 𝓔_M dr.
- Charge: Q = 4π ω ∫ r² φ² dr  (φ = ρ in the ansatz φ = ρ e^{iωt}).
- Grand-canonical / constrained functional: F_ω = E_M − ω Q = 4π ∫ r² [ ½(∂r φ)² + Ω(φ) ],
  with Ω(φ) = V(φ) − 1/2 ω² φ². Use F_ω for fixed-ω barrier comparisons, not for "physical energy".
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import simpson

from .potential_bubble import Omega_phi


def compute_charge(r: np.ndarray, phi: np.ndarray, omega: float) -> float:
    """
    Charge for 1D O(3) bounce: Q = 4π ω ∫ dr r² φ².
    Convention matches 2D: n_Q ∝ 2 ρ² ∂t α with α = ω t => Q = 4π ω ∫ r² ρ² dr.
    Uses Simpson's rule for integration.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return float(4.0 * np.pi * omega * simpson(r**2 * phi**2, x=r))


# Alias for clarity in docs and notebook
compute_charge_1d_spherical = compute_charge


def compute_Fomega_1d_spherical(
    r: np.ndarray,
    phi: np.ndarray,
    phi0: float,
    v1: float,
    v2: float,
    omega: float,
    phi_false: float,
    *,
    subtract_background: bool = True,
) -> float:
    """
    Grand-canonical / constrained functional F_ω (NOT Minkowski energy).

    F_ω = E_M − ω Q = 4π ∫ r² [ ½(∂_r φ)² + Ω(φ) ],  Ω(φ) = V(φ) − 1/2 ω² φ².

    Use for fixed-ω barrier / branch continuation. For microcanonical or "physical energy"
    comparisons use compute_energy_minkowski_1d_spherical (or compute_energy_physical_1d_spherical).

    If subtract_background=True (default): integrand uses Ω(φ) − Ω(φ_false).
    If subtract_background=False: integrand is ½(∂_r φ)² + Ω(φ).
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    dphi_dr = np.gradient(phi, r, edge_order=2)
    dphi_dr[0] = 0.0  # Regularity at origin
    Omega_phi_vals = np.array([Omega_phi(phi_i, phi0, v1, v2, omega) for phi_i in phi])
    if subtract_background:
        Omega_false = Omega_phi(phi_false, phi0, v1, v2, omega)
        integrand = 0.5 * dphi_dr**2 + (Omega_phi_vals - Omega_false)
    else:
        integrand = 0.5 * dphi_dr**2 + Omega_phi_vals
    return float(4.0 * np.pi * simpson(r**2 * integrand, x=r))


def compute_energy(
    r: np.ndarray,
    phi: np.ndarray,
    phi0: float,
    v1: float,
    v2: float,
    omega: float,
    phi_false: float,
    *,
    subtract_background: bool = True,
) -> float:
    """
    DEPRECATED: Returns F_ω (grand-canonical functional), not Minkowski energy.
    Use compute_energy_minkowski_1d_spherical for physical energy E_M, or
    compute_Fomega_1d_spherical for the constrained functional F_ω.
    """
    warnings.warn(
        "compute_energy() returns F_ω (grand-canonical functional), not Minkowski energy. "
        "Use compute_energy_minkowski_1d_spherical for physical E_M, or compute_Fomega_1d_spherical for F_ω.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_Fomega_1d_spherical(
        r, phi, phi0, v1, v2, omega, phi_false, subtract_background=subtract_background
    )


def compute_charge_density(r: np.ndarray, phi: np.ndarray, omega: float) -> float:
    """
    Charge density ρ_Q = Q / V for the bounce in a ball of radius r_max = r[-1],
    with V = (4/3)π r_max³.
    """
    charge = compute_charge(r, phi, omega)
    r_max = float(r[-1])
    volume = (4.0 / 3.0) * np.pi * r_max**3
    return charge / volume if volume > 0 else 0.0


def compute_energy_density(
    r: np.ndarray,
    phi: np.ndarray,
    phi0: float,
    v1: float,
    v2: float,
    omega: float,
    phi_false: float,
) -> float:
    """
    Density of F_ω (grand-canonical functional) per volume: ρ = F_ω / V,
    V = (4/3)π r_max³, r_max = r[-1]. For Minkowski energy density use
    compute_energy_minkowski_1d_spherical then divide by V.
    """
    energy = compute_Fomega_1d_spherical(
        r, phi, phi0, v1, v2, omega, phi_false, subtract_background=False
    )
    r_max = float(r[-1])
    volume = (4.0 / 3.0) * np.pi * r_max**3
    return energy / volume if volume > 0 else 0.0


def Q_homogeneous_ball(omega: float, phi_false: float, r_max: float) -> float:
    """
    Charge of the homogeneous configuration φ = φ_false in the ball [0, r_max]:
        Q_hom = 4π ω φ_false² (r_max³/3).
    """
    return float(4.0 * np.pi * omega * (phi_false**2) * (r_max**3 / 3.0))


def Q_homogeneous_ball_from_phi(omega: float, phi_amp: float, r_max: float) -> float:
    """
    Homogeneous charge when the input amplitude is |phi| = rho/sqrt(2):
        Q_hom = 8π ω |phi|^2 (r_max^3/3).
    """
    return float(8.0 * np.pi * omega * (phi_amp**2) * (r_max**3 / 3.0))


def _V_of_rho_array(V_of_rho, rho: np.ndarray) -> np.ndarray:
    """Evaluate V_of_rho on array; return 1D float array."""
    rho = np.asarray(rho, dtype=float)
    out = np.asarray(V_of_rho(rho), dtype=float)
    return out.reshape(rho.shape)


def compute_energy_physical_1d_spherical(
    r: np.ndarray,
    rho: np.ndarray,
    omega: float,
    V_of_rho,
    R_ref: float | None = None,
) -> float:
    """
    Physical Minkowski energy E_M for 1D O(3)-symmetric configuration φ = ρ(r) e^{iωt}.

    𝓔_M = 1/2(∂r ρ)² + 1/2 ω² ρ² + V(ρ)  =>  E_M = 4π ∫_0^{R_ref} dr r² 𝓔_M.

    Same convention for homogeneous and bubble; use for microcanonical / conserved energy comparisons.
    V_of_rho is a callable: V_of_rho(rho) returns V(ρ), e.g. the solver potential U(ρ).

    Parameters
    ----------
    r, rho : 1D arrays (same length), radial grid and profile ρ(r).
    omega : float, chemical potential / frequency.
    V_of_rho : callable, potential V(ρ).
    R_ref : optional. If None, R_ref = r[-1]. Integration domain [0, R_ref]; if r[-1] < R_ref
            the integral is only over [0, r[-1]] (use compute_energy_physical_1d_volume_corrected for extended profile).

    Returns
    -------
    float
        Minkowski energy E_M.
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if r.shape != rho.shape or r.ndim != 1:
        raise ValueError("compute_energy_physical_1d_spherical: r and rho must be 1D arrays with same shape.")
    if len(r) < 2:
        return 0.0

    R_ref = float(R_ref) if R_ref is not None else float(r[-1])
    # Restrict to [0, R_ref]: take points with r <= R_ref; if r[-1] > R_ref add point at R_ref by interpolation
    if r[-1] <= R_ref:
        r_use, rho_use = r, rho
    else:
        mask = r <= R_ref
        if not np.any(mask):
            return 0.0
        r_use = np.append(r[r <= R_ref], R_ref)
        rho_at_R = float(np.interp(R_ref, r, rho))
        rho_use = np.append(rho[r <= R_ref], rho_at_R)

    drho_dr = np.gradient(rho_use, r_use, edge_order=2)
    drho_dr[0] = 0.0  # regularity at origin
    V_vals = _V_of_rho_array(V_of_rho, rho_use)
    integrand = 0.5 * (drho_dr**2) + 0.5 * (omega**2) * (rho_use**2) + V_vals
    return float(4.0 * np.pi * simpson(r_use**2 * integrand, x=r_use))


def compute_energy_physical_1d_volume_corrected(
    r: np.ndarray,
    rho: np.ndarray,
    omega: float,
    V_of_rho,
    r_max_ref: float,
    rho_tail: float,
) -> float:
    """
    Physical Minkowski energy on fixed volume [0, r_max_ref]: bounce profile from 0 to r_max_bubble,
    then ρ = rho_tail from r_max_bubble to r_max_ref. Same volume as homogeneous for fair comparison.
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if r.ndim != 1 or rho.shape != r.shape:
        raise ValueError("r and rho must be 1D arrays with same shape.")
    if len(r) < 2:
        return 0.0

    r_max_bubble = float(r[-1])
    if r_max_bubble >= r_max_ref:
        r_trunc = r[r <= r_max_ref]
        rho_trunc = rho[r <= r_max_ref]
        if len(r_trunc) == 0:
            return 0.0
        if r_trunc[-1] < r_max_ref - 1e-12:
            r_ext = np.append(r_trunc, r_max_ref)
            rho_ext = np.append(rho_trunc, np.interp(r_max_ref, r, rho))
        else:
            r_ext, rho_ext = r_trunc, rho_trunc
    else:
        r_ext = np.append(r, r_max_ref)
        rho_ext = np.append(rho, rho_tail)
    return compute_energy_physical_1d_spherical(r_ext, rho_ext, omega, V_of_rho, R_ref=r_max_ref)


def compute_free_energy_grandcanonical(
    r: np.ndarray,
    rho: np.ndarray,
    omega: float,
    V_of_rho,
    R_ref: float | None = None,
) -> float:
    """
    Grand-canonical functional F_ω = E − ωQ used in EoM (not the physical energy).

    F_ω = 4π ∫ r² [ ½(∂r ρ)² + V(ρ) − ½ ω² ρ² ] dr.

    Do not label as "energy" in plots/prints; use for diagnostics only (e.g. ΔFω).
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if r.shape != rho.shape or r.ndim != 1 or len(r) < 2:
        return 0.0
    R_ref = float(R_ref) if R_ref is not None else float(r[-1])
    if r[-1] <= R_ref:
        r_use, rho_use = r, rho
    else:
        mask = r <= R_ref
        if not np.any(mask):
            return 0.0
        r_use = np.append(r[r <= R_ref], R_ref)
        rho_use = np.append(rho[r <= R_ref], np.interp(R_ref, r, rho))
    drho_dr = np.gradient(rho_use, r_use, edge_order=2)
    drho_dr[0] = 0.0
    V_vals = _V_of_rho_array(V_of_rho, rho_use)
    integrand = 0.5 * (drho_dr**2) + V_vals - 0.5 * (omega**2) * (rho_use**2)
    return float(4.0 * np.pi * simpson(r_use**2 * integrand, x=r_use))


# Alias for explicit "Minkowski energy" in notebook and docs
compute_energy_minkowski_1d_spherical = compute_energy_physical_1d_spherical


def check_homogeneous_consistency_1d(
    omega: float,
    rho0: float,
    V_of_rho,
    r_max: float,
    *,
    tol: float = 1e-5,
) -> None:
    """
    Sanity check: for homogeneous φ = ρ0, E_M = V·(1/2 ω² ρ0² + V(ρ0)) and
    F_ω = V·(V(ρ0) − 1/2 ω² ρ0²).
    Raises AssertionError if computed values differ from these formulas beyond tol.
    """
    V_space = (4.0 / 3.0) * np.pi * r_max**3
    V_at_rho0 = float(np.asarray(V_of_rho(np.array([rho0]))).flat[0])
    E_M_expected = V_space * (0.5 * (omega**2) * (rho0**2) + V_at_rho0)
    F_omega_expected = V_space * (V_at_rho0 - 0.5 * (omega**2) * (rho0**2))

    r = np.linspace(1e-10, r_max, 50)
    rho = np.full_like(r, rho0)
    E_M = compute_energy_physical_1d_spherical(r, rho, omega, V_of_rho, R_ref=r_max)
    F_omega = compute_free_energy_grandcanonical(r, rho, omega, V_of_rho, R_ref=r_max)

    if abs(E_M - E_M_expected) > tol:
        raise AssertionError(
            f"Homogeneous E_M mismatch: computed {E_M:.6e}, expected {E_M_expected:.6e}"
        )
    if abs(F_omega - F_omega_expected) > tol:
        raise AssertionError(
            f"Homogeneous F_ω mismatch: computed {F_omega:.6e}, expected {F_omega_expected:.6e}"
        )


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
    Q_bubble_part = float(simpson(r**2 * phi**2, x=r))

    if r_max_bubble >= r_max_ref:
        mask = r <= r_max_ref
        r_trunc = r[mask]
        phi_trunc = phi[mask]
        if len(r_trunc) == 0:
            phi_at_rmax = float(np.interp(r_max_ref, r, phi))
            Q_bubble_part = float(
                simpson(
                    np.array([0.0, r_max_ref]) ** 2 * np.array([phi[0], phi_at_rmax]) ** 2,
                    x=np.array([0.0, r_max_ref]),
                )
            )
        else:
            if r_trunc[-1] < r_max_ref - 1e-12:
                r_ext = np.append(r_trunc, r_max_ref)
                phi_ext = np.append(phi_trunc, np.interp(r_max_ref, r, phi))
                Q_bubble_part = float(simpson(r_ext**2 * phi_ext**2, x=r_ext))
            else:
                Q_bubble_part = float(simpson(r_trunc**2 * phi_trunc**2, x=r_trunc))
        Q_tail = 0.0
    else:
        Q_tail = (phi_false_tail**2) * (r_max_ref**3 - r_max_bubble**3) / 3.0

    return float(4.0 * np.pi * omega * (Q_bubble_part + Q_tail))


__all__ = [
    "compute_charge",
    "compute_charge_1d_spherical",
    "compute_energy",
    "compute_Fomega_1d_spherical",
    "compute_charge_density",
    "compute_energy_density",
    "Q_homogeneous_ball",
    "Q_homogeneous_ball_from_phi",
    "compute_charge_1d_volume_corrected",
    "compute_energy_physical_1d_spherical",
    "compute_energy_minkowski_1d_spherical",
    "compute_energy_physical_1d_volume_corrected",
    "compute_free_energy_grandcanonical",
    "check_homogeneous_consistency_1d",
]
