"""
observables_1d.py — 1D-only charge and energy for O(3) bounce.

Single source of truth for 1D observables: Q, E, charge/energy density profiles,
homogeneous ball Q/E in finite volume. 2D code that needs these imports from here.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .potential_bubble import Omega_phi


def compute_charge(r: np.ndarray, phi: np.ndarray, omega: float) -> float:
    """
    Charge for 1D O(3) bounce: Q = 4π ω ∫ dr r² φ².
    Uses Simpson's rule for integration.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return float(4.0 * np.pi * omega * simpson(r**2 * phi**2, x=r))


def compute_energy(
    r: np.ndarray,
    phi: np.ndarray,
    phi0: float,
    v1: float,
    v2: float,
    omega: float,
    phi_false: float,
) -> float:
    """
    Energy for 1D O(3) bounce: E = 4π ∫ dr r² [ ½ (dφ/dr)² + Ω(φ) - Ω(φ_false) ].
    Uses Simpson's rule for integration.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    dphi_dr = np.gradient(phi, r, edge_order=2)
    dphi_dr[0] = 0.0  # Regularity at origin
    Omega_phi_vals = np.array([Omega_phi(phi_i, phi0, v1, v2, omega) for phi_i in phi])
    Omega_false = Omega_phi(phi_false, phi0, v1, v2, omega)
    integrand = 0.5 * dphi_dr**2 + (Omega_phi_vals - Omega_false)
    return float(4.0 * np.pi * simpson(r**2 * integrand, x=r))


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
    Energy density ρ_E = E / V for the bounce in a ball of radius r_max = r[-1],
    with V = (4/3)π r_max³.
    """
    energy = compute_energy(r, phi, phi0, v1, v2, omega, phi_false)
    r_max = float(r[-1])
    volume = (4.0 / 3.0) * np.pi * r_max**3
    return energy / volume if volume > 0 else 0.0


def Q_homogeneous_ball(omega: float, phi_false: float, r_max: float) -> float:
    """
    Charge of the homogeneous configuration φ = φ_false in the ball [0, r_max]:
        Q_hom = 4π ω φ_false² (r_max³/3).
    """
    return float(4.0 * np.pi * omega * (phi_false**2) * (r_max**3 / 3.0))


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
    "compute_energy",
    "compute_charge_density",
    "compute_energy_density",
    "Q_homogeneous_ball",
    "compute_charge_1d_volume_corrected",
]
