"""
Utilities for computing physical observables of Q-ball solutions obtained
with the radial bounce solver.

The functions operate on ``BounceSolution`` objects returned by
``bounce_solver.solve_bounce`` when the effective potential corresponds to
the Q-ball case (i.e. ``V_hat(χ) = V(χ) - ½ ω² χ²``).  In that setting the
field profile stored in ``BounceSolution.phi`` is the real radial modulus
``χ(r)`` of the complex scalar.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .bounce_solver import BounceSolution

ArrayLike = np.ndarray | float


def _ensure_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def compute_dimensionless_charge(
    solution: BounceSolution,
    *,
    m: float,
    v: float,
    omega: float,
) -> float:
    """
    Dimensionless charge g²Q for a Q-ball profile:

      g² Q = 4π · (ω/m) ∫ dr̃ r̃² ϕ²,       r̃ = m r,   ϕ = χ / v,

    where χ(r) = solution.phi.  Used to locate the critical charge.
    """
    r_phys = _ensure_array(solution.r)
    chi = _ensure_array(solution.phi)

    r_tilde = m * r_phys
    varphi = chi / v

    integral = np.trapz((r_tilde**2) * (varphi**2), r_tilde)
    return 4.0 * np.pi * (omega / m) * integral


def compute_energy(
    solution: BounceSolution,
    *,
    omega: float,
    potential_chi: Callable[[ArrayLike], ArrayLike],
    V_at_zero: Optional[float] = None,
) -> float:
    """
    Compute the Q-ball energy above the vacuum:

      E = 4π ∫ dr r² [ ½ (dχ/dr)² + ½ ω² χ² + (V_chi(χ) - V_chi(0)) ].

    Parameters
    ----------
    solution : BounceSolution
        Output of ``solve_bounce`` where ``phi`` is interpreted as χ(r).
    omega : float
        Frequency used to build the effective potential.
    potential_chi : callable
        The original potential V(χ) WITHOUT the ``-½ ω² χ²`` term.
    V_at_zero : float, optional
        Value of V(χ) at χ = 0.  If omitted it is evaluated on the fly.
    """
    r = _ensure_array(solution.r)
    chi = _ensure_array(solution.phi)

    if solution.phip is not None:
        dchi_dr = _ensure_array(solution.phip)
    else:
        dchi_dr = np.gradient(chi, r, edge_order=2)
        dchi_dr[0] = 0.0  # regularity at the origin

    if V_at_zero is None:
        V0 = float(potential_chi(0.0))
    else:
        V0 = float(V_at_zero)

    V_vals = _ensure_array(potential_chi(chi))

    integrand = 0.5 * dchi_dr * dchi_dr + 0.5 * omega * omega * chi * chi + (V_vals - V0)
    energy = 4.0 * np.pi * np.trapz(r * r * integrand, r)
    return energy


def compute_charge(
    solution: BounceSolution,
    *,
    omega: float,
) -> float:
    """
    Compute the physical Q (with mass dimension) assuming χ = |φ|.  This
    corresponds to

      Q = 4π · 2 ω ∫ dr r² χ² = 8π ω ∫ dr r² χ².
    """
    r = _ensure_array(solution.r)
    chi = _ensure_array(solution.phi)
    integral = np.trapz(r * r * chi * chi, r)
    return 8.0 * np.pi * omega * integral


__all__ = ["compute_dimensionless_charge", "compute_energy", "compute_charge"]

