"""
1D O(d)-symmetric bounce solution for bubble nucleation.

Solves the radial equation φ'' + (d-1)/r φ' = dΩ/dφ with overshoot/undershoot
bisection on φ(0). Uses vacua_of_Omega and dOmega_dphi from potential_bubble.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from Bubble_finder.potential_bubble import vacua_of_Omega, dOmega_dphi, Omega_phi
from Bubble_finder.observables_1d import (
    compute_charge,
    compute_energy,
    compute_charge_density,
    compute_energy_density,
)


def solve_bounce(
    phi0: float,
    v1: float,
    v2: float,
    omega: float,
    d: int = 3,
    r0: float = 1e-6,
    rmax: float = 500.0,
    max_iter: int = 100,
    verbose: bool = False,
    extend_to: float | None = None,
    n_grid_points: int = 1000,
):
    """
    Robust bounce solution finder using overshoot/undershoot with events.

    Uses bisection on φ(0) with events:
    - event_phi_false: φ crosses phi_false (overshoot)
    - event_dphi_zero: φ' crosses zero (undershoot)

    Parameters
    ----------
    phi0, v1, v2, omega : float
        Potential parameters: phi0 (field value at center), v1 (false vacuum),
        v2 (true vacuum), omega (chemical potential)
    d : int
        Spatial dimension in O(d) reduction (default 3 for O(3), use 4 for O(4)-like radial 1D).
    r0, rmax : float
        Integration range
    max_iter : int
        Maximum bisection iterations
    verbose : bool
        Print progress
    extend_to : float, optional
        If provided, extend solution to this r value after reaching phi_false
        (solution stays constant at phi_false beyond the event point)
    n_grid_points : int
        Number of points in the solution grid (default 1000)

    Returns
    -------
    r_grid, phi_grid, phi0_final, phi_false, phi_true
        Solution arrays and initial/final values. Returns (None, None, None, None, None)
        if the solver fails.
    """
    phi_false, phi_true = vacua_of_Omega(phi0, v1, v2, omega, verbose=verbose)
    if verbose:
        print(f"Vacua of Ω: phi_false={phi_false:.10f}, phi_true={phi_true:.10f}")

    phi_L = min(phi_true, phi_false)
    phi_R = max(phi_true, phi_false)

    def event_phi_false(r, y):
        return y[0] - phi_false

    event_phi_false.terminal = True
    event_phi_false.direction = -1

    def event_dphi_zero(r, y):
        return y[1] if r > 10 * r0 else 1.0

    event_dphi_zero.terminal = True
    event_dphi_zero.direction = 1

    r_stop = rmax
    sol_best = None
    phi0_best = None

    for iteration in range(max_iter):
        phi0_init = 0.5 * (phi_L + phi_R)

        sol = solve_ivp(
            fun=lambda r, y: [
                y[1],
                -(d - 1) / r * y[1] + dOmega_dphi(y[0], phi0, v1, v2, omega),
            ],
            t_span=(r0, rmax),
            y0=[phi0_init, 0.0],
            method="BDF",
            events=[event_phi_false, event_dphi_zero],
            dense_output=True,
            max_step=np.inf,
            atol=1e-8,
            rtol=1e-8,
        )

        if not sol.success:
            if verbose:
                print(f"  Iteration {iteration}: Integration failed for φ(0)={phi0_init:.6f}")
            phi_L = phi_L + 0.01 * (phi0_init - phi_L)
            phi_R = phi_R - 0.01 * (phi_R - phi0_init)
            continue

        t_phi_event = sol.t_events[0][0] if sol.t_events[0].size > 0 else np.inf
        t_dphi_event = sol.t_events[1][0] if sol.t_events[1].size > 0 else np.inf

        phi_end = sol.y[0, -1]
        dphi_end = sol.y[1, -1]

        if t_dphi_event < t_phi_event:
            phi_L = phi0_init
            r_stop = t_dphi_event
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: Undershoot, φ(0)={phi0_init:.6f}, φ(∞)={phi_end:.6f}, r_stop={r_stop:.4f}")
        elif t_phi_event < np.inf:
            phi_R = phi0_init
            r_stop = t_phi_event
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: Overshoot, φ(0)={phi0_init:.6f}, φ(∞)={phi_end:.6f}, r_stop={r_stop:.4f}")
        else:
            if phi_end > phi_false:
                phi_L = phi0_init
                if verbose and iteration % 10 == 0:
                    print(f"  Iteration {iteration}: No event, above false, φ(0)={phi0_init:.6f}, φ(∞)={phi_end:.6f}")
            else:
                phi_R = phi0_init
                if verbose and iteration % 10 == 0:
                    print(f"  Iteration {iteration}: No event, below false, φ(0)={phi0_init:.6f}, φ(∞)={phi_end:.6f}")

        sol_best = sol
        phi0_best = phi0_init

        if abs(phi_R - phi_L) < 1e-8:
            if verbose:
                print(f"  Converged after {iteration+1} iterations: φ(0)={phi0_init:.6f}, φ(∞)={phi_end:.6f}, φ'(∞)={dphi_end:.3e}")
            break

    if sol_best is None or not sol_best.success:
        if verbose:
            print("  Failed to find solution")
        return None, None, None, None, None

    # Final clean integration at phi0_best with absolute false-vacuum event.
    tol_event = 1e-8

    def event_phi_false_abs(r, y):
        return abs(y[0] - phi_false) - tol_event

    event_phi_false_abs.terminal = True
    event_phi_false_abs.direction = 0

    def event_dphi_zero_final(r, y):
        return y[1] if r > 10 * r0 else 1.0

    event_dphi_zero_final.terminal = True
    event_dphi_zero_final.direction = 1

    sol_final = solve_ivp(
        fun=lambda r, y: [
            y[1],
            -(d - 1) / r * y[1] + dOmega_dphi(y[0], phi0, v1, v2, omega),
        ],
        t_span=(r0, rmax),
        y0=[float(phi0_best), 0.0],
        method="BDF",
        events=[event_phi_false_abs, event_dphi_zero_final],
        dense_output=True,
        max_step=np.inf,
        atol=1e-8,
        rtol=1e-8,
    )

    if not sol_final.success:
        # fallback to last best integration
        sol_final = sol_best

    t_phi_abs = sol_final.t_events[0][0] if sol_final.t_events[0].size > 0 else np.inf
    t_dphi_zero = sol_final.t_events[1][0] if sol_final.t_events[1].size > 0 else np.inf
    r_end = float(min(t_phi_abs, t_dphi_zero, sol_final.t[-1]))

    r_grid = np.linspace(r0, r_end, n_grid_points)
    phi_grid = sol_final.sol(r_grid)[0]

    # Snap endpoint exactly to false vacuum if we reached (or nearly reached) it.
    if np.isfinite(t_phi_abs) or abs(phi_grid[-1] - phi_false) < 1e-6:
        phi_grid[-1] = phi_false

    # Extend with constant false vacuum tail if requested.
    if extend_to is not None and extend_to > r_end:
        n_extend = int(n_grid_points * (extend_to - r_end) / max(r_end - r0, 1e-12))
        n_extend = max(200, min(n_extend, 1000))
        r_extend = np.linspace(r_end, extend_to, n_extend)
        phi_extend = np.full_like(r_extend, phi_false)
        r_grid = np.concatenate([r_grid, r_extend[1:]])
        phi_grid = np.concatenate([phi_grid, phi_extend[1:]])
    else:
        # Ensure last point is exactly the false vacuum for downstream interpolation stability.
        phi_grid[-1] = phi_false

    phi0_final = float(phi0_best) if phi0_best is not None else float(phi0)
    return r_grid, phi_grid, phi0_final, phi_false, phi_true


def solve_bounce_O3(
    phi0: float,
    v1: float,
    v2: float,
    omega: float,
    **kwargs,
):
    """Convenience wrapper for O(3): solve_bounce(..., d=3, ...)."""
    return solve_bounce(phi0, v1, v2, omega, d=3, **kwargs)


def solve_bounce_O4(
    phi0: float,
    v1: float,
    v2: float,
    omega: float,
    **kwargs,
):
    """Convenience wrapper for O(4): solve_bounce(..., d=4, ...)."""
    return solve_bounce(phi0, v1, v2, omega, d=4, **kwargs)


# Re-export 1D observables from central module (bounce_1d remains the notebook entry point)
__all__ = [
    "solve_bounce",
    "solve_bounce_O3",
    "solve_bounce_O4",
    "compute_energy",
    "compute_charge",
    "compute_energy_density",
    "compute_charge_density",
]
