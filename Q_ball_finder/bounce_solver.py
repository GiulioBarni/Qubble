"""
General-purpose O(d) bounce solver with support for potentials that are either
bistable or unbounded on one side.  The implementation distils the logic that
was prototyped in the original notebook into a reusable function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar

ArrayLike = np.ndarray | float


# --------------------------------------------------------------------------- #
# Numerical utilities
# --------------------------------------------------------------------------- #


def numerical_gradient(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def numerical_hessian(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def find_local_extrema(
    V: Callable[[float], float],
    x_min: float,
    x_max: float,
    ngrid: int = 2001,
) -> list[Dict[str, float]]:
    """
    Scan the interval [x_min, x_max] and locate local minima/maxima.  The
    function is intentionally brute-force but robust, relying on a coarse grid
    and local one-dimensional optimisations to refine each extremum.
    """
    xs = np.linspace(x_min, x_max, ngrid)
    Vs = np.array([V(x) for x in xs], dtype=float)
    extrema: list[Dict[str, float]] = []

    dVd = np.gradient(Vs, xs)
    for i in range(1, len(xs) - 1):
        left, mid, right = dVd[i - 1], dVd[i], dVd[i + 1]

        # candidate maximum
        if left > 0.0 and right < 0.0:
            res = minimize_scalar(lambda z: -V(z), bounds=(xs[i - 1], xs[i + 1]), method="bounded")
            if res.success:
                x0 = res.x
                kind = "max" if numerical_hessian(V, x0) < 0 else "min"
                extrema.append({"x": x0, "V": V(x0), "type": kind})

        # candidate minimum
        if left < 0.0 and right > 0.0:
            res = minimize_scalar(lambda z: V(z), bounds=(xs[i - 1], xs[i + 1]), method="bounded")
            if res.success:
                x0 = res.x
                kind = "min" if numerical_hessian(V, x0) > 0 else "max"
                extrema.append({"x": x0, "V": V(x0), "type": kind})

    # merge duplicates
    extrema_sorted: list[Dict[str, float]] = []
    for e in sorted(extrema, key=lambda d: d["x"]):
        if not extrema_sorted or abs(e["x"] - extrema_sorted[-1]["x"]) > 1e-3:
            extrema_sorted.append(e)

    return extrema_sorted


def identify_false_true_vacua(
    V: Callable[[float], float],
    x_min: float,
    x_max: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    extrema = find_local_extrema(V, x_min, x_max)
    mins = [e for e in extrema if e["type"] == "min"]
    if len(mins) < 2:
        raise RuntimeError("Need at least two minima to identify false/true vacua.")
    mins_sorted = sorted(mins, key=lambda d: d["V"])
    true_vac = mins_sorted[0]
    false_vac = mins_sorted[-1]
    return false_vac, true_vac


def identify_false_vac_and_top_for_unbounded(
    V: Callable[[float], float],
    x_min: float,
    x_max: float,
    prefer_side: Optional[str] = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    extrema = find_local_extrema(V, x_min, x_max)
    mins = [e for e in extrema if e["type"] == "min"]
    maxs = [e for e in extrema if e["type"] == "max"]
    if not mins:
        raise RuntimeError("Unbounded case: no local minimum found near the origin.")

    false_vac = sorted(mins, key=lambda d: abs(d["x"]))[0]
    phi_f = false_vac["x"]

    if prefer_side not in {None, "+", "-"}:
        raise ValueError("prefer_side must be in {None, '+', '-'}")

    if prefer_side is None:
        candidates = sorted(maxs, key=lambda d: abs(d["x"] - phi_f))
    elif prefer_side == "+":
        candidates = [m for m in maxs if m["x"] > phi_f]
        candidates = sorted(candidates, key=lambda d: abs(d["x"] - phi_f))
    else:
        candidates = [m for m in maxs if m["x"] < phi_f]
        candidates = sorted(candidates, key=lambda d: abs(d["x"] - phi_f))

    if not candidates:
        # fallback: try the opposite side, otherwise take the closest maximum
        other = [m for m in maxs if (prefer_side == "+" and m["x"] < phi_f) or (prefer_side == "-" and m["x"] > phi_f)]
        pool = other if other else maxs
        if not pool:
            raise RuntimeError("Unbounded case: could not locate a barrier maximum in the scan range.")
        candidates = sorted(pool, key=lambda d: abs(d["x"] - phi_f))

    top = candidates[0]
    return false_vac, top


# --------------------------------------------------------------------------- #
# Bounce solver
# --------------------------------------------------------------------------- #


@dataclass
class BounceSolution:
    r: np.ndarray
    phi: np.ndarray
    phip: np.ndarray
    SE: float
    phi0: float
    r_end: float
    false_vac: Optional[Dict[str, float]]
    true_vac: Optional[Dict[str, float]]
    top: Optional[Dict[str, float]]
    d: int
    metadata: Dict[str, float]


def solve_bounce(
    V: Callable[[float], float],
    dV: Optional[Callable[[float], float]] = None,
    *,
    d: int = 3,
    r0: float = 1e-6,
    rmax: float = 200.0,
    max_step: float = 0.01,
    x_min: float = -5.0,
    x_max: float = 5.0,
    tol: float = 1e-8,
    verbose: bool = False,
    allow_unbounded: bool = False,
    phi0_cap: Optional[float] = None,
    prefer_side: Optional[str] = None,
) -> BounceSolution:
    """
    Solve the radial bounce equation for an O(d)-symmetric configuration.
    The effective potential `V` is expected to already include the ω-dependent
    term when solving for Q-balls.

    Parameters
    ----------
    V, dV : callable
        Potential and its derivative with respect to the field.  If `dV` is
        omitted, a numerical gradient is used.
    d : int
        Dimensionality (1 ≤ d ≤ 4).  d=3 corresponds to standard Q-ball bounce.
    allow_unbounded : bool
        Enable the alternative bracketing when only one minimum is available.
        Requires a finite `phi0_cap`.
    prefer_side : {'+', '-', None}
        Guidance for which side of the false vacuum to look for the barrier
        maximum in the unbounded case.
    """
    if dV is None:
        dV = lambda x: numerical_gradient(V, x)  # noqa: E731

    metadata: Dict[str, float] = {}
    use_unbounded = False

    # --- Step 1: identify the relevant extrema and bracket φ(0)
    try:
        false_vac, true_vac = identify_false_true_vacua(V, x_min, x_max)
        phi_f, phi_t = false_vac["x"], true_vac["x"]
        metadata.update({"phi_f": phi_f, "phi_t": phi_t})

        a, b = sorted([phi_f, phi_t])
        res_top = minimize_scalar(lambda z: -V(z), bounds=(a, b), method="bounded")
        phi_top = res_top.x if res_top.success else 0.5 * (a + b)

        left = phi_t + 1e-4 * np.sign(phi_top - phi_t)
        right = phi_top - 1e-4 * np.sign(phi_top - phi_t)
        if left > right:
            left, right = right, left
        true_vac_dict: Optional[Dict[str, float]] = {"x": phi_t, "V": V(phi_t)}
        top_dict: Optional[Dict[str, float]] = {"x": phi_top, "V": V(phi_top)}

        if verbose:
            print(f"[bistable] phi_f={phi_f:.6g}, phi_t={phi_t:.6g}, phi_top={phi_top:.6g}")
            print(f"[bistable] Bracket phi(0): [{left:.6g}, {right:.6g}]")

    except Exception:
        if not allow_unbounded or phi0_cap is None:
            raise
        use_unbounded = True
        false_vac, top = identify_false_vac_and_top_for_unbounded(V, x_min, x_max, prefer_side)
        phi_f = false_vac["x"]
        phi_top = top["x"]
        metadata.update({"phi_f": phi_f, "phi_top": phi_top})

        sgn = np.sign(phi_top - phi_f) or 1.0
        left = phi_top + 1e-4 * sgn
        right = phi0_cap * sgn
        if (sgn > 0 and left > right) or (sgn < 0 and left < right):
            right = left + sgn * abs(phi0_cap)
        true_vac_dict = None
        top_dict = {"x": phi_top, "V": V(phi_top)}

        if verbose:
            print(f"[unbounded] phi_f={phi_f:.6g}, phi_top={phi_top:.6g}, user cap={phi0_cap}")
            print(f"[unbounded] Bracket phi(0): [{left:.6g}, {right:.6g}]")

    false_vac_dict = {"x": phi_f, "V": V(phi_f)}

    # --- Step 2: set up ODE system and events
    def ode(r: float, y: np.ndarray) -> np.ndarray:
        phi, phip = y
        if r == 0.0:
            friction = 0.0
        else:
            friction = (d - 1) / r * phip
        return np.array([phip, -friction + dV(phi)])

    def ev_reach_false(r: float, y: np.ndarray) -> float:
        return y[0] - phi_f

    ev_reach_false.terminal = True
    ev_reach_false.direction = 0

    def ev_turn_up(r: float, y: np.ndarray) -> float:
        return y[1]

    ev_turn_up.terminal = True
    ev_turn_up.direction = 1

    # --- Step 3: shooting on φ(0)
    phi0_L, phi0_R = left, right
    sol_best = None
    hit = None

    for _ in range(100):
        phi0 = 0.5 * (phi0_L + phi0_R)
        sol = solve_ivp(
            ode,
            (r0, rmax),
            y0=np.array([phi0, 0.0]),
            events=(ev_reach_false, ev_turn_up),
            dense_output=True,
            max_step=max_step,
            rtol=1e-8,
            atol=1e-10,
        )

        reached_false = sol.t_events[0].size > 0
        turned_up = sol.t_events[1].size > 0

        if reached_false:
            r_end = sol.t_events[0][0]
            hit = "false"
            sol_best = (sol, r_end, phi0)
            phi0_R = phi0  # overshoot
        elif turned_up:
            r_end = sol.t_events[1][0]
            hit = "turn"
            sol_best = (sol, r_end, phi0)
            phi0_L = phi0  # undershoot
        else:
            r_end = sol.t[-1]
            sol_best = (sol, r_end, phi0)
            break

        if abs(phi0_R - phi0_L) < tol:
            break

    if sol_best is None:
        raise RuntimeError("Shooting failed: no solution found within the iteration limit.")

    sol, r_end, phi0 = sol_best
    metadata.update({"phi0": phi0, "r_end": r_end})

    r_grid = np.linspace(r0, r_end, 5000)
    phi_grid = sol.sol(r_grid)[0]
    phip_grid = sol.sol(r_grid)[1]

    # --- Step 4: action S_E (shifted by V(false))
    Vf = false_vac_dict["V"]

    if d == 1:
        Omega = 2.0
    elif d == 2:
        Omega = 2.0 * np.pi
    elif d == 3:
        Omega = 4.0 * np.pi
    elif d == 4:
        Omega = 2.0 * np.pi**2
    else:
        raise ValueError("Supported dimensions are d = 1, 2, 3, 4.")

    def density(r: float) -> float:
        val = sol.sol(r)
        phi_val, phip_val = val[0], val[1]
        return r ** (d - 1) * (0.5 * phip_val**2 + V(phi_val) - Vf)

    SE_radial, _ = quad(lambda rr: density(rr), r0, r_end, epsabs=1e-10, epsrel=1e-8, limit=500)
    SE = Omega * SE_radial
    metadata["SE"] = SE

    if verbose:
        tag = "unbounded" if use_unbounded else "bistable"
        print(f"[{tag}] phi(0) ≈ {phi0:.8g} | r_end ≈ {r_end:.6g} | event={hit}")
        print(f"S_E (O({d})) = {SE:.10g}")

    return BounceSolution(
        r=r_grid,
        phi=phi_grid,
        phip=phip_grid,
        SE=SE,
        phi0=phi0,
        r_end=r_end,
        false_vac=false_vac_dict,
        true_vac=true_vac_dict,
        top=top_dict,
        d=d,
        metadata=metadata,
    )


__all__ = [
    "ArrayLike",
    "BounceSolution",
    "find_local_extrema",
    "identify_false_true_vacua",
    "identify_false_vac_and_top_for_unbounded",
    "numerical_gradient",
    "numerical_hessian",
    "solve_bounce",
]

