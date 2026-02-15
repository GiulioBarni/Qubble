# ansatz_bubble.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Iterable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    sp = None
    spla = None


# ---------------------------------------------------------------------
# Q-ball adapter: Grid2D, Fields, PotentialModel, ResidualOperator, QCalculator
# (bounce2d uses (Nr, Nt); ansatz uses (Nt, Nr) row=tau, col=r)
# ---------------------------------------------------------------------

from .bounce2d import Bubble2DSettings, Bubble2DSolver, solve_rho0_for_omega


@dataclass
class Grid2D:
    """Grid for ansatz/selection: .r (Nr,), .tau (Nt,), .T (= beta/2), .Nr, .Nt, .dr, .dtau."""
    r: np.ndarray
    tau: np.ndarray
    Nr: int
    Nt: int
    T: float
    dr: float
    dtau: float

    @classmethod
    def from_solver(cls, solver: Bubble2DSolver) -> "Grid2D":
        g = solver.grid
        return cls(
            r=np.asarray(g.r),
            tau=np.asarray(g.tau),
            Nr=int(solver.Nr),
            Nt=int(solver.Nt),
            T=float(solver.settings.beta) / 2.0,
            dr=float(g.dr),
            dtau=float(g.dtau),
        )


class Fields:
    """Fields for ansatz: .grid, .phi_false (= rho0), .pack(Y,YB), .unpack(x), .reconstruct(y,ybar)."""

    def __init__(self, solver: Bubble2DSolver):
        self._solver = solver
        self._grid2d = Grid2D.from_solver(solver)

    @property
    def grid(self) -> Grid2D:
        return self._grid2d

    @property
    def phi_false(self) -> float:
        return float(self._solver.rho0)

    def set_omega(self, omega: float) -> None:
        self._solver.omega = float(omega)
        self._solver.rho0 = float(
            solve_rho0_for_omega(
                omega,
                self._solver.dU,
                self._solver.settings.rho0_bracket,
            )
        )

    def pack(self, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
        """(Nt, Nr) -> (Nr, Nt) then pack via solver."""
        yr = np.asarray(y).T
        ybr = np.asarray(ybar).T
        return self._solver.pack(yr, ybr)

    def unpack(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Packed -> solver.unpack -> (Nr, Nt) -> return (Nt, Nr)."""
        yr, ybr = self._solver.unpack(x0)
        return yr.T, ybr.T

    def reconstruct(self, y: np.ndarray, ybar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(Nt, Nr) -> (Nr, Nt) -> phi, phibar."""
        yr = np.asarray(y).T
        ybr = np.asarray(ybar).T
        return self._solver.phi(yr, ybr)


class PotentialModel:
    """Potential interface for compute_negative_mode_1d: dV_ds(s), d2V_ds2(s) with s = phi*phibar (real)."""

    def __init__(self, U: Callable, dU: Callable, d2U: Callable):
        self._U = U
        self._dU = dU
        self._d2U = d2U

    def dV_ds(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        rho = np.sqrt(np.maximum(s, 1e-20))
        return np.asarray(self._dU(rho) / (2.0 * rho), dtype=float)

    def d2V_ds2(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        rho = np.sqrt(np.maximum(s, 1e-20))
        dU = self._dU(rho)
        d2U = self._d2U(rho)
        return np.asarray((d2U * rho - dU) / (4.0 * rho**3), dtype=float)


class ResidualOperator:
    """residual(x0, omega_bulk, eta=None): set solver state then return solver.residual(x0)."""

    def __init__(self, solver: Bubble2DSolver, fields: Fields):
        self._solver = solver
        self._fields = fields

    def residual(self, x0: np.ndarray, omega_bulk: float, eta: Any = None) -> np.ndarray:
        self._fields.set_omega(omega_bulk)
        if eta is not None:
            # eta0 is defined by the solver and must be passed without rescaling.
            # Do NOT use 2.0*eta: the physics eta0 is the twist parameter directly.
            self._solver.eta0 = float(eta)
        return self._solver.residual(x0)


class QCalculator:
    """compute(phi, phibar, omega, grid) -> Q (real)."""

    def __init__(self, solver: Bubble2DSolver):
        self._solver = solver

    def compute(
        self,
        phi: np.ndarray,
        phibar: np.ndarray,
        omega: float,
        grid: Grid2D,
    ) -> float:
        r = np.asarray(grid.r)[:, None]
        tau = np.asarray(grid.tau)[None, :]
        rho0 = float(self._solver.rho0)
        phi_arr = np.asarray(phi)
        phibar_arr = np.asarray(phibar)
        yr = r * (np.exp(-omega * tau) * phi_arr - rho0)
        ybr = r * (np.exp(omega * tau) * phibar_arr - rho0)
        return float(self._solver.compute_charge(yr, ybr, 0).real)


def make_q_ball_objects(
    solver: Bubble2DSolver,
) -> Tuple[Grid2D, Fields, PotentialModel, ResidualOperator, "QCalculator"]:
    """Build adapter objects from a Bubble2DSolver for ansatz / selection."""
    grid2d = Grid2D.from_solver(solver)
    fields = Fields(solver)
    potential = PotentialModel(solver.U, solver.dU, solver.d2U)
    res_op = ResidualOperator(solver, fields)
    qcalc = QCalculator(solver)
    return grid2d, fields, potential, res_op, qcalc


def make_bubble_profile_1d_from_arrays(
    r_bubble: np.ndarray,
    phi_bubble: np.ndarray,
) -> Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return bubble_profile_1d(omega_tilde, r) by interpolating (r_bubble, phi_bubble). Used with build_seed_bubble."""
    from scipy.interpolate import interp1d
    r_b = np.asarray(r_bubble, dtype=float)
    phi_b = np.asarray(phi_bubble, dtype=float)
    if r_b.size < 2:
        def _f(_omega_tilde: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            r = np.asarray(r, dtype=float)
            v = float(phi_b.flat[0]) if phi_b.size else 0.0
            p = np.full_like(r, v)
            return p, p.copy()
        return _f
    f = interp1d(
        r_b, phi_b, kind="linear", bounds_error=False,
        fill_value=(float(phi_b[0]), float(phi_b[-1])),
    )

    def bubble_profile_1d(omega_tilde: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        p = np.asarray(f(r), dtype=float)
        return p, p.copy()

    return bubble_profile_1d


def make_bubble_profile_1d_from_solve_bounce(
    solve_bounce_fn: Callable,
    phi0: float,
    v1: float,
    v2: float,
) -> Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return bubble_profile_1d(omega_tilde, r) from solve_bounce_fn(phi0, v1, v2, omega) -> r_bounce, phi_bounce, ..."""
    from scipy.interpolate import interp1d
    def bubble_profile_1d(omega_tilde: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        out = solve_bounce_fn(phi0, v1, v2, float(omega_tilde))
        r_bounce, phi_bounce = out[0], out[1]
        r_bounce = np.asarray(r_bounce)
        phi_bounce = np.asarray(phi_bounce)
        if r_bounce.size < 2:
            phi1d = np.full_like(r, phi_bounce.flat[0] if phi_bounce.size else v1)
            return phi1d.copy(), phi1d.copy()
        f = interp1d(r_bounce, phi_bounce, kind="linear", bounds_error=False, fill_value=(phi_bounce[0], phi_bounce[-1]))
        phi1d = np.asarray(f(r), dtype=float)
        return phi1d.copy(), phi1d.copy()
    return bubble_profile_1d


# ---------------------------------------------------------------------
# Ansatz and BCs
# ---------------------------------------------------------------------


@dataclass
class AnsatzParams:
    eps: float = 0.02         # kick amplitude along negative mode
    k: int = 1                # tau harmonic
    phase: float = 0.0        # tau phase
    amp: float = 1.0          # overall amplitude of the base (usually 1.0 if y1d already “correct”)
    tau_gate_frac: float = 0.15
    r_window_frac: float = 0.15


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 3.0*x*x - 2.0*x*x*x


def tau_gate(tau: np.ndarray, T: float, frac: float) -> np.ndarray:
    # gate=0 at endpoints, ~1 in the middle
    dist = np.minimum(tau + 0.5*T, 0.5*T - tau)
    w = frac*T
    g = np.ones_like(tau, float)
    mask = dist < w
    g[mask] = smoothstep(dist[mask] / w)
    return g


def radial_window(r: np.ndarray, rmax: float, frac: float) -> np.ndarray:
    # window=1 in bulk, goes to 0 near rmax
    w0 = (1.0 - frac) * rmax
    w = np.ones_like(r, float)
    mask = r >= w0
    if np.any(mask):
        x = (r[mask] - w0) / (rmax - w0)
        w[mask] = 1.0 - smoothstep(x)
    return w


import numpy as np
from typing import Tuple

def bc_projector_y(
    y: np.ndarray,
    ybar: np.ndarray,
    *,
    fields,
    eta: float,
    sign_convention: int = +1,
    tau_mode: str = "twisted",   # only "twisted" supported (matches Bubble2DSolver tau_bc)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete BC projector in Q-ball-style variables y=r(phi_rot-rho0), ybar=r(phibar_rot-rho0).
    Consistent with Bubble2DSolver tau_bc="twisted" only. No half-box assumptions.

    Radial BCs (always):
      - r=0: if r[0]==0, enforce y=0 and ybar=0 (regularity of y variables)
      - r=rmax: Neumann in phi, phibar -> y_r - y/r = 0 (backward diff)
                => y_N = y_{N-1} / (1 - dr/rmax)

    Tau BCs (twisted):
      - The solver applies twist closure on ghost points; we do NOT overwrite tau slices.
      - Interior values come from the ansatz. The twist is enforced by the solver's residual.
      - No Dirichlet at tau=-beta/2 or tau=0.
    """
    g = fields.grid
    r = np.asarray(g.r, dtype=float)
    dr = float(g.dr)
    rmax = float(r[-1])

    y = y.copy()
    ybar = ybar.copy()

    # tau_mode "twisted": do not overwrite tau boundary slices. Solver enforces twist via ghosts.
    if tau_mode != "twisted":
        raise ValueError(f"Only tau_mode='twisted' is supported. Got '{tau_mode}'.")

    # -------------------------
    # r=0 regularity in y variables (only if r[0]==0)
    # -------------------------
    if abs(r[0]) <= 1e-14:
        y[:, 0] = 0.0
        ybar[:, 0] = 0.0

    # -------------------------
    # r=rmax Neumann in phi,phibar: y_r - y/r = 0
    # -------------------------
    denom = (1.0 - dr / rmax)
    if abs(denom) > 1e-14:
        alpha = 1.0 / denom  # y_N = alpha * y_{N-1}
        y[:, -1] = alpha * y[:, -2]
        ybar[:, -1] = alpha * ybar[:, -2]

    return y, ybar



import numpy as np
from typing import Tuple, Optional, Dict, Any

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


def compute_Dp_shooting_1d(
    *,
    r: np.ndarray,
    phi: np.ndarray,
    phibar: np.ndarray,
    potential,                      # must provide dV_ds(s), d2V_ds2(s)
    omega: float,
    gamma_scan: Tuple[float, float] = (0.01, 0.5),
    n_scan: int = 80,
    ode_method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    interp_kind: str = "cubic",
    max_step: Optional[float] = None,
    sign_convention: int = +1,
    require_bracket: bool = True,   # if False, return gamma_grid/Dp_vals even when no sign change (for plotting)
) -> Dict[str, Any]:
    """
    Q-ball-faithful coupled shooting for the unstable mode.

    We work with *regular* variables y_R=r*xi_R and y_I=r*xi_I
    so regularity at r=0 is y_R(0)=y_I(0)=0 (like Q-ball).
    Neumann on the underlying field at r=rmax translates to:
        y'(rmax) - y(rmax)/rmax = 0  (applied to both components).

    The coupled ODE is the same structure as in your Q-ball compute_unstable_mode,
    but with V1,V2 obtained from PotentialModel for V(s), s=rho^2:

        rho(r) = sqrt( Re(phi*phibar) )   (for the critical profile this should be real/positive)
        V1(r)  = 2 * V_s(s)
        V2(r)  = 4 * rho * V_ss(s)

        coeff = (gamma^2 - omega^2) + V1
        extra = 2 * V2 * rho

        yR'' = (coeff + extra) yR - 2 * omega * gamma * yI
        yI'' = coeff yI + 2 * omega * gamma * yR

    Returns:
      gamma_grid, Dp_vals, bracket, gamma_star,
      yR,yI (regular modes), and also (xi_y,xi_ybar) embedding.
    """

    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    phibar = np.asarray(phibar, dtype=float)
    assert r.ndim == 1 and phi.shape == r.shape and phibar.shape == r.shape
    assert np.all(np.diff(r) > 0), "r must be strictly increasing"
    r_min, r_max = float(r[0]), float(r[-1])
    if r_max <= 0:
        raise ValueError("r_max must be positive")

    # Background amplitude rho via s = Re(phi*phibar)
    s_bg = (phi * phibar).real
    # Clamp tiny negative numerical noise
    s_bg = np.maximum(s_bg, 0.0)
    rho_bg = np.sqrt(s_bg)

    V_s = potential.dV_ds(s_bg)
    V_ss = potential.d2V_ds2(s_bg)

    # Q-ball mapping for V(s) with s=rho^2
    V1_bg = 2.0 * V_s
    V2_bg = 4.0 * rho_bg * V_ss

    # Interpolants
    V1 = interp1d(r, V1_bg, kind=interp_kind, fill_value="extrapolate")
    V2 = interp1d(r, V2_bg, kind=interp_kind, fill_value="extrapolate")
    rho = interp1d(r, rho_bg, kind=interp_kind, fill_value="extrapolate")

    sgn = +1.0 if sign_convention >= 0 else -1.0

    def ode_rhs(rv: float, Y: np.ndarray, gamma: float) -> np.ndarray:
        # Y = [yR, yR', yI, yI']
        yR, dyR, yI, dyI = Y
        V1v = float(V1(rv))
        V2v = float(V2(rv))
        rhv = float(rho(rv))

        coeff = (gamma * gamma - omega * omega) + V1v
        extra = 2.0 * V2v * rhv

        # Same mixing structure as Q-ball; allow sign flip via sign_convention if needed
        yR2 = (coeff + extra) * yR - 2.0 * sgn * omega * gamma * yI
        yI2 = coeff * yI + 2.0 * sgn * omega * gamma * yR

        return np.array([dyR, yR2, dyI, yI2], dtype=float)

    # Integrate inward like Q-ball
    t_eval = r[::-1]

    # Neumann in underlying field => y'(rmax) = y(rmax)/rmax
    def solve_basis(gamma: float):
        # Basis A: yR=1, yI=0 at rmax
        y0_A = np.array([1.0, 1.0 / r_max, 0.0, 0.0], dtype=float)
        # Basis B: yR=0, yI=1 at rmax
        y0_B = np.array([0.0, 0.0, 1.0, 1.0 / r_max], dtype=float)

        kwargs: Dict[str, Any] = dict(method=ode_method, t_eval=t_eval, rtol=rtol, atol=atol)
        if max_step is not None:
            kwargs["max_step"] = max_step

        sol_A = solve_ivp(lambda rr, yy: ode_rhs(rr, yy, gamma), (r_max, r_min), y0_A, **kwargs)
        sol_B = solve_ivp(lambda rr, yy: ode_rhs(rr, yy, gamma), (r_max, r_min), y0_B, **kwargs)

        if not (sol_A.success and sol_B.success):
            raise RuntimeError(f"ODE solver failed for gamma={gamma}")

        # Reverse back to increasing r order
        yR_A = sol_A.y[0, ::-1]
        yI_A = sol_A.y[2, ::-1]
        yR_B = sol_B.y[0, ::-1]
        yI_B = sol_B.y[2, ::-1]
        return yR_A, yI_A, yR_B, yI_B

    def Dp_of_gamma(gamma: float) -> float:
        yR_A, yI_A, yR_B, yI_B = solve_basis(gamma)
        # enforce regularity at r=0: yR(0)=0 and yI(0)=0
        yR_A0, yI_A0 = float(yR_A[0]), float(yI_A[0])
        yR_B0, yI_B0 = float(yR_B[0]), float(yI_B[0])
        return yR_A0 * yI_B0 - yR_B0 * yI_A0

    gamma_grid = np.linspace(gamma_scan[0], gamma_scan[1], int(n_scan))
    Dp_vals = np.array([Dp_of_gamma(g) for g in gamma_grid], dtype=float)

    bracket = None
    for i in range(len(gamma_grid) - 1):
        if np.sign(Dp_vals[i]) != np.sign(Dp_vals[i + 1]):
            bracket = (float(gamma_grid[i]), float(gamma_grid[i + 1]))
            break
    if bracket is None:
        if require_bracket:
            raise RuntimeError(
                "Could not bracket negative mode eigenvalue; adjust gamma_scan/n_scan "
                "or check omega/potential/profile."
            )
        # Diagnostic return: no eigenvalue found, but return scan for plotting
        return {
            "gamma_grid": gamma_grid,
            "Dp_vals": Dp_vals,
            "bracket": None,
            "gamma_star": np.nan,
            "yR": np.zeros_like(r),
            "yI": np.zeros_like(r),
            "xi_y": np.zeros_like(r),
            "xi_ybar": np.zeros_like(r),
            "V1": V1_bg,
            "V2": V2_bg,
            "rho": rho_bg,
        }

    root = root_scalar(Dp_of_gamma, bracket=bracket, method="brentq", xtol=1e-10, rtol=1e-10)
    if not root.converged:
        raise RuntimeError("Root finding for gamma did not converge.")
    gamma_star = float(root.root)

    # Build the actual mode at gamma_star via the same linear combination trick
    yR_A, yI_A, yR_B, yI_B = solve_basis(gamma_star)
    yR_A0, yR_B0 = float(yR_A[0]), float(yR_B[0])

    c_A = yR_B0
    c_B = -yR_A0

    yR = c_A * yR_A + c_B * yR_B
    yI = c_A * yI_A + c_B * yI_B

    # Normalize in regular variables (matches Q-ball: ∫ dr (yR^2+yI^2))
    norm = np.sqrt(np.trapz(yR * yR + yI * yI, r))
    if norm > 0:
        yR /= norm
        yI /= norm

    # Embed into your (y, ybar) real arrays in the conjugate-like subspace:
    # delta_y   = yR + yI
    # delta_ybar= yR - yI
    # (so that delta_y and delta_ybar represent Re/Im as in Q-ball)
    xi_y = yR + yI
    xi_ybar = yR - yI

    return {
        "gamma_grid": gamma_grid,
        "Dp_vals": Dp_vals,
        "bracket": bracket,
        "gamma_star": gamma_star,
        "yR": yR,
        "yI": yI,
        "xi_y": xi_y,
        "xi_ybar": xi_ybar,
        "V1": V1_bg,
        "V2": V2_bg,
        "rho": rho_bg,
    }

from typing import Tuple

def compute_negative_mode_1d(
    *,
    r: np.ndarray,
    dr: float,                 # kept for API compatibility (not used by shooting)
    phi_false: float,          # kept for API compatibility (not used here)
    potential,                 # PotentialModel
    y1d: np.ndarray,           # y=r(phi-phi_false)
    ybar1d: np.ndarray,
    omega: float,              # REQUIRED for Q-ball-like negative mode
    gamma_scan: Tuple[float, float] = (0.01, 0.5),
    n_scan: int = 80,
    sign_convention: int = +1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Q-ball-faithful negative mode for the 1D critical bubble profile.

    Returns:
      xi_y(r), xi_ybar(r), lambda_min

    where lambda_min = -gamma_star^2 < 0.
    """
    r = np.asarray(r, dtype=float)
    y1d = np.asarray(y1d, dtype=float)
    ybar1d = np.asarray(ybar1d, dtype=float)

    # Reconstruct phi, phibar from y, ybar using constant shift
    phi = np.empty_like(y1d)
    phibar = np.empty_like(ybar1d)
    phi[0] = phi_false
    phibar[0] = phi_false
    phi[1:] = phi_false + y1d[1:] / r[1:]
    phibar[1:] = phi_false + ybar1d[1:] / r[1:]

    out = compute_Dp_shooting_1d(
        r=r,
        phi=phi,
        phibar=phibar,
        potential=potential,
        omega=float(omega),
        gamma_scan=gamma_scan,
        n_scan=n_scan,
        sign_convention=sign_convention,
    )

    gamma_star = float(out["gamma_star"])
    lam = -gamma_star * gamma_star

    xi_y = np.asarray(out["xi_y"], dtype=float)
    xi_ybar = np.asarray(out["xi_ybar"], dtype=float)

    # Sanity: regularity in y variables (like Q-ball)
    xi_y[0] = 0.0
    xi_ybar[0] = 0.0

    return xi_y, xi_ybar, float(lam)

# ---------------------------------------------------------------------
# Main ansatz builder (Q-ball-style): base + eps*negative_mode + projector
# ---------------------------------------------------------------------

def _estimate_rho_hom_from_tail(phi: np.ndarray, phibar: np.ndarray, m: int = 8) -> float:
    """Robust estimate of rho_hom ~ sqrt(Re(phi*phibar)) from the tail."""
    m = int(max(1, min(m, phi.size)))
    s_tail = np.mean((phi[-m:] * phibar[-m:]).real)
    return float(np.sqrt(max(s_tail, 0.0)))


def _tau_ramp_halfbox(tau: np.ndarray) -> np.ndarray:
    """
    Monotonic ramp h(τ) with h=0 at τ_min and h=1 at τ_max.
    For half-box τ∈[-β/2,0], τ_max=0. Smoothstep over the whole interval.
    """
    tmin = float(np.min(tau))
    tmax = float(np.max(tau))
    if abs(tmax - tmin) < 1e-14:
        return np.zeros_like(tau, float)
    u = (tau - tmin) / (tmax - tmin)  # in [0,1]
    return smoothstep(u)


def build_seed_bubble(
    *,
    grid,
    fields,
    potential,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    params: AnsatzParams,
    sign_convention: int = +1,
    cache: Optional[Dict[float, Tuple[np.ndarray, np.ndarray, float]]] = None,
    neg_mode_override: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
    rho_tilde_override: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Seed with a τ-dependent background at large r:
      τ=-β/2  -> rho = rho_ref  (homogeneous at ω_ref = rho_hom)
      τ=0     -> rho = rho_tilde = rho_hom(ω̃) = φ_bubble(r=r_max),
                and add localized critical-bubble profile on top.
    If rho_tilde_override is given, use it as rho_tilde (tail of 1D bubble); else estimate from profile tail.

    Minimal changes vs your current version:
      - NO radial_window on the base/background (so r=rmax can move with τ)
      - radial_window ONLY on the kick (negative mode)
      - background ramp term added explicitly in y,ybar (affine in r)
      - bubble profile embedded as (phi_cb - rho_tilde), so it decays at large r

    Everything else (negative mode, gating, projector, packing) stays the same.
    """
    r = np.asarray(grid.r, float)
    tau = np.asarray(grid.tau, float)
    T = float(grid.T)
    rmax = float(r[-1])

    # Twist parameter kept (outer loop can still use it); projector may or may not use it depending on tau_mode
    eta = T * (omega_tilde - omega_ref)

    # --- 1D critical bubble profile at omega_tilde (HOOK)
    phi1d, phibar1d = bubble_profile_1d(omega_tilde, r)
    phi1d = np.asarray(phi1d, float).copy()
    phibar1d = np.asarray(phibar1d, float).copy()
    assert phi1d.shape == r.shape and phibar1d.shape == r.shape

    # --- Homogeneous values: τ=-β/2 -> rho_ref, τ=0 -> rho_tilde = φ_bubble(r_max) = rho_hom(ω̃)
    rho_ref = float(fields.phi_false)  # rho_hom(omega_ref)
    rho_tilde = (
        float(rho_tilde_override)
        if rho_tilde_override is not None
        else _estimate_rho_hom_from_tail(phi1d, phibar1d, m=8)
    )

    # --- Build τ-dependent background ramp: rho_bg(τ) = rho_ref + h(τ)*(rho_tilde-rho_ref)
    h_bg = _tau_ramp_halfbox(tau)  # 0 at τ=-β/2, 1 at τ=0
    # In y = r(φ - rho_ref): pure background shift is y_bg = r*(rho_bg - rho_ref) = r*h*(rho_tilde-rho_ref)
    y_bg = np.outer(h_bg, r * (rho_tilde - rho_ref))
    ybar_bg = np.outer(h_bg, r * (rho_tilde - rho_ref))

    # --- Localized bubble part around rho_tilde: δφ_cb(r) = φ_cb(r) - rho_tilde
    # In y variables relative to rho_ref: y_loc = r*(δφ_cb)  (no large linear tail)
    y1d_loc = r * (phi1d - rho_tilde)
    ybar1d_loc = r * (phibar1d - rho_tilde)

    # Bubble gate: off near τ=-β/2, on near τ=0 (half-box). Keep your existing gate knob.
    g_tau = tau_gate(tau, T, params.tau_gate_frac)  # shape (Nt,)

    # Base = background ramp + gated localized bubble
    Y0 = y_bg + params.amp * np.outer(g_tau, y1d_loc)
    YB0 = ybar_bg + params.amp * np.outer(g_tau, ybar1d_loc)

    # --- Negative mode (Q-ball-like, 2-component)
    if neg_mode_override is not None:
        xi_y, xi_ybar, lam = neg_mode_override
        xi_y = np.asarray(xi_y, float).reshape(-1)
        xi_ybar = np.asarray(xi_ybar, float).reshape(-1)
        assert xi_y.shape == r.shape and xi_ybar.shape == r.shape
    else:
        cache_key = None
        if cache is not None:
            for k in cache:
                if abs(k - omega_tilde) < 1e-10:
                    cache_key = k
                    break
        if cache_key is not None:
            xi_y, xi_ybar, lam = cache[cache_key]
        else:
            # NOTE: for the negative mode you want the critical object itself.
            # Here we pass y(r)=r(φ_cb - rho_ref) as “background around which we fluctuate”.
            # That is: y1d_full = r*(phi1d - rho_ref).
            y1d_full = r * (phi1d - rho_ref)
            ybar1d_full = r * (phibar1d - rho_ref)
            xi_y, xi_ybar, lam = compute_negative_mode_1d(
                r=r, dr=float(grid.dr),
                phi_false=float(rho_ref),
                potential=potential,
                y1d=y1d_full, ybar1d=ybar1d_full,
                omega=float(omega_tilde),
                sign_convention=sign_convention,
            )
            if cache is not None:
                cache[omega_tilde] = (xi_y, xi_ybar, lam)

    # --- Kick (window ONLY here; do not kill the base at rmax)
    w_kick = radial_window(r, rmax, params.r_window_frac)   # 1 in bulk, 0 near rmax
    cos_tau = np.cos(2.0*np.pi*params.k * (tau / T) + params.phase)
    kick = np.outer(g_tau * cos_tau, w_kick)

    Y = Y0 + params.eps * kick * xi_y.reshape(1, -1)
    YB = YB0 + params.eps * kick * xi_ybar.reshape(1, -1)

    # --- Project BCs (discrete) in y,ybar. Use twisted only (no half-box).
    # Do NOT overwrite r=rmax after projection: leave Neumann extrapolation from the projector.
    Y, YB = bc_projector_y(Y, YB, fields=fields, eta=eta, sign_convention=sign_convention, tau_mode="twisted")

    x0 = fields.pack(Y, YB)
    meta = {
        "omega_ref": float(omega_ref),
        "omega_tilde": float(omega_tilde),
        "eta": float(eta),
        "rho_ref": float(rho_ref),
        "rho_tilde": float(rho_tilde),
        "params": params.__dict__.copy(),
        "neg_mode_lambda": float(lam),
        "note": (
            "Seed = tau-ramped homogeneous background (rho_ref->rho_tilde) "
            "+ gated localized bubble around rho_tilde + eps*kick(negmode). "
            "Base not windowed at rmax; window only on kick."
        ),
    }
    return x0, meta


def build_twisted_seed_from_static(
    *,
    solver,
    r_1d: np.ndarray,
    phi_1d: np.ndarray,
    rho0: float,
    g_tau: Optional[np.ndarray] = None,
    a: float = 0.0,
    seed_tau_bump: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Build a twisted-consistent seed from a static tau-independent 1D bubble.

    When a=0 and seed_tau_bump is None: reduces exactly to the static embedded bubble.
    Optional tau-shape g(tau) and amplitude a add tau dependence:
        y = y_static + a * g(tau) * bump_y
        ybar = ybar_static + a * g(tau) * bump_ybar

    seed_tau_bump: optional (bump_y, bump_ybar) each (Nr, Nt) in solver layout,
        localized near tau=0 and small r. If None and a!=0, no tau modulation.

    Returns packed x for the solver.
    """
    from scipy.interpolate import interp1d

    r_grid = np.asarray(solver.grid.r, dtype=float)
    r_1d = np.asarray(r_1d, dtype=float)
    phi_1d = np.asarray(phi_1d, dtype=float)
    order = np.argsort(r_1d)
    r_1d = r_1d[order]
    phi_1d = phi_1d[order]
    interp_fn = interp1d(
        r_1d, phi_1d, kind="cubic", bounds_error=False,
        fill_value=(phi_1d[0], phi_1d[-1]),
    )
    varphi_r = interp_fn(r_grid)
    Nr, Nt = solver.Nr, solver.Nt
    r = np.asarray(solver.grid.r, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    varphi_2d = np.broadcast_to(varphi_r[:, None], (Nr, Nt))
    y = r * (varphi_2d - rho0)
    ybar = y.copy()
    if r.size > 0 and float(r.flat[0]) < 1e-12 * (float(np.max(r)) + 1e-12):
        y[0, :] = 0.0
        ybar[0, :] = 0.0

    if a != 0.0 and seed_tau_bump is not None:
        bump_y, bump_ybar = seed_tau_bump
        g = g_tau if g_tau is not None else np.ones(Nt, dtype=float)
        if g.ndim == 0:
            g = np.full(Nt, float(g))
        g = np.asarray(g, dtype=float).reshape(-1)[:Nt]
        if len(g) < Nt:
            g = np.pad(g, (0, Nt - len(g)), constant_values=1.0)
        y = y + a * bump_y * g[None, :]
        ybar = ybar + a * bump_ybar * g[None, :]

    fields = Fields(solver)
    Y = y.T.copy()
    YB = ybar.T.copy()
    Y, YB = bc_projector_y(Y, YB, fields=fields, eta=0.0, tau_mode="twisted")
    return fields.pack(Y, YB)


# ---------------------------------------------------------------------
# Negative mode from 1D bounce (centralized for notebooks)
# ---------------------------------------------------------------------

@dataclass
class NegativeModeResult:
    """Result of compute_negative_mode_from_bounce: gamma, eigenmode, Dp scan."""
    gamma: float
    r: np.ndarray
    xi_y: np.ndarray
    xi_ybar: np.ndarray
    lam_neg: float
    beta_natural: float
    gamma_grid: np.ndarray
    Dp_vals: np.ndarray

    @property
    def xi(self) -> np.ndarray:
        """Alias for xi_y (for notebook compatibility)."""
        return self.xi_y


def prepare_negative_mode_region(
    r_bounce: np.ndarray,
    phi_bounce: np.ndarray,
    phi_false_bounce: float,
    U: Callable,
    dU: Callable,
    d2U: Callable,
    *,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Any, float, float]:
    """
    Build (r, phi_slice, potential_1d, dr, phi_false) for negative-mode or Dp diagnostic.
    Used when compute_negative_mode_from_bounce fails with "Could not bracket".
    """
    r_full = np.asarray(r_bounce, dtype=float)
    phi_full = np.asarray(phi_bounce, dtype=float)
    mask = r_full >= r_min
    if r_max is not None:
        mask = mask & (r_full <= r_max)
    r = r_full[mask]
    phi_slice = phi_full[mask]
    dr = float(r[1] - r[0]) if len(r) > 1 else 0.01
    phi_false = float(phi_false_bounce)
    potential_1d = PotentialModel(U, dU, d2U)
    return r, phi_slice, potential_1d, dr, phi_false


def compute_negative_mode_from_bounce(
    r_bounce: np.ndarray,
    phi_bounce: np.ndarray,
    phi_false_bounce: float,
    U: Callable,
    dU: Callable,
    d2U: Callable,
    omega: float,
    *,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    gamma_scan_neg: Tuple[float, float] = (0.01, 5.0),
    n_scan_neg: int = 80,
    gamma_scan_dp: Tuple[float, float] = (0.01, 10.0),
    n_scan_dp: int = 80,
    sign_convention: int = 1,
):
    """
    Compute the negative mode around the 1D critical bubble and optional Dp(γ) scan.

    Uses the same potential interface as the 2D solver (U, dU, d2U on ρ). Builds
    (r, phi_slice), y1d = r*(phi - phi_false), PotentialModel(U, dU, d2U), then
    compute_negative_mode_1d and compute_Dp_shooting_1d.

    Returns
    -------
    NegativeModeResult
        .gamma, .r, .xi_y, .xi_ybar, .lam_neg, .beta_natural, .gamma_grid, .Dp_vals

    Raises
    ------
    RuntimeError
        If fewer than 4 points in [r_min, r_max] or if compute_negative_mode_1d
        fails (e.g. "Could not bracket"). Caller can catch and run Dp diagnostic.
    """
    r_full = np.asarray(r_bounce, dtype=float)
    phi_full = np.asarray(phi_bounce, dtype=float)
    mask = r_full >= r_min
    if r_max is not None:
        mask = mask & (r_full <= r_max)
    r = r_full[mask]
    phi_slice = phi_full[mask]

    if len(r) < 4:
        raise RuntimeError(
            "Negative mode: need at least 4 points in range; widen r_min / r_max."
        )

    dr = float(r[1] - r[0]) if len(r) > 1 else 0.01
    phi_false = float(phi_false_bounce)
    y1d = np.zeros_like(r, float)
    ybar1d = np.zeros_like(r, float)
    y1d[0] = 0.0
    ybar1d[0] = 0.0
    y1d[1:] = r[1:] * (phi_slice[1:] - phi_false)
    ybar1d[1:] = r[1:] * (phi_slice[1:] - phi_false)

    potential_1d = PotentialModel(U, dU, d2U)
    xi_y, xi_ybar, lam_neg = compute_negative_mode_1d(
        r=r,
        dr=dr,
        phi_false=phi_false,
        potential=potential_1d,
        y1d=y1d,
        ybar1d=ybar1d,
        omega=float(omega),
        sign_convention=sign_convention,
        gamma_scan=gamma_scan_neg,
        n_scan=n_scan_neg,
    )

    gamma = np.sqrt(-lam_neg) if lam_neg < 0 else np.nan
    beta_natural = (2.0 * np.pi / gamma) if (gamma > 0 and np.isfinite(gamma)) else np.nan

    result_dp = compute_Dp_shooting_1d(
        r=r,
        phi=phi_slice,
        phibar=phi_slice,
        potential=potential_1d,
        omega=float(omega),
        gamma_scan=gamma_scan_dp,
        n_scan=n_scan_dp,
        sign_convention=sign_convention,
    )

    return NegativeModeResult(
        gamma=float(gamma),
        r=np.asarray(r, dtype=float),
        xi_y=np.asarray(xi_y, dtype=float),
        xi_ybar=np.asarray(xi_ybar, dtype=float),
        lam_neg=float(lam_neg),
        beta_natural=float(beta_natural),
        gamma_grid=result_dp["gamma_grid"],
        Dp_vals=result_dp["Dp_vals"],
    )


def plot_Dp_shooting_verification(
    unstable_mode: Any,
    *,
    show_plot: bool = True,
) -> None:
    """
    Plot D_p(γ) shooting curve and print verification (bracket, D_p at γ).
    Expects unstable_mode to have .gamma, .gamma_grid, .Dp_vals (e.g. NegativeModeResult).
    If gamma_grid/Dp_vals are missing, falls back to a simple gamma bar plot.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    if unstable_mode is None:
        print("Error: Negative mode not computed. Please run the previous cell first.")
        return

    gamma_grid = getattr(unstable_mode, "gamma_grid", None)
    Dp_vals = getattr(unstable_mode, "Dp_vals", None)
    gamma = getattr(unstable_mode, "gamma", None)

    if gamma_grid is not None and Dp_vals is not None and len(gamma_grid) > 0:
        bracket = None
        for i in range(len(gamma_grid) - 1):
            if np.sign(Dp_vals[i]) != np.sign(Dp_vals[i + 1]):
                bracket = (float(gamma_grid[i]), float(gamma_grid[i + 1]))
                break

        if show_plot:
            plt.figure(figsize=(7, 5))
            plt.plot(gamma_grid, np.log10(np.abs(Dp_vals) + 1e-300), "-o", lw=1.5, ms=4)
            plt.axhline(0, color="k", lw=0.8)
            if gamma is not None and np.isfinite(gamma):
                plt.axvline(gamma, color="r", ls="--", lw=1.5, label=rf"$\gamma_{{found}} = {gamma:.6f}$", alpha=0.7)
            if bracket is not None:
                plt.axvspan(bracket[0], bracket[1], color="orange", alpha=0.2, label="Sign change interval")
            plt.xlabel(r"$\gamma$")
            plt.ylabel(r"$\log_{10}|D_p|$")
            plt.title(r"Shooting function $D_p(\gamma)$")
            plt.xlim(float(np.min(gamma_grid)), float(np.max(gamma_grid)))
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.show()

        print("\nShooting function verification:")
        if gamma is not None and np.isfinite(gamma):
            print(f"  Found γ = {gamma:.6f}")
        if bracket is not None:
            print(f"  Sign change bracket: [{bracket[0]:.6f}, {bracket[1]:.6f}]")
            if gamma is not None and np.isfinite(gamma):
                print(f"  γ is within bracket: {bracket[0] <= gamma <= bracket[1]}")
        if len(gamma_grid) > 0 and gamma is not None and np.isfinite(gamma):
            Dp_interp = interp1d(gamma_grid, Dp_vals, kind="linear", bounds_error=False, fill_value=np.nan)
            Dp_at_gamma = float(Dp_interp(gamma))
            print(f"  D_p(γ_found) ≈ {Dp_at_gamma:.6e}")
            print(f"  |D_p(γ_found)| < 1e-3: {abs(Dp_at_gamma) < 1e-3}")
    else:
        if gamma is not None and np.isfinite(gamma) and show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.bar([0], [gamma], color="steelblue", edgecolor="k", width=0.5)
            ax.axhline(gamma, color="r", ls="--", lw=1.5, alpha=0.7, label=rf"$\gamma = {gamma:.6f}$")
            ax.set_ylabel(r"$\gamma = \sqrt{-\lambda}$")
            ax.set_title(rf"Negative mode: $\gamma$ = {gamma:.6f} (from eigenvalue $\lambda$)")
            ax.set_xticks([])
            ax.legend()
            ax.grid(True, axis="y", alpha=0.4)
            plt.tight_layout()
            plt.show()
            print(f"\nDiagnostic: gamma = sqrt(-lambda) = {gamma:.6f}")
        else:
            print("Shooting function not computed; gamma not available.")


def run_best_seed_selection(
    *,
    omega_ref: float,
    omega_tilde: float,
    r_bubble: Any,
    phi_bubble: Any,
    rho_hom: float,
    Q_target: float,
    unstable_mode: Any,
    param_grid: Any,
    U: Callable,
    dU: Callable,
    d2U: Callable,
    nr: int = 100,
    ntau: int = 60,
    wF: float = 1.0,
    wQ: float = 5.0,
    verbose: bool = True,
    rho_tilde: Optional[float] = None,
    beta: Optional[float] = None,
    beta_scale: Optional[float] = None,
    beta_scales: Optional[Iterable[float]] = None,
):
    """
    Run best-seed selection using build_seed_bubble. Uses pre-computed 1D bubble
    (r_bubble, phi_bubble). Single grid: Lr = max(3, r_bubble[-1]), beta from args or 2π/γ.
    Returns BestSeedSelectionResult.

    rho_tilde: ρ(ω̃) = rho_hom(omega_tilde) = φ_false(ω̃) (homogeneous at omega_tilde).
      Used as background at τ=0. If None, falls back to phi_bubble[-1] (tail at r_max).
      Pass phi_false_tilde from the charge-matched cell for correctness.

    beta: if set, use this as the temporal extent (τ ∈ [-β/2, β/2]). Overrides beta_scale and beta_scales.
    beta_scale: if beta is None and beta_scales is None, use beta_scale * (2π/γ) (default 1.0).
    beta_scales: if set (e.g. [1.0, 2.0, 5.0]), run the scan for each beta = s * beta_natural
      and return the result with the best score (allows scanning 1×, 2×, 5× natural beta).
    """
    r_b = np.asarray(r_bubble, dtype=float)
    phi_b = np.asarray(phi_bubble, dtype=float)
    Lr_sel = float(max(3.0, r_b[-1])) if r_b.size else 3.0
    gamma_neg = getattr(unstable_mode, "gamma", None) if unstable_mode is not None else None
    beta_natural = (
        float(2.0 * np.pi / gamma_neg)
        if (gamma_neg is not None and np.isfinite(gamma_neg) and gamma_neg > 0)
        else 20.0
    )
    bubble_profile_1d = make_bubble_profile_1d_from_arrays(r_b, phi_b)
    # Do not pass rho_tilde_override so build_seed_bubble uses _estimate_rho_hom_from_tail
    # identically to the single-ansatz example (bubble_2D.ipynb).

    def run_one_beta(beta_sel: float):
        settings = Bubble2DSettings(
            Nr=nr,
            Ntau=ntau,
            Lr=Lr_sel,
            beta=float(beta_sel),
            omega_ref=omega_ref,
            rho0=float(rho_hom),
            rho0_bracket=(1.0, 1.1),
            newton_tol=1e-8,
            newton_max_iter=35,
            damping=0.5,
            newton_verbose=False,
            tau_bc="twisted",
        )
        solver = Bubble2DSolver(settings, U, dU, d2U)
        solver.omega = float(omega_ref)
        grid2d, fields, potential, res_op, qcalc = make_q_ball_objects(solver)
        neg_mode_override = None
        if unstable_mode is not None and hasattr(unstable_mode, "xi_ybar") and hasattr(unstable_mode, "lam_neg"):
            r_neg = np.asarray(unstable_mode.r, dtype=float)
            r_grid = np.asarray(fields.grid.r, dtype=float)
            neg_mode_override = (
                np.interp(r_grid, r_neg, unstable_mode.xi),
                np.interp(r_grid, r_neg, unstable_mode.xi_ybar),
                unstable_mode.lam_neg,
            )
        x0, cand = select_best_seed(
            omega_ref=omega_ref,
            omega_grid=[float(omega_tilde)],
            param_grid=param_grid,
            Q_target=Q_target,
            fields=fields,
            potential=potential,
            res_op=res_op,
            qcalc=qcalc,
            bubble_profile_1d=bubble_profile_1d,
            sign_convention=1,
            wF=wF,
            wQ=wQ,
            verbose=verbose,
            neg_mode_override=neg_mode_override,
            rho_tilde_override=None,
        )
        return BestSeedSelectionResult(
            best_x0=x0,
            best_candidate=cand,
            best_grid=grid2d,
            best_beta=float(beta_sel),
            effective_Q_target=Q_target,
        )

    if beta_scales is not None:
        scales = list(beta_scales)
        if verbose:
            print(f"[SeedScan] Scanning beta = scale * beta_natural (beta_natural = {beta_natural:.4f}) for scales {scales}")
        best_result = None
        best_score = None
        for s in scales:
            beta_sel = float(s * beta_natural)
            if verbose:
                print(f"[SeedScan] --- beta = {s} * beta_natural = {beta_sel:.4f} ---")
            result = run_one_beta(beta_sel)
            sc = result.best_candidate.score
            if best_score is None or sc < best_score:
                best_score = sc
                best_result = result
                if verbose:
                    print(f"[SeedScan] New best score {best_score:.3e} at beta = {beta_sel:.4f}")
        if verbose and best_result is not None:
            print(f"[SeedScan] Best overall: beta = {best_result.best_beta:.4f}, score = {best_result.best_candidate.score:.3e}")
        return best_result

    if beta is not None:
        beta_sel = float(beta)
    elif beta_scale is not None:
        beta_sel = float(beta_scale * beta_natural)
    else:
        beta_sel = beta_natural
    return run_one_beta(beta_sel)


# ---------------------------------------------------------------------
# Best-seed selection (moved from best_selection_seed.py)
# ---------------------------------------------------------------------

@dataclass
class SeedCandidate:
    omega_tilde: float
    params: AnsatzParams
    normF: float
    Q: float
    Q_excess: float
    score: float
    meta: Dict[str, Any]


def _call_residual(res_op, x: np.ndarray, omega_bulk: float, *, eta: Optional[float] = None) -> np.ndarray:
    """Call residual with (x, omega) or (x, omega, eta) depending on solver."""
    if eta is None:
        return res_op.residual(x, omega_bulk)
    try:
        return res_op.residual(x, omega_bulk, eta)
    except TypeError:
        try:
            return res_op.residual(x, omega_bulk, eta=eta)
        except TypeError:
            return res_op.residual(x, omega_bulk)


def _compute_Q(fields, qcalc, x: np.ndarray, omega_for_Q: float) -> float:
    y, ybar = fields.unpack(x)
    phi, phibar = fields.reconstruct(y, ybar)
    return float(qcalc.compute(phi=phi, phibar=phibar, omega=omega_for_Q, grid=fields.grid))


def score_candidate(
    *,
    x0: np.ndarray,
    omega_ref: float,
    omega_tilde: float,
    Q_target: float,
    fields,
    res_op,
    qcalc,
    wF: float = 1.0,
    wQ: float = 10.0,
) -> Tuple[float, float, float]:
    """Score = wF*||F|| + wQ*|Q-Q_target|."""
    eta = fields.grid.T * (omega_tilde - omega_ref)
    F = _call_residual(res_op, x0, omega_bulk=omega_ref, eta=eta)
    normF = float(np.linalg.norm(F))
    Q = _compute_Q(fields, qcalc, x0, omega_for_Q=omega_tilde)
    Q_excess = Q - Q_target
    score = wF * normF + wQ * abs(Q_excess)
    return normF, Q, score


def select_best_seed(
    *,
    omega_ref: float,
    omega_grid: Iterable[float],
    param_grid: Iterable[AnsatzParams],
    Q_target: float,
    fields,
    potential,
    res_op,
    qcalc,
    bubble_profile_1d: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    sign_convention: int = +1,
    wF: float = 1.0,
    wQ: float = 10.0,
    verbose: bool = True,
    neg_mode_override: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
    rho_tilde_override: Optional[float] = None,
) -> Tuple[np.ndarray, SeedCandidate]:
    """
    Scan (omega_tilde, AnsatzParams) -> build seed (Q-ball-style) -> score -> pick best.
    If rho_tilde_override is set, use it as rho at τ=0 (φ_bubble(r_max)) for the background ramp.
    """
    neg_mode_cache: Dict[float, Tuple[np.ndarray, np.ndarray, float]] = {}
    best: Optional[SeedCandidate] = None
    best_x: Optional[np.ndarray] = None

    for omega_tilde in omega_grid:
        for params in param_grid:
            x0, meta = build_seed_bubble(
                grid=fields.grid,
                fields=fields,
                potential=potential,
                omega_ref=omega_ref,
                omega_tilde=float(omega_tilde),
                bubble_profile_1d=bubble_profile_1d,
                params=params,
                sign_convention=sign_convention,
                cache=neg_mode_cache,
                neg_mode_override=neg_mode_override,
                rho_tilde_override=rho_tilde_override,
            )
            normF, Q, score = score_candidate(
                x0=x0,
                omega_ref=omega_ref,
                omega_tilde=float(omega_tilde),
                Q_target=Q_target,
                fields=fields,
                res_op=res_op,
                qcalc=qcalc,
                wF=wF,
                wQ=wQ,
            )
            Q_excess = Q - Q_target
            cand = SeedCandidate(
                omega_tilde=float(omega_tilde),
                params=params,
                normF=normF,
                Q=Q,
                Q_excess=Q_excess,
                score=score,
                meta=meta,
            )
            if verbose:
                print(
                    f"[SeedScan] ω~={cand.omega_tilde:.6g} "
                    f"eps={params.eps:.3g} k={params.k} phase={params.phase:.3g} "
                    f"||F||={cand.normF:.3e} Q={cand.Q:.6e} score={cand.score:.3e}"
                )
            if best is None or cand.score < best.score:
                best = cand
                best_x = x0

    if best is None or best_x is None:
        raise RuntimeError("Seed scan produced no candidates (empty grids?)")
    return best_x, best


@dataclass
class BestSeedSelectionResult:
    """Result of best-seed selection (best_x0, best_candidate, best_grid, best_beta, effective_Q_target)."""

    best_x0: np.ndarray
    best_candidate: SeedCandidate
    best_grid: Any
    best_beta: float
    effective_Q_target: Optional[float] = None

    def to_best_seed(self) -> Any:
        """Return a namespace object compatible with downstream cells (best_seed.best_x0, .best_score, etc.)."""
        o = type("_BestSeed", (), {})()
        o.best_x0 = self.best_x0
        o.best_score = self.best_candidate.score
        o.best_norm = self.best_candidate.normF
        o.best_params = {**vars(self.best_candidate.params), "omega_tilde": self.best_candidate.omega_tilde}
        o.best_Q = self.best_candidate.Q
        o.best_Q_excess = self.best_candidate.Q_excess
        o.best_meta = self.best_candidate.meta
        o.table = []
        return o
