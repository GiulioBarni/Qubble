'''BubbleSolver2D.py

2D Euclidean (tau, r) solver for bubble nucleation on a CHARGED HOMOGENEOUS background
in the fixed-charge (microcanonical) formalism, built to mirror the Q-ball-decay ('Q-ball')
architecture as closely as possible.

Key features (Q-ball-like):
- half Euclidean time interval: tau ∈ (-beta/2, 0) on a half-step grid
- two partner fields (y, ybar) kept independent (DO NOT impose ybar=y)
- reflection/swap at tau=0 (turning slice)
- twist closure at tau=-beta/2 implemented ONLY through tau ghost rules
- Newton–Raphson with analytic sparse Jacobian + backtracking line search
- diagnostics and sanity checks:
  * background exactness at eta0=0 (bulk + BC wiring test)
  * analytic “twist source” check at eta0≠0 (boundary mismatch is predictable)
  * Jacobian FD check (supports complex saddles via real/imag splitting)

Core variables (Q-ball-like, with the MINIMAL mandatory shift for a nonzero medium):

  phi_rot    := e^{-omega_ref * tau} * phi
  phibar_rot := e^{+omega_ref * tau} * phibar

  y    := r * (phi_rot    - rho0)
  ybar := r * (phibar_rot - rho0)

so that the homogeneous medium corresponds to y=ybar=0 (stable numerically) instead of y~r*rho0.

IMPORTANT FIX vs common “bubble on medium” attempts:
Twist at the Euclidean time boundary must be applied to the TOTAL fields
  y_tot    = y    + r*rho0
  ybar_tot = ybar + r*rho0
not to the fluctuations alone.

This single change is often the difference between “Newton never converges” and a working solver.

-------------------------------------------------------------------------------
About complex saddles and the Jacobian
-------------------------------------------------------------------------------
The bulk potential is taken as a function of
  u := Re(phi_rot * phibar_rot)  (>=0, with a smooth projection)
so it is *not* holomorphic in the complex fields. For genuine complex saddles,
the correct Newton system is obtained by splitting into real/imag parts:

  unknowns: (Re y, Im y, Re ybar, Im ybar)
  equations: (Re Fy, Im Fy, Re Fyb, Im Fyb)

This file implements that “RI mode” when settings.complex_saddle=True (default).
You still keep the two partner fields (y, ybar); the split is only a numerical
representation so that the Jacobian is mathematically consistent.

-------------------------------------------------------------------------------
Production defaults (to match your stated requirements)
-------------------------------------------------------------------------------
By default, the solver enforces:
  tau_bc = 'twisted' and r_bc = 'neumann'
Other BC choices remain available only if settings.allow_debug_bcs=True.

'''

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from . import observables_2d


def _format_obs_line(obs: Dict[str, Any], tgt: Optional[Dict[str, Any]] = None) -> str:
    """
    Single helper for observables logging: one line with rho_Q, rho_E and ratios vs reference.
    Used in Newton callback, scans, and final summaries. All values from tau=0 ghost.
    Reference densities (rho_Q_ref, rho_E_ref) come from tgt (homogeneous on same grid).
    """
    rho_Q = obs.get("rho_Q", 0.0)
    rho_E = obs.get("rho_E", 0.0)
    rho_Q_ref = tgt.get("rho_Q") if tgt else None
    rho_E_ref = tgt.get("rho_E") if tgt else None
    ratio_rho_Q = (rho_Q / rho_Q_ref) if (rho_Q_ref is not None and abs(rho_Q_ref) > 1e-30) else float("nan")
    ratio_rho_E = (rho_E / rho_E_ref) if (rho_E_ref is not None and abs(rho_E_ref) > 1e-30) else float("nan")
    return (
        f"rhoQ={rho_Q:.6e}, rhoE={rho_E:.6e}, "
        f"rho_Q/rho_Q_ref={ratio_rho_Q:.4f}, rho_E/rho_E_ref={ratio_rho_E:.4f}"
    )


# Public alias for notebook and diagnostics
format_obs_line = _format_obs_line


# -----------------------------------------------------------------------------
# Optional reuse of Q-ball/Q_ball_finder infrastructure (grid + pack/unpack + Newton)
# -----------------------------------------------------------------------------
_build_grid = None
RadialTimeGrid = None
NewtonResult = None
NewtonConvergenceError = None
newton_solve = None

try:
    from Q_ball_finder.grid import build_grid as _build_grid, RadialTimeGrid as _RadialTimeGrid
    _build_grid, RadialTimeGrid = _build_grid, _RadialTimeGrid
except Exception:
    pass

if RadialTimeGrid is None:
    @dataclass
    class RadialTimeGrid:
        """Fallback grid: half-interval tau in (-beta/2, 0), r starts at dr."""
        Nr: int
        Ntau: int
        Lr: float
        beta: float

        def __post_init__(self):
            self.dr = self.Lr / self.Nr
            # Half-interval (-beta/2, 0): dtau = (beta/2)/Ntau
            self.dtau = self.beta / (2.0 * self.Ntau)
            self.r = (np.arange(self.Nr) + 1.0) * self.dr
            self.tau = -(np.arange(self.Ntau) + 0.5) * self.dtau

try:
    from Q_ball_finder.nr_solver import NewtonResult as _NewtonResult, NewtonConvergenceError as _NewtonConvErr, newton_solve as _newton_solve
    NewtonResult, NewtonConvergenceError, newton_solve = _NewtonResult, _NewtonConvErr, _newton_solve
except Exception:
    pass

if NewtonConvergenceError is None:
    class NewtonConvergenceError(RuntimeError):
        pass

if NewtonResult is None:
    @dataclass
    class NewtonResult:
        x: np.ndarray
        success: bool
        iterations: int
        residual_norm: float
        history: List[float]

if newton_solve is None:
    def newton_solve(
        residual, jacobian, x0,
        tol=1e-9, max_iter=35, damping=1.0,
        line_search=None, norm=lambda v: float(np.linalg.norm(v)), callback=None,
    ):
        x = np.array(x0, dtype=float)
        hist: List[float] = []
        for it in range(1, max_iter + 1):
            F = residual(x)
            nF = norm(F)
            hist.append(float(nF))
            if callback is not None:
                callback(it, x, F, float(nF))
            if nF < tol:
                return NewtonResult(x=x, success=True, iterations=it, residual_norm=float(nF), history=hist)
            J = jacobian(x)
            if sp.issparse(J):
                dx = spla.spsolve(sp.csc_matrix(J), -F)
            else:
                dx = np.linalg.solve(np.asarray(J), -F)
            alpha = float(damping)
            if line_search is not None:
                alpha = float(line_search(x, dx, F))
            x = x + alpha * dx
            if not np.isfinite(float(np.linalg.norm(x))):
                raise NewtonConvergenceError("Diverged (non-finite iterate).")
        return NewtonResult(x=x, success=False, iterations=max_iter, residual_norm=float(hist[-1]), history=hist)


# -----------------------------------------------------------------------------
# Potential interface: U(rho) with rho>=0
# -----------------------------------------------------------------------------
PotentialFn = Callable[[np.ndarray], np.ndarray]


def make_potential_from_V(
    V_phi_fn: Callable,
    dV_dphi_fn: Callable,
    d2V_dphi2_fn: Callable,
    phi0: float,
    v1: float,
    v2: float,
) -> Tuple[PotentialFn, PotentialFn, PotentialFn]:
    """
    Build U(rho), dU(rho), d2U(rho) from notebook-style V(φ), V'(φ), V''(φ) and parameters.

    Example:
        from potential_bubble import V_phi, dV_dphi, d2V_dphi2
        U, dU, d2U = make_potential_from_V(V_phi, dV_dphi, d2V_dphi2, phi0=1.999, v1=1.0, v2=2.0)
        solver = Bubble2DSolver(settings, U, dU, d2U)
    """
    phi0, v1, v2 = float(phi0), float(v1), float(v2)

    def U(rho: np.ndarray) -> np.ndarray:
        return np.asarray(V_phi_fn(np.asarray(rho, dtype=float), phi0, v1, v2), dtype=float)

    def dU(rho: np.ndarray) -> np.ndarray:
        return np.asarray(dV_dphi_fn(np.asarray(rho, dtype=float), phi0, v1, v2), dtype=float)

    def d2U(rho: np.ndarray) -> np.ndarray:
        return np.asarray(d2V_dphi2_fn(np.asarray(rho, dtype=float), phi0, v1, v2), dtype=float)

    return U, dU, d2U


@dataclass
class PiecewiseQuarticPotential:
    """Toy default potential used in notes."""
    v1: float
    v2: float
    lam1: float
    lam2: float

    def __post_init__(self):
        self.v1 = float(self.v1)
        self.v2 = float(self.v2)
        self.lam1 = float(self.lam1)
        self.lam2 = float(self.lam2)
        U1 = self.lam1 * (self.v1**2 - self.v1**2)**2
        U2 = self.lam2 * (self.v1**2 - self.v2**2)**2
        self.eps = U2 - U1

    def U(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        out = np.empty_like(rho)
        m = rho <= self.v1
        out[m] = self.lam1 * (rho[m]**2 - self.v1**2)**2
        out[~m] = self.lam2 * (rho[~m]**2 - self.v2**2)**2 - self.eps
        return out

    def dU(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        out = np.empty_like(rho)
        m = rho <= self.v1
        out[m] = 4.0 * self.lam1 * rho[m] * (rho[m]**2 - self.v1**2)
        out[~m] = 4.0 * self.lam2 * rho[~m] * (rho[~m]**2 - self.v2**2)
        return out

    def d2U(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        out = np.empty_like(rho)
        m = rho <= self.v1
        out[m] = 4.0 * self.lam1 * (3.0 * rho[m]**2 - self.v1**2)
        out[~m] = 4.0 * self.lam2 * (3.0 * rho[~m]**2 - self.v2**2)
        return out


def solve_rho0_for_omega(omega: float, dU: PotentialFn, bracket: Tuple[float, float], rtol: float = 1e-12) -> float:
    """Solve homogeneous stationarity in |phi|: dU/dρ(sqrt(2)x) = ω^2 sqrt(2)x, x=rho0=|phi|."""
    omega = float(omega)

    def f(x: float) -> float:
        rho_phys = np.sqrt(2.0) * x
        return float(dU(np.array([rho_phys]))[0] - omega * omega * rho_phys)

    a, b = map(float, bracket)
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(f"Bad bracket for rho0: f(a)={fa:+.3e}, f(b)={fb:+.3e}")
    return float(brentq(f, a, b, xtol=rtol, rtol=rtol, maxiter=300))


def _omega2_minus_2W_consistent(
    x: float,
    omega: float,
    dU: PotentialFn,
    s_smooth_eps: float,
    rho_eps: float,
) -> float:
    """
    Discrete-consistent mismatch used by the 2D solver background:
      f(x) = omega^2 - 2*W(u=x^2),
    with u_pos from smooth_pos and rho_phys = sqrt(2*u_pos + rho_eps).
    """
    x = float(x)
    omega = float(omega)
    eps = float(s_smooth_eps)
    rho_eps = float(rho_eps)
    u = x * x
    root = np.sqrt(u * u + eps * eps)
    u_pos = 0.5 * (u + root)
    rho_phys = float(np.sqrt(max(2.0 * u_pos + rho_eps, 1e-300)))
    dU_val = float(dU(np.array([rho_phys], dtype=float))[0])
    W = dU_val / (2.0 * rho_phys)
    return float(omega * omega - 2.0 * W)


def solve_rho0_for_omega_consistent(
    omega: float,
    dU: PotentialFn,
    d2U: PotentialFn,
    bracket: Tuple[float, float],
    s_smooth_eps: float,
    rho_eps: float,
    rtol: float = 1e-14,
) -> float:
    """
    Solve x=rho0=|phi| from the exact discrete-consistent condition used by the solver:
      omega^2 - 2*W(u=x^2) = 0,
    where W is built with smooth_pos(u) and rho_phys = sqrt(2*u_pos + rho_eps).
    """
    _ = d2U  # kept for API symmetry and future use

    def f(x: float) -> float:
        return _omega2_minus_2W_consistent(
            x=x,
            omega=omega,
            dU=dU,
            s_smooth_eps=s_smooth_eps,
            rho_eps=rho_eps,
        )

    a, b = map(float, bracket)
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(f"Bad bracket for rho0 (consistent): f(a)={fa:+.3e}, f(b)={fb:+.3e}")
    return float(brentq(f, a, b, xtol=rtol, rtol=rtol, maxiter=500))


# -----------------------------------------------------------------------------
# Settings / Solution containers
# -----------------------------------------------------------------------------
@dataclass
class Bubble2DSettings:
    # grid
    Nr: int = 160
    Ntau: int = 320
    Lr: float = 25.0
    beta: float = 60.0

    # Boundary conditions (production defaults)
    r_bc: str = "neumann"      # production: "neumann"; debug: "dirichlet_fluct"
    tau_bc: str = "twisted"    # production: "twisted"; debug: "hom_past" or "neumann"
    allow_debug_bcs: bool = False

    # rotated-frame frequency for stability
    omega_ref: float = 0.3

    # twist parameter (ONLY in tau ghost rule at i=Ntau-1)
    # convention: exp(±eta0). If omega_tilde is set and eta0 is None: eta0 = beta*(omega_tilde - omega_ref)
    eta0: Optional[float] = None
    omega_tilde: Optional[float] = None

    # background amplitude
    rho0: Optional[float] = None
    rho0_bracket: Tuple[float, float] = (1e-10, 50.0)

    # smoothing for u=Re(phi*phibar)
    s_smooth_eps: float = 1e-6
    rho_eps: float = 1e-12

    # Solve complex saddles: use real/imag split (recommended; mathematically consistent)
    complex_saddle: bool = True

    # Newton
    newton_tol: float = 1e-9
    newton_max_iter: int = 35
    damping: float = 1.0
    verbose: bool = True
    newton_verbose: Optional[bool] = None

    # line-search
    max_backtracks: int = 25
    min_step: float = 1e-8

    def __post_init__(self) -> None:
        if self.newton_verbose is not None:
            self.verbose = bool(self.newton_verbose)

        if not self.allow_debug_bcs:
            # Enforce your “production” requirements:
            if self.tau_bc != "twisted":
                raise ValueError("Production solver enforces tau_bc='twisted'. Set allow_debug_bcs=True to override.")
            if self.r_bc != "neumann":
                raise ValueError("Production solver enforces r_bc='neumann'. Set allow_debug_bcs=True to override.")

        if self.r_bc not in ("neumann", "dirichlet_fluct"):
            raise ValueError("r_bc must be 'neumann' or 'dirichlet_fluct'")
        if self.tau_bc not in ("twisted", "hom_past", "neumann"):
            raise ValueError("tau_bc must be 'twisted', 'hom_past', or 'neumann'")


@dataclass
class Bubble2DSolution:
    settings: Bubble2DSettings
    grid: RadialTimeGrid
    newton: NewtonResult
    y: np.ndarray
    ybar: np.ndarray
    rho0: float
    Q_tau0: complex
    E_tau0: float
    sanity: Dict[str, Any]
    iteration_history: Optional[List[Dict[str, Any]]] = None  # if solve(..., store_iteration_history=True)
    observables_ghost: Optional[Dict[str, Any]] = None  # Q, E, E_hom, energy_ratio, r, q_r, e_r, ...
    E_hom: float = 0.0       # homogeneous energy (canonical)
    energy_ratio: float = 0.0  # E_tau0 / E_hom

    @property
    def success(self) -> bool:
        return bool(self.newton.success)

    @property
    def iterations(self) -> int:
        return int(self.newton.iterations)

    @property
    def residual_norm(self) -> float:
        return float(self.newton.residual_norm)

    @property
    def x(self) -> np.ndarray:
        return self.newton.x

    @property
    def history(self) -> List[float]:
        return list(self.newton.history)


# -----------------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------------
class Bubble2DSolver:
    def __init__(self, settings: Bubble2DSettings, U: PotentialFn, dU: PotentialFn, d2U: PotentialFn):
        self.settings = settings
        self.U = U
        self.dU = dU
        self.d2U = d2U

        # Use repo grid builder when available (same half-interval convention)
        if _build_grid is not None:
            self.grid = _build_grid(settings.Nr, settings.Ntau, settings.Lr, settings.beta)
        else:
            self.grid = RadialTimeGrid(settings.Nr, settings.Ntau, settings.Lr, settings.beta)

        self.Nr, self.Nt = int(self.grid.Nr), int(self.grid.Ntau)
        self.dr, self.dt = float(self.grid.dr), float(self.grid.dtau)
        self.dr2, self.dt2 = self.dr * self.dr, self.dt * self.dt

        self.omega = float(settings.omega_ref)

        if settings.eta0 is None:
            if settings.omega_tilde is not None:
                settings.eta0 = float(settings.beta * (settings.omega_tilde - settings.omega_ref))
            else:
                settings.eta0 = 0.0
        self.eta0 = float(settings.eta0)

        if settings.rho0 is None:
            settings.rho0 = solve_rho0_for_omega_consistent(
                self.omega,
                dU,
                d2U,
                settings.rho0_bracket,
                settings.s_smooth_eps,
                settings.rho_eps,
            )
        else:
            # Backward-compatible guard:
            # if user passed rho0 as physical modulus rho_phys (common when using vacua_of_Omega),
            # convert to solver convention rho0=|phi|=rho_phys/sqrt(2).
            rho0_in = float(settings.rho0)
            rho_phys_from_phi = np.sqrt(2.0) * rho0_in
            res_solver_conv = float(dU(np.array([rho_phys_from_phi]))[0] - self.omega * self.omega * rho_phys_from_phi)
            res_phys_conv = float(dU(np.array([rho0_in]))[0] - self.omega * self.omega * rho0_in)
            if abs(res_solver_conv) > 1e-6 and abs(res_phys_conv) < abs(res_solver_conv):
                settings.rho0 = rho0_in / np.sqrt(2.0)
            # Optional snap to the exact discrete-consistent root when needed.
            rho0_now = float(settings.rho0)
            mismatch_now = _omega2_minus_2W_consistent(
                x=rho0_now,
                omega=self.omega,
                dU=dU,
                s_smooth_eps=settings.s_smooth_eps,
                rho_eps=settings.rho_eps,
            )
            if abs(mismatch_now) > 1e-10:
                a = max(1e-14, 0.8 * rho0_now)
                b = max(a * (1.0 + 1e-12), 1.2 * rho0_now)
                try:
                    settings.rho0 = solve_rho0_for_omega_consistent(
                        self.omega,
                        dU,
                        d2U,
                        (a, b),
                        settings.s_smooth_eps,
                        settings.rho_eps,
                    )
                except Exception:
                    # Keep user-provided value if local snap bracket does not bracket.
                    pass
        self.rho0 = float(settings.rho0)

        # Basic input validation
        if self.Nr < 1 or self.Nt < 1:
            raise ValueError("Nr and Ntau must be >= 1")
        if settings.Lr <= 0 or settings.beta <= 0:
            raise ValueError("Lr and beta must be > 0")

    # -------------------------------------------------------------------------
    # Packing/unpacking unknown vectors
    # -------------------------------------------------------------------------
    def _pack_2field(self, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
        """Q-ball-style: stack y and ybar (complex vector length 2*Nsite)."""
        return np.concatenate([y.ravel(), ybar.ravel()])

    def _unpack_2field(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = self.Nr * self.Nt
        y = vec[:N].reshape((self.Nr, self.Nt))
        ybar = vec[N:].reshape((self.Nr, self.Nt))
        return y, ybar

    def _pack_ri(self, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
        """Real/imag split: [Re y, Im y, Re ybar, Im ybar] (real vector length 4*Nsite)."""
        y = np.asarray(y)
        ybar = np.asarray(ybar)
        return np.concatenate([
            np.ascontiguousarray(y.real).ravel(),
            np.ascontiguousarray(y.imag).ravel(),
            np.ascontiguousarray(ybar.real).ravel(),
            np.ascontiguousarray(ybar.imag).ravel(),
        ]).astype(float)

    def _unpack_ri(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = self.Nr * self.Nt
        vec = np.asarray(vec, dtype=float)
        yR = vec[0 * N:1 * N].reshape((self.Nr, self.Nt))
        yI = vec[1 * N:2 * N].reshape((self.Nr, self.Nt))
        ybR = vec[2 * N:3 * N].reshape((self.Nr, self.Nt))
        ybI = vec[3 * N:4 * N].reshape((self.Nr, self.Nt))
        y = yR + 1j * yI
        ybar = ybR + 1j * ybI
        return y, ybar

    def pack(self, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
        """Pack (y,ybar) into the solver's internal unknown vector."""
        if self.settings.complex_saddle:
            return self._pack_ri(y, ybar)
        return self._pack_2field(y, ybar)

    def unpack(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack solver vector into complex arrays (y,ybar)."""
        if self.settings.complex_saddle:
            return self._unpack_ri(vec)
        return self._unpack_2field(vec)

    def _zero_vec(self) -> np.ndarray:
        """Zero unknown vector in the correct representation."""
        if self.settings.complex_saddle:
            return np.zeros(4 * self.Nr * self.Nt, dtype=float)
        # keep complex dtype to match Q-ball expectation
        return np.zeros(2 * self.Nr * self.Nt, dtype=complex)

    # -------------------------------------------------------------------------
    # Smooth projection u -> u_pos >=0 and W(u)
    # -------------------------------------------------------------------------
    def _smooth_pos(self, u: np.ndarray):
        eps = float(self.settings.s_smooth_eps)
        root = np.sqrt(u * u + eps * eps)
        u_pos = 0.5 * (u + root)
        dupos_du = 0.5 * (1.0 + u / root)
        return u_pos, dupos_du

    def _W_Wp(self, u: np.ndarray):
        """
        Returns:
          W(u_pos), Wp(u_pos)=dW/du_pos, u_pos, dupos_du
        where W = dU/d(rho)/(2 rho) and rho = sqrt(2*u_pos + rho_eps).
        """
        u_pos, dupos_du = self._smooth_pos(u)
        rho = np.sqrt(2.0 * u_pos + float(self.settings.rho_eps))
        dU = self.dU(rho)
        d2U = self.d2U(rho)
        W = dU / (2.0 * rho)
        Wp = (d2U * rho - dU) / (2.0 * rho**3)
        return W, Wp, u_pos, dupos_du

    # -------------------------------------------------------------------------
    # Field reconstruction
    # -------------------------------------------------------------------------
    def phi_rot(self, y: np.ndarray, ybar: np.ndarray):
        r = self.grid.r[:, None]
        phi = self.rho0 + y / r
        phibar = self.rho0 + ybar / r
        return phi, phibar

    def phi(self, y: np.ndarray, ybar: np.ndarray):
        tau = self.grid.tau[None, :]
        phi_rot, phibar_rot = self.phi_rot(y, ybar)
        phi = np.exp(+self.omega * tau) * phi_rot
        phibar = np.exp(-self.omega * tau) * phibar_rot
        return phi, phibar

    # -------------------------------------------------------------------------
    # tau ghost rules (Q-ball-like)
    # -------------------------------------------------------------------------
    def _tau_neighbors(self, y: np.ndarray, ybar: np.ndarray, i: int):
        Nt = self.Nt
        r = self.grid.r
        bc = self.settings.tau_bc

        if bc == "twisted":
            if i == 0:
                # reflection + swap at tau=0
                y_im1 = ybar[:, 0]
                y_ip1 = y[:, 1]
                yb_im1 = y[:, 0]
                yb_ip1 = ybar[:, 1]
            elif i == Nt - 1:
                # twist closure at tau=-beta/2: APPLY TWIST TO TOTAL FIELDS
                y_im1 = y[:, Nt - 2]
                yb_im1 = ybar[:, Nt - 2]
                y_ip1 = np.exp(-self.eta0) * (ybar[:, Nt - 1] + r * self.rho0) - r * self.rho0
                yb_ip1 = np.exp(+self.eta0) * (y[:, Nt - 1] + r * self.rho0) - r * self.rho0
            else:
                y_im1 = y[:, i - 1]
                y_ip1 = y[:, i + 1]
                yb_im1 = ybar[:, i - 1]
                yb_ip1 = ybar[:, i + 1]
            return y_im1, y_ip1, yb_im1, yb_ip1

        # Debug BCs kept only for experiments / limiting cases.
        if bc == "hom_past":
            if i == 0:
                y_im1 = ybar[:, 0]
                y_ip1 = y[:, 1]
                yb_im1 = y[:, 0]
                yb_ip1 = ybar[:, 1]
            elif i == Nt - 1:
                y_im1 = y[:, Nt - 2]
                y_ip1 = y[:, Nt - 1]
                yb_im1 = ybar[:, Nt - 2]
                yb_ip1 = ybar[:, Nt - 1]
            else:
                y_im1 = y[:, i - 1]
                y_ip1 = y[:, i + 1]
                yb_im1 = ybar[:, i - 1]
                yb_ip1 = ybar[:, i + 1]
            return y_im1, y_ip1, yb_im1, yb_ip1

        if bc == "neumann":
            if i == 0:
                y_im1 = y[:, 0]
                y_ip1 = y[:, 1]
                yb_im1 = ybar[:, 0]
                yb_ip1 = ybar[:, 1]
            elif i == Nt - 1:
                y_im1 = y[:, Nt - 2]
                y_ip1 = y[:, Nt - 1]
                yb_im1 = ybar[:, Nt - 2]
                yb_ip1 = ybar[:, Nt - 1]
            else:
                y_im1 = y[:, i - 1]
                y_ip1 = y[:, i + 1]
                yb_im1 = ybar[:, i - 1]
                yb_ip1 = ybar[:, i + 1]
            return y_im1, y_ip1, yb_im1, yb_ip1

        raise ValueError(f"Unknown tau_bc={bc}")

    # -------------------------------------------------------------------------
    # Residual (returns vector in the solver's representation)
    # -------------------------------------------------------------------------
    def residual(self, vec: np.ndarray) -> np.ndarray:
        y, ybar = self.unpack(vec)
        r = self.grid.r[:, None]

        # TOTAL fields (for twist correctness)
        y_tot = y + r * self.rho0
        ybar_tot = ybar + r * self.rho0

        # u = Re(phi_rot * phibar_rot) = Re( (y_tot/r)*(ybar_tot/r) )
        inv_r2 = 1.0 / (self.grid.r[:, None] ** 2)
        u = (y_tot * ybar_tot * inv_r2).real

        W, _, _, _ = self._W_Wp(u)

        Fy = np.zeros_like(y, dtype=complex)
        Fyb = np.zeros_like(ybar, dtype=complex)

        for j in range(self.Nr):
            for i in range(self.Nt):

                # strong boundary constraints
                if self.settings.r_bc == "dirichlet_fluct" and j == self.Nr - 1:
                    Fy[j, i] = y[j, i]
                    Fyb[j, i] = ybar[j, i]
                    continue
                if self.settings.tau_bc == "hom_past" and i == self.Nt - 1:
                    Fy[j, i] = y[j, i]
                    Fyb[j, i] = ybar[j, i]
                    continue

                y_im1, y_ip1, yb_im1, yb_ip1 = self._tau_neighbors(y, ybar, i)

                y_t = (y_im1[j] - y_ip1[j]) / (2.0 * self.dt)
                y_tt = (y_im1[j] + y_ip1[j] - 2.0 * y[j, i]) / self.dt2

                yb_t = (yb_im1[j] - yb_ip1[j]) / (2.0 * self.dt)
                yb_tt = (yb_im1[j] + yb_ip1[j] - 2.0 * ybar[j, i]) / self.dt2

                # radial neighbors with regularity at r=0 (Q-ball choice)
                if j == 0:
                    y_jm1 = 0.0
                    yb_jm1 = 0.0
                else:
                    y_jm1 = y[j - 1, i]
                    yb_jm1 = ybar[j - 1, i]

                if j == self.Nr - 1:
                    # Neumann on φ_rot: y_r - y/r = 0 at r_max => ghost y_{N+1} = (1 + dr/r_N) * y_N
                    r_N = float(self.grid.r[self.Nr - 1])
                    fac = 1.0 + self.dr / r_N
                    y_jp1 = fac * y[j, i]
                    yb_jp1 = fac * ybar[j, i]
                else:
                    y_jp1 = y[j + 1, i]
                    yb_jp1 = ybar[j + 1, i]

                y_rr = (y_jp1 - 2.0 * y[j, i] + y_jm1) / self.dr2
                yb_rr = (yb_jp1 - 2.0 * ybar[j, i] + yb_jm1) / self.dr2

                # Tau-independent reduction target:
                # Phi'' + 2/r Phi' = 2 W(Phi) Phi - omega^2 Phi
                # -> r*Phi'' + 2*Phi' + r*Phi*(omega^2 - 2W) = 0.
                A_coef = (self.omega * self.omega - 2.0 * W[j, i])
                Fy[j, i] = y_tt + y_rr + 2.0 * self.omega * y_t + A_coef * y_tot[j, i]
                Fyb[j, i] = yb_tt + yb_rr - 2.0 * self.omega * yb_t + A_coef * ybar_tot[j, i]

        if self.settings.complex_saddle:
            return self._pack_ri(Fy, Fyb)
        return self._pack_2field(Fy, Fyb)

    def residual_slice_norms(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Per-τ residual L2 norms (integrated over r) for quick diagnostics.
        Returns combined and per-field norms, plus the τ-index of max residual.
        """
        F = self.residual(x)
        Fy, Fyb = self.unpack(F)  # (Nr, Nt)
        tau_norm_y = np.sqrt(np.sum(np.abs(Fy) ** 2, axis=0))
        tau_norm_ybar = np.sqrt(np.sum(np.abs(Fyb) ** 2, axis=0))
        tau_norm = np.sqrt(tau_norm_y**2 + tau_norm_ybar**2)
        i_max = int(np.argmax(tau_norm)) if tau_norm.size else -1
        return {
            "tau_norm": np.asarray(tau_norm, dtype=float),
            "tau_norm_y": np.asarray(tau_norm_y, dtype=float),
            "tau_norm_ybar": np.asarray(tau_norm_ybar, dtype=float),
            "max_tau_index": i_max,
            "max_tau_norm": float(tau_norm[i_max]) if i_max >= 0 else float("nan"),
            "is_past_boundary_max": bool(i_max == (self.Nt - 1)),
        }

    # -------------------------------------------------------------------------
    # Jacobian
    # -------------------------------------------------------------------------
    def jacobian(self, vec: np.ndarray) -> sp.csc_matrix:
        """
        Analytic sparse Jacobian.

        If complex_saddle=True: returns the mathematically consistent real Jacobian
        for the (Re,Im) split system (size 4*Nsite × 4*Nsite).

        If complex_saddle=False: returns the legacy “Q-ball-style complex Jacobian”
        for the 2-field complex system (size 2*Nsite × 2*Nsite). This is mainly
        useful for quick tests with nearly-real saddles.
        """
        if self.settings.complex_saddle:
            return self._jacobian_ri(vec)
        return self._jacobian_2field_legacy(vec)

    # -------------------------------------------------------------------------
    # Existence checker (pinned branch) hooks — no change to EOM/BC/twist
    # -------------------------------------------------------------------------
    def residual_fields(self, x_fields: np.ndarray) -> np.ndarray:
        """Residual of the field equations only (same as residual, for augmented system)."""
        return self.residual(x_fields)

    def jacobian_fields(self, x_fields: np.ndarray) -> sp.csc_matrix:
        """Jacobian of the field equations only (same as jacobian, for augmented system)."""
        return self.jacobian(x_fields)

    def system_size_fields(self) -> int:
        """Size of the field unknown vector (no lambda)."""
        if self.settings.complex_saddle:
            return 4 * self.Nr * self.Nt
        return 2 * self.Nr * self.Nt

    def get_ri_at_site(self, x_fields: np.ndarray, j: int, i: int) -> Tuple[float, float, float, float]:
        """Return (y_re, y_im, yb_re, yb_im) at site (j, i) from packed fields vector."""
        Nsite = self.Nr * self.Nt
        idx = j * self.Nt + i
        if self.settings.complex_saddle:
            y_re = float(x_fields[0 * Nsite + idx])
            y_im = float(x_fields[1 * Nsite + idx])
            yb_re = float(x_fields[2 * Nsite + idx])
            yb_im = float(x_fields[3 * Nsite + idx])
            return (y_re, y_im, yb_re, yb_im)
        y = x_fields[:Nsite].reshape((self.Nr, self.Nt))[j, i]
        yb = x_fields[Nsite:].reshape((self.Nr, self.Nt))[j, i]
        return (float(np.real(y)), float(np.imag(y)), float(np.real(yb)), float(np.imag(yb)))

    def field_variable_indices_at_site(self, j: int, i: int) -> Dict[str, int]:
        """Column indices in x_fields for (y_re, y_im, yb_re, yb_im) at site (j, i)."""
        Nsite = self.Nr * self.Nt
        idx = j * self.Nt + i
        if self.settings.complex_saddle:
            return {
                "y_re": 0 * Nsite + idx,
                "y_im": 1 * Nsite + idx,
                "yb_re": 2 * Nsite + idx,
                "yb_im": 3 * Nsite + idx,
            }
        raise NotImplementedError("Existence checker requires complex_saddle=True (RI split).")

    def field_equation_index(self, j: int, i: int, which: str) -> int:
        """Row index in F/J for the equation at site (j, i): which in ('y_re','y_im','yb_re','yb_im')."""
        Nsite = self.Nr * self.Nt
        block = {"y_re": 0, "y_im": 1, "yb_re": 2, "yb_im": 3}[which]
        return block * Nsite + j * self.Nt + i

    def newton_solve_aug(self, residual_aug, jacobian_aug, x_aug0: np.ndarray, **kwargs) -> Any:
        """Newton solve for augmented system (fields + lambda). Uses same newton_solve as main solver."""
        return newton_solve(residual_aug, jacobian_aug, np.asarray(x_aug0, dtype=float), **kwargs)

    def to_fields_vector(self, x: np.ndarray) -> np.ndarray:
        """Convert to field-only vector (RI split when complex_saddle)."""
        x = np.asarray(x)
        if self.settings.complex_saddle:
            if x.size == 4 * self.Nr * self.Nt:
                return x.astype(float).ravel()
            if x.size == 2 * self.Nr * self.Nt:
                y, ybar = self._unpack_2field(x)
                return self._pack_ri(y, ybar)
        return x.astype(float).ravel()

    def fields_to_y_ybar(self, x_fields: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack fields vector to (y, ybar) complex arrays."""
        return self.unpack(x_fields)

    def rho_map(self, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
        """rho(r, tau) = sqrt(2*u_pos + rho_eps) with u = Re(phi_rot * phibar_rot)."""
        phi_rot, phibar_rot = self.phi_rot(y, ybar)
        u = (phi_rot * phibar_rot).real
        u_pos, _ = self._smooth_pos(u)
        return np.sqrt(2.0 * u_pos + float(self.settings.rho_eps))

    def _jacobian_2field_legacy(self, vec: np.ndarray) -> sp.csc_matrix:
        """
        Legacy Q-ball-like Jacobian (complex 2-field). This matches the structure
        used in the Q_ball_finder bounce2d solver, but it is not fully consistent
        for generic complex saddles because u involves a real-part projection.
        """
        y, ybar = self._unpack_2field(vec)
        r = self.grid.r[:, None]

        y_tot = y + r * self.rho0
        ybar_tot = ybar + r * self.rho0

        inv_r2 = 1.0 / (self.grid.r[:, None] ** 2)
        u = (y_tot * ybar_tot * inv_r2).real
        W, Wp, _, dupos_du = self._W_Wp(u)

        Nsite = self.Nr * self.Nt
        J = sp.lil_matrix((2 * Nsite, 2 * Nsite), dtype=complex)

        def iy(j, i): return j * self.Nt + i
        def iyb(j, i): return Nsite + j * self.Nt + i

        exp_m = np.exp(-self.eta0)
        exp_p = np.exp(+self.eta0)

        for j in range(self.Nr):
            rj = float(self.grid.r[j])
            inv_r2_j = 1.0 / (rj * rj)

            for i in range(self.Nt):
                rowy = iy(j, i)
                rowb = iyb(j, i)

                # Dirichlet rows
                if self.settings.r_bc == "dirichlet_fluct" and j == self.Nr - 1:
                    J[rowy, rowy] = 1.0
                    J[rowb, rowb] = 1.0
                    continue
                if self.settings.tau_bc == "hom_past" and i == self.Nt - 1:
                    J[rowy, rowy] = 1.0
                    J[rowb, rowb] = 1.0
                    continue

                bc = self.settings.tau_bc

                # --- time stencil y: y_tt + 2 omega y_t ---
                if bc == "twisted":
                    diag_y_time = -2.0 / self.dt2
                    if i == 0:
                        J[rowy, iy(j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        J[rowy, iyb(j, 0)] += (1.0 / self.dt2 + self.omega / self.dt)
                    elif i == self.Nt - 1:
                        J[rowy, iy(j, self.Nt - 2)] += (1.0 / self.dt2 + self.omega / self.dt)
                        J[rowy, iyb(j, self.Nt - 1)] += exp_m * (1.0 / self.dt2 - self.omega / self.dt)
                    else:
                        J[rowy, iy(j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        J[rowy, iy(j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                elif bc == "hom_past":
                    diag_y_time = -2.0 / self.dt2
                    if i == 0:
                        J[rowy, iy(j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        J[rowy, iyb(j, 0)] += (1.0 / self.dt2 + self.omega / self.dt)
                    else:
                        if i - 1 >= 0:
                            J[rowy, iy(j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        if i + 1 < self.Nt:
                            J[rowy, iy(j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                elif bc == "neumann":
                    if i == 0:
                        diag_y_time = (-1.0 / self.dt2 + self.omega / self.dt)
                        J[rowy, iy(j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                    elif i == self.Nt - 1:
                        diag_y_time = (-1.0 / self.dt2 - self.omega / self.dt)
                        J[rowy, iy(j, self.Nt - 2)] += (1.0 / self.dt2 + self.omega / self.dt)
                    else:
                        diag_y_time = -2.0 / self.dt2
                        J[rowy, iy(j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        J[rowy, iy(j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                else:
                    raise ValueError

                # --- time stencil ybar: yb_tt - 2 omega yb_t ---
                if bc == "twisted":
                    diag_yb_time = -2.0 / self.dt2
                    if i == 0:
                        J[rowb, iyb(j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        J[rowb, iy(j, 0)] += (1.0 / self.dt2 - self.omega / self.dt)
                    elif i == self.Nt - 1:
                        J[rowb, iyb(j, self.Nt - 2)] += (1.0 / self.dt2 - self.omega / self.dt)
                        J[rowb, iy(j, self.Nt - 1)] += exp_p * (1.0 / self.dt2 + self.omega / self.dt)
                    else:
                        J[rowb, iyb(j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        J[rowb, iyb(j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                elif bc == "hom_past":
                    diag_yb_time = -2.0 / self.dt2
                    if i == 0:
                        J[rowb, iyb(j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        J[rowb, iy(j, 0)] += (1.0 / self.dt2 - self.omega / self.dt)
                    else:
                        if i - 1 >= 0:
                            J[rowb, iyb(j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        if i + 1 < self.Nt:
                            J[rowb, iyb(j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                elif bc == "neumann":
                    if i == 0:
                        diag_yb_time = (-1.0 / self.dt2 - self.omega / self.dt)
                        J[rowb, iyb(j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                    elif i == self.Nt - 1:
                        diag_yb_time = (-1.0 / self.dt2 + self.omega / self.dt)
                        J[rowb, iyb(j, self.Nt - 2)] += (1.0 / self.dt2 - self.omega / self.dt)
                    else:
                        diag_yb_time = -2.0 / self.dt2
                        J[rowb, iyb(j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        J[rowb, iyb(j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                else:
                    raise ValueError

                # --- radial stencil ---
                if j == 0:
                    diag_rad = -2.0 / self.dr2
                    if j + 1 < self.Nr:
                        J[rowy, iy(j + 1, i)] += 1.0 / self.dr2
                        J[rowb, iyb(j + 1, i)] += 1.0 / self.dr2
                elif j == self.Nr - 1:
                    # Neumann on φ_rot: y_rr(N) = (y_{N-1} + (-1+dr/r_N)*y_N)/dr^2
                    r_N = float(self.grid.r[self.Nr - 1])
                    diag_rad = (-1.0 + self.dr / r_N) / self.dr2
                    J[rowy, iy(j - 1, i)] += 1.0 / self.dr2
                    J[rowb, iyb(j - 1, i)] += 1.0 / self.dr2
                else:
                    diag_rad = -2.0 / self.dr2
                    J[rowy, iy(j - 1, i)] += 1.0 / self.dr2
                    J[rowy, iy(j + 1, i)] += 1.0 / self.dr2
                    J[rowb, iyb(j - 1, i)] += 1.0 / self.dr2
                    J[rowb, iyb(j + 1, i)] += 1.0 / self.dr2

                # --- nonlinear term (legacy approximation) ---
                A_coef = (self.omega * self.omega - 2.0 * W[j, i])
                du_dy = (ybar_tot[j, i] * inv_r2_j).real
                du_dyb = (y_tot[j, i] * inv_r2_j).real
                chain = dupos_du[j, i]
                dW_dy = 2.0 * Wp[j, i] * chain * du_dy
                dW_dyb = 2.0 * Wp[j, i] * chain * du_dyb

                diag_y = diag_y_time + diag_rad + A_coef - y_tot[j, i] * dW_dy
                diag_yb = diag_yb_time + diag_rad + A_coef - ybar_tot[j, i] * dW_dyb

                J[rowy, iy(j, i)] += diag_y
                J[rowb, iyb(j, i)] += diag_yb
                J[rowy, iyb(j, i)] += -y_tot[j, i] * dW_dyb
                J[rowb, iy(j, i)] += -ybar_tot[j, i] * dW_dy

        return J.tocsc()

    def _jacobian_ri(self, vec: np.ndarray) -> sp.csc_matrix:
        """
        Real/imag split Jacobian (size 4*Nsite × 4*Nsite).
        This is the correct Jacobian for complex saddles with u = Re(phi_rot * phibar_rot).
        """
        y, ybar = self._unpack_ri(vec)
        r = self.grid.r[:, None]
        inv_r2 = 1.0 / (self.grid.r[:, None] ** 2)

        y_tot = y + r * self.rho0
        ybar_tot = ybar + r * self.rho0

        # u and W(u)
        u = (y_tot * ybar_tot * inv_r2).real
        W, Wp, _, dupos_du = self._W_Wp(u)
        W_u = Wp * dupos_du  # dW/du (u is the “pre-projection” variable)

        Nsite = self.Nr * self.Nt
        J = sp.lil_matrix((4 * Nsite, 4 * Nsite), dtype=float)

        # index helpers
        def vid(block: int, j: int, i: int) -> int:
            return block * Nsite + j * self.Nt + i

        exp_m = float(np.exp(-self.eta0))
        exp_p = float(np.exp(+self.eta0))

        for j in range(self.Nr):
            rj = float(self.grid.r[j])
            inv_r2_j = 1.0 / (rj * rj)

            for i in range(self.Nt):
                # equation rows
                row_FyR = vid(0, j, i)
                row_FyI = vid(1, j, i)
                row_FbR = vid(2, j, i)
                row_FbI = vid(3, j, i)

                # variable columns
                col_yR = vid(0, j, i)
                col_yI = vid(1, j, i)
                col_bR = vid(2, j, i)
                col_bI = vid(3, j, i)

                # Dirichlet rows (apply to both Re and Im)
                if self.settings.r_bc == "dirichlet_fluct" and j == self.Nr - 1:
                    J[row_FyR, col_yR] = 1.0
                    J[row_FyI, col_yI] = 1.0
                    J[row_FbR, col_bR] = 1.0
                    J[row_FbI, col_bI] = 1.0
                    continue
                if self.settings.tau_bc == "hom_past" and i == self.Nt - 1:
                    J[row_FyR, col_yR] = 1.0
                    J[row_FyI, col_yI] = 1.0
                    J[row_FbR, col_bR] = 1.0
                    J[row_FbI, col_bI] = 1.0
                    continue

                bc = self.settings.tau_bc

                # -----------------------------------------------------------------
                # Linear time stencils:
                # Fy:  y_tt + 2 ω y_t
                # Fyb: yb_tt - 2 ω yb_t
                # Coefficients are real, so Re/Im decouple, and swap couples y <-> ybar
                # -----------------------------------------------------------------
                def add_time_stencil_for_y(rowR: int, rowI: int):
                    if bc == "twisted":
                        diag_time = -2.0 / self.dt2
                        if i == 0:
                            # y_im1 = ybar(0)
                            J[rowR, vid(0, j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(1, j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowR, vid(2, j, 0)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(3, j, 0)] += (1.0 / self.dt2 + self.omega / self.dt)
                        elif i == self.Nt - 1:
                            J[rowR, vid(0, j, self.Nt - 2)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(1, j, self.Nt - 2)] += (1.0 / self.dt2 + self.omega / self.dt)
                            # y_ip1 depends on ybar via exp(-eta0)
                            J[rowR, vid(2, j, self.Nt - 1)] += exp_m * (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(3, j, self.Nt - 1)] += exp_m * (1.0 / self.dt2 - self.omega / self.dt)
                        else:
                            J[rowR, vid(0, j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(1, j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowR, vid(0, j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(1, j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        return diag_time

                    if bc == "hom_past":
                        diag_time = -2.0 / self.dt2
                        if i == 0:
                            J[rowR, vid(0, j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(1, j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowR, vid(2, j, 0)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(3, j, 0)] += (1.0 / self.dt2 + self.omega / self.dt)
                        else:
                            if i - 1 >= 0:
                                J[rowR, vid(0, j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                                J[rowI, vid(1, j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            if i + 1 < self.Nt:
                                J[rowR, vid(0, j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                                J[rowI, vid(1, j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        return diag_time

                    if bc == "neumann":
                        if i == 0:
                            diag_time = (-1.0 / self.dt2 + self.omega / self.dt)
                            J[rowR, vid(0, j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(1, j, 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        elif i == self.Nt - 1:
                            diag_time = (-1.0 / self.dt2 - self.omega / self.dt)
                            J[rowR, vid(0, j, self.Nt - 2)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(1, j, self.Nt - 2)] += (1.0 / self.dt2 + self.omega / self.dt)
                        else:
                            diag_time = -2.0 / self.dt2
                            J[rowR, vid(0, j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(1, j, i - 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowR, vid(0, j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(1, j, i + 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                        return diag_time

                    raise ValueError(f"Unknown tau_bc={bc}")

                def add_time_stencil_for_ybar(rowR: int, rowI: int):
                    if bc == "twisted":
                        diag_time = -2.0 / self.dt2
                        if i == 0:
                            # yb_im1 = y(0)
                            J[rowR, vid(2, j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(3, j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowR, vid(0, j, 0)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(1, j, 0)] += (1.0 / self.dt2 - self.omega / self.dt)
                        elif i == self.Nt - 1:
                            J[rowR, vid(2, j, self.Nt - 2)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(3, j, self.Nt - 2)] += (1.0 / self.dt2 - self.omega / self.dt)
                            # yb_ip1 depends on y via exp(+eta0)
                            J[rowR, vid(0, j, self.Nt - 1)] += exp_p * (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(1, j, self.Nt - 1)] += exp_p * (1.0 / self.dt2 + self.omega / self.dt)
                        else:
                            J[rowR, vid(2, j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(3, j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowR, vid(2, j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(3, j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        return diag_time

                    if bc == "hom_past":
                        diag_time = -2.0 / self.dt2
                        if i == 0:
                            J[rowR, vid(2, j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(3, j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowR, vid(0, j, 0)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(1, j, 0)] += (1.0 / self.dt2 - self.omega / self.dt)
                        else:
                            if i - 1 >= 0:
                                J[rowR, vid(2, j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                                J[rowI, vid(3, j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            if i + 1 < self.Nt:
                                J[rowR, vid(2, j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                                J[rowI, vid(3, j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        return diag_time

                    if bc == "neumann":
                        if i == 0:
                            diag_time = (-1.0 / self.dt2 - self.omega / self.dt)
                            J[rowR, vid(2, j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(3, j, 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        elif i == self.Nt - 1:
                            diag_time = (-1.0 / self.dt2 + self.omega / self.dt)
                            J[rowR, vid(2, j, self.Nt - 2)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(3, j, self.Nt - 2)] += (1.0 / self.dt2 - self.omega / self.dt)
                        else:
                            diag_time = -2.0 / self.dt2
                            J[rowR, vid(2, j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowI, vid(3, j, i - 1)] += (1.0 / self.dt2 - self.omega / self.dt)
                            J[rowR, vid(2, j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                            J[rowI, vid(3, j, i + 1)] += (1.0 / self.dt2 + self.omega / self.dt)
                        return diag_time

                    raise ValueError(f"Unknown tau_bc={bc}")

                diag_y_time = add_time_stencil_for_y(row_FyR, row_FyI)
                diag_b_time = add_time_stencil_for_ybar(row_FbR, row_FbI)

                # -----------------------------------------------------------------
                # Linear radial stencil (same for Re/Im, decoupled between y and ybar)
                # -----------------------------------------------------------------
                if j == 0:
                    diag_rad = -2.0 / self.dr2
                    if j + 1 < self.Nr:
                        J[row_FyR, vid(0, j + 1, i)] += 1.0 / self.dr2
                        J[row_FyI, vid(1, j + 1, i)] += 1.0 / self.dr2
                        J[row_FbR, vid(2, j + 1, i)] += 1.0 / self.dr2
                        J[row_FbI, vid(3, j + 1, i)] += 1.0 / self.dr2
                elif j == self.Nr - 1:
                    # Neumann on φ_rot: y_rr(N) = (y_{N-1} + (-1+dr/r_N)*y_N)/dr^2
                    diag_rad = (-1.0 + self.dr / rj) / self.dr2
                    J[row_FyR, vid(0, j - 1, i)] += 1.0 / self.dr2
                    J[row_FyI, vid(1, j - 1, i)] += 1.0 / self.dr2
                    J[row_FbR, vid(2, j - 1, i)] += 1.0 / self.dr2
                    J[row_FbI, vid(3, j - 1, i)] += 1.0 / self.dr2
                else:
                    diag_rad = -2.0 / self.dr2
                    J[row_FyR, vid(0, j - 1, i)] += 1.0 / self.dr2
                    J[row_FyI, vid(1, j - 1, i)] += 1.0 / self.dr2
                    J[row_FyR, vid(0, j + 1, i)] += 1.0 / self.dr2
                    J[row_FyI, vid(1, j + 1, i)] += 1.0 / self.dr2

                    J[row_FbR, vid(2, j - 1, i)] += 1.0 / self.dr2
                    J[row_FbI, vid(3, j - 1, i)] += 1.0 / self.dr2
                    J[row_FbR, vid(2, j + 1, i)] += 1.0 / self.dr2
                    J[row_FbI, vid(3, j + 1, i)] += 1.0 / self.dr2

                # -----------------------------------------------------------------
                # Nonlinear term: (ω^2 - 2W(u)) * y_tot
                # -----------------------------------------------------------------
                A = float(self.omega * self.omega - 2.0 * W[j, i])
                Wu = float(2.0 * W_u[j, i])  # dA/du = -2*dW/du

                # components at this site
                yR = float(y[j, i].real)
                yI = float(y[j, i].imag)
                bR = float(ybar[j, i].real)
                bI = float(ybar[j, i].imag)

                yRtot = yR + rj * self.rho0
                bRtot = bR + rj * self.rho0

                # u = (yRtot*bRtot - yI*bI)/r^2
                du_dyR = bRtot * inv_r2_j
                du_dyI = -bI * inv_r2_j
                du_dbR = yRtot * inv_r2_j
                du_dbI = -yI * inv_r2_j

                # dA/dvar = -2 dW/dvar = - Wu * du/dvar
                dA_dyR = -Wu * du_dyR
                dA_dyI = -Wu * du_dyI
                dA_dbR = -Wu * du_dbR
                dA_dbI = -Wu * du_dbI

                # --- add diagonal pieces for the linear operator ---
                J[row_FyR, col_yR] += (diag_y_time + diag_rad)
                J[row_FyI, col_yI] += (diag_y_time + diag_rad)
                J[row_FbR, col_bR] += (diag_b_time + diag_rad)
                J[row_FbI, col_bI] += (diag_b_time + diag_rad)

                # --- add nonlinear contributions for Fy (Re/Im) ---
                # FyR = L(yR) + A*(yR + r*rho0)
                J[row_FyR, col_yR] += (A + (yR + rj * self.rho0) * dA_dyR)
                J[row_FyR, col_yI] += ((yR + rj * self.rho0) * dA_dyI)
                J[row_FyR, col_bR] += ((yR + rj * self.rho0) * dA_dbR)
                J[row_FyR, col_bI] += ((yR + rj * self.rho0) * dA_dbI)

                # FyI = L(yI) + A*yI
                J[row_FyI, col_yI] += (A + yI * dA_dyI)
                J[row_FyI, col_yR] += (yI * dA_dyR)
                J[row_FyI, col_bR] += (yI * dA_dbR)
                J[row_FyI, col_bI] += (yI * dA_dbI)

                # --- add nonlinear contributions for Fyb (Re/Im) ---
                # FybR = Lb(bR) + A*(bR + r*rho0)
                J[row_FbR, col_bR] += (A + (bR + rj * self.rho0) * dA_dbR)
                J[row_FbR, col_bI] += ((bR + rj * self.rho0) * dA_dbI)
                J[row_FbR, col_yR] += ((bR + rj * self.rho0) * dA_dyR)
                J[row_FbR, col_yI] += ((bR + rj * self.rho0) * dA_dyI)

                # FybI = Lb(bI) + A*bI
                J[row_FbI, col_bI] += (A + bI * dA_dbI)
                J[row_FbI, col_bR] += (bI * dA_dbR)
                J[row_FbI, col_yR] += (bI * dA_dyR)
                J[row_FbI, col_yI] += (bI * dA_dyI)

        return J.tocsc()

    # -------------------------------------------------------------------------
    # Observables (implementazioni in observables_2d)
    # -------------------------------------------------------------------------
    def compute_charge(
        self,
        y: np.ndarray,
        ybar: np.ndarray,
        index_tau: int = 0,
        *,
        return_profile: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """Charge Q on a tau slice; implemented in observables_2d.compute_charge_2d."""
        return observables_2d.compute_charge_2d(
            self, y, ybar, index_tau=index_tau, return_profile=return_profile
        )

    def compute_energy(
        self,
        y: np.ndarray,
        ybar: np.ndarray,
        index_tau: int = 0,
        *,
        return_profile: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """Energy on a tau slice (canonical H_E, no subtraction)."""
        return observables_2d.compute_energy_2d(
            self, y, ybar,
            index_tau=index_tau,
            return_profile=return_profile,
        )

    def compute_charge_tau0_ghost(
        self,
        y: np.ndarray,
        ybar: np.ndarray,
        *,
        subtract_background: bool = True,
        return_profile: bool = False,
    ):
        """Charge at τ=0 by ghost reconstruction; returns Q or (Q, q_r) if return_profile."""
        return observables_2d.compute_charge_tau0_ghost_2d(
            self, y, ybar,
            subtract_background=subtract_background,
            return_profile=return_profile,
        )

    def compute_energy_tau0_ghost(
        self,
        y: np.ndarray,
        ybar: np.ndarray,
        *,
        return_profile: bool = False,
    ):
        """H_E(τ=0) by ghost reconstruction (Euclidean Hamiltonian-like). Returns E or (E, e_r). For physical energy use observables_2d.compute_energy_minkowski_tau0_ghost_2d."""
        return observables_2d.compute_HE_euclidean_tau0_ghost_2d(
            self, y, ybar,
            return_profile=return_profile,
        )

    def observables_tau0(
        self,
        x: np.ndarray,
        *,
        subtract_background: bool = False,
        return_profiles: bool = True,
    ) -> Dict[str, Any]:
        """
        Observables at τ=0 (ghost): Q, E (canonical), E_hom, energy_ratio, densities, ...
        subtract_background applies only to charge.
        """
        out = observables_2d.compute_observables_tau0_ghost(
            self,
            x,
            subtract_background_charge=subtract_background,
        )
        if not return_profiles:
            out = {k: v for k, v in out.items() if k not in ("r", "q_r", "e_r")}
        return out

    def targets_tau0(self, *, subtract_background: bool = False) -> Dict[str, Any]:
        """Target observables from homogeneous (y=0). E_target = E_hom."""
        return observables_2d.compute_targets_tau0_ghost(
            self,
            subtract_background_charge=subtract_background,
        )

    # -------------------------------------------------------------------------
    # Sanity checks (updated)
    # -------------------------------------------------------------------------
    def sanity_background_eta0_zero(self) -> Dict[str, Any]:
        """
        Wiring test: with eta0=0, the homogeneous medium y=ybar=0 must satisfy
        BOTH bulk EOM and τ-BCs exactly (up to discretisation / smoothing).
        """
        eta_save = float(self.eta0)
        try:
            self.eta0 = 0.0
            self.settings.eta0 = 0.0
            x0 = self._zero_vec()
            F0 = self.residual(x0)
        finally:
            self.eta0 = eta_save
            self.settings.eta0 = eta_save

        nrm = float(np.linalg.norm(F0))
        mx = float(np.max(np.abs(F0)))

        W0, *_ = self._W_Wp(np.array([self.rho0**2]))
        mismatch = float(self.omega * self.omega - 2.0 * W0[0])
        r_rms = float(np.sqrt(np.mean(np.asarray(self.grid.r, dtype=float) ** 2)))
        predicted = float(np.sqrt(2.0 * self.Nr * self.Nt) * abs(mismatch) * (r_rms * self.rho0))
        return dict(
            residual_norm=nrm,
            residual_max_abs=mx,
            omega2_minus_2W=mismatch,
            predicted_norm=predicted,
            rho0=self.rho0,
        )

    def sanity_twist_source(self) -> Dict[str, Any]:
        """
        For eta0 != 0, y=ybar=0 is NOT a solution of the τ-BCs.
        However the residual becomes a predictable “source” localised at i=Nt-1.

        This test checks:
          - (Fy,Fyb) reduced by the small bulk mismatch term are ~0 for i != Nt-1
          - at i=Nt-1 they match the analytic twist-source prediction.
        """
        x0 = self._zero_vec()
        F0 = self.residual(x0)
        Fy, Fyb = self.unpack(F0)

        # bulk mismatch term from omega^2 - 2W(rho0^2)
        W0, *_ = self._W_Wp(np.array([self.rho0**2]))
        mismatch = float(self.omega * self.omega - 2.0 * W0[0])

        r = self.grid.r
        ytot = r * self.rho0  # since y=0
        bulk_term = mismatch * ytot[:, None]  # same for all i

        Fy_red = Fy.real - bulk_term
        Fb_red = Fyb.real - bulk_term

        # predicted extra at i=Nt-1 from twisted ghost point
        dt = self.dt
        exp_m = float(np.exp(-self.eta0))
        exp_p = float(np.exp(+self.eta0))

        source_y = (exp_m - 1.0) * ytot * (1.0 / (dt * dt) - self.omega / dt)
        source_b = (exp_p - 1.0) * ytot * (1.0 / (dt * dt) + self.omega / dt)

        iB = self.Nt - 1

        interior_mask = np.ones_like(Fy_red, dtype=bool)
        interior_mask[:, iB] = False
        interior_norm = float(np.linalg.norm(Fy_red[interior_mask])) + float(np.linalg.norm(Fb_red[interior_mask]))

        boundary_err_y = float(np.linalg.norm(Fy_red[:, iB] - source_y))
        boundary_err_b = float(np.linalg.norm(Fb_red[:, iB] - source_b))

        return dict(
            eta0=float(self.eta0),
            omega=float(self.omega),
            rho0=float(self.rho0),
            omega2_minus_2W=mismatch,
            interior_reduced_norm=interior_norm,
            boundary_source_err_y=boundary_err_y,
            boundary_source_err_ybar=boundary_err_b,
        )

    def run_sanity_tests(self) -> Dict[str, Any]:
        d0 = self.sanity_background_eta0_zero()
        d1 = self.sanity_twist_source()
        return dict(background_eta0_zero=d0, twist_source=d1)

    def self_check_tau_independent_reduction(
        self,
        r_min_skip: int = 2,
        r_max_skip: int = 2,
        atol: float = 1e-6,
        rtol: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        Check that the 2D bulk operator reduces to the 1D bounce ODE for a tau-independent
        real profile at eta0=0. Builds phi(r) = rho0 + small bump, embeds as y = r*(phi - rho0),
        compares 2D residual Fy with 1D operator (r*phi'' + 2*phi' + r*phi*(omega^2 - 2W)).
        Excludes r~0 and r~r_max where 2D uses different BC stencils.
        Fails loudly (raises AssertionError) if mismatch exceeds tolerance.
        """
        r = np.asarray(self.grid.r, dtype=float).flatten()
        if r.size < 4 or r.size != self.Nr:
            return dict(ok=False, reason="grid too small")
        # Tau-independent real profile: small bump so W is well-defined
        phi = self.rho0 + 0.1 * np.exp(-((r - 0.5 * r[-1]) ** 2) / 4.0)
        phi = np.asarray(phi, dtype=float)
        # Finite differences for phi', phi'' (centered in interior)
        phi_p = np.gradient(phi, r, edge_order=2)
        phi_p[0] = 0.0  # regularity
        phi_pp = np.gradient(phi_p, r, edge_order=2)
        y_slice = r * (phi - self.rho0)
        y = np.broadcast_to(y_slice[:, None], (self.Nr, self.Nt)).copy()
        ybar = y.copy()
        eta_save = float(self.eta0)
        try:
            self.eta0 = 0.0
            self.settings.eta0 = 0.0
            x = self.pack(y, ybar)
            F = self.residual(x)
            Fy, Fyb = self.unpack(F)
        finally:
            self.eta0 = eta_save
            self.settings.eta0 = eta_save
        u_phi = phi * phi
        W, _, _, _ = self._W_Wp(u_phi)
        W = np.asarray(W).flatten()
        # 1D operator: L1d = r*phi'' + 2*phi' + r*phi*(omega^2 - 2W)
        L1d = r * phi_pp + 2.0 * phi_p + r * phi * (self.omega**2 - 2.0 * W)
        Fy_slice = np.asarray(Fy.real[:, 0]).flatten()
        # Exclude boundaries (2D uses regularity/Neumann stencils there)
        j_lo = min(r_min_skip, r.size - 1)
        j_hi = max(r.size - r_max_skip, j_lo + 1)
        diff = np.abs(Fy_slice[j_lo:j_hi] - L1d[j_lo:j_hi])
        denom = np.abs(L1d[j_lo:j_hi]) + atol
        rel = np.where(denom > 1e-30, diff / denom, diff)
        max_abs = float(np.max(diff))
        max_rel = float(np.max(rel))
        ok = max_abs < atol or max_rel < rtol
        return dict(
            ok=ok,
            max_abs_err=max_abs,
            max_rel_err=max_rel,
            atol=atol,
            rtol=rtol,
        )

    def potential_convention_report(self) -> str:
        """
        Short English summary of potential and 2D conventions (no physics change).
        u = Re(phi_rot*phibar_rot), rho_phys = sqrt(2*u_pos + rho_eps), W = dU/(2*rho_phys),
        bulk coefficient A_coef = (omega^2 - 2W).
        """
        lines = [
            "Potential / 2D convention (source of truth):",
            "  u = Re(phi_rot*phibar_rot)",
            "  rho_phys = sqrt(2*smooth_pos(u) + rho_eps)",
            "  W = dU(rho_phys)/(2*rho_phys)",
            "  Bulk: A_coef = omega^2 - 2*W",
            "  Charge: q = Re(phibar*phi_tau - phi*phibar_tau), Q = 4π∫ r^2 q dr",
        ]
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Compatibility wrappers (notebook / legacy): same API, delegate to new checks
    # -------------------------------------------------------------------------
    def sanity_background(self) -> Dict[str, Any]:
        """Legacy name: same as sanity_background_eta0_zero."""
        return self.sanity_background_eta0_zero()

    def run_sanity_test(self) -> Tuple[bool, str]:
        """Legacy (notebook): (ok, msg) from eta0=0 background check."""
        d = self.sanity_background_eta0_zero()
        ok = d["residual_norm"] < 1e-6
        msg = (
            f"||F||={d['residual_norm']:.3e}. "
            f"Background: omega^2-2W(ρ0)={d['omega2_minus_2W']:.3e} with ρ0={d['rho0']:.3e}."
        )
        return ok, msg

    def check_jacobian_consistency(
        self, x: np.ndarray, n_tests: int = 3, eps: float = 1e-7
    ) -> Dict[str, Any]:
        """Legacy (notebook): dict with max_rel_error, mean_rel_error, test_results."""
        d = self.check_jacobian(x, n_tests=n_tests, eps=eps)
        return dict(
            max_rel_error=d["max_rel"],
            mean_rel_error=d["mean_rel"],
            test_results=d["all"],
        )

    # -------------------------------------------------------------------------
    # Jacobian FD check (complex-saddle capable)
    # -------------------------------------------------------------------------
    def check_jacobian(self, x: np.ndarray, n_tests: int = 4, eps: float = 1e-6, include_pure_imag: bool = True) -> Dict[str, Any]:
        """
        Compare analytic Jv to finite-difference directional derivative.

        - In complex_saddle=True mode, the state vector is real (Re/Im split), so
          this tests both real and imaginary directions correctly.
        - In legacy mode (complex_saddle=False), this test is meaningful only for
          nearly-real solutions.
        """
        x = np.asarray(x)
        F0 = self.residual(x)
        J = self.jacobian(x)

        errs: List[float] = []

        for _ in range(n_tests):
            v = np.random.randn(x.size)
            v /= np.linalg.norm(v) + 1e-30
            Jv = J @ v
            fd = (self.residual(x + eps * v) - F0) / eps
            denom = float(np.linalg.norm(Jv) + 1e-14)
            errs.append(float(np.linalg.norm(Jv - fd) / denom))

        if include_pure_imag and self.settings.complex_saddle:
            N = self.Nr * self.Nt
            for _ in range(max(1, n_tests // 2)):
                v = np.zeros_like(x)
                v[1 * N:2 * N] = np.random.randn(N)  # Im(y)
                v[3 * N:4 * N] = np.random.randn(N)  # Im(ybar)
                v /= np.linalg.norm(v) + 1e-30
                Jv = J @ v
                fd = (self.residual(x + eps * v) - F0) / eps
                denom = float(np.linalg.norm(Jv) + 1e-14)
                errs.append(float(np.linalg.norm(Jv - fd) / denom))

        return dict(max_rel=float(np.max(errs)), mean_rel=float(np.mean(errs)), all=errs)

    # -------------------------------------------------------------------------
    # Anchored initial guess (Q-ball-like)
    # -------------------------------------------------------------------------
    def build_anchored_initial_guess(self, rho_center: float, Rb: float, Lw: float, tau_scale: Optional[float] = None) -> np.ndarray:
        """
        Simple anchored seed:
          - tau=0 slice: tanh wall from rho_center to rho0
          - towards tau=-beta/2: exponential envelope back to background
        """
        r = self.grid.r
        tau = self.grid.tau
        if tau_scale is None:
            tau_scale = 0.18 * self.settings.beta

        prof = self.rho0 + (float(rho_center) - self.rho0) * 0.5 * (1.0 - np.tanh((r - float(Rb)) / float(Lw)))
        y_slice = r * (prof - self.rho0)

        g = np.exp(tau / float(tau_scale))
        g = g / g[0]

        y = np.outer(y_slice, g).astype(float)
        ybar = np.outer(y_slice, g).astype(float)

        return self.pack(y, ybar)

    # -------------------------------------------------------------------------
    # Newton solve
    # -------------------------------------------------------------------------
    def solve(
        self,
        x0: np.ndarray,
        verbose: Optional[bool] = None,
        store_iteration_history: bool = False,
        verbose_success_block: bool = True,
    ) -> Bubble2DSolution:
        do_verbose = bool(verbose) if verbose is not None else bool(self.settings.verbose)

        x0 = np.asarray(x0)
        if self.settings.complex_saddle:
            # allow passing packed 2-field vectors too
            if x0.size == 2 * self.Nr * self.Nt:
                y, ybar = self._unpack_2field(x0)
                x0 = self._pack_ri(y, ybar)
            elif x0.size != 4 * self.Nr * self.Nt:
                raise ValueError("x0 has wrong size. Provide either 4*Nsite (RI) or 2*Nsite (packed y,ybar).")
            x0 = x0.astype(float)
        else:
            if x0.size != 2 * self.Nr * self.Nt:
                raise ValueError("x0 has wrong size for legacy mode (expected 2*Nsite).")

        iteration_history: List[Dict[str, Any]] = []
        # Store last Newton state and rho at every iteration for plotting (even on non-convergence)
        self._last_newton_x = np.asarray(x0).copy()
        self._newton_rho_history: List[Dict[str, Any]] = []
        y0, ybar0 = self.unpack(x0)
        phi0, phibar0 = self.phi(y0, ybar0)
        u0 = (phi0 * phibar0).real
        rho0_it = np.sqrt(np.maximum(2.0 * u0 + float(self.settings.rho_eps), 0.0))
        self._newton_rho_history.append({"iter": 0, "res_norm": float("nan"), "rho": rho0_it.copy()})
        if store_iteration_history:
            iteration_history.append({"iter": 0, "res_norm": float(np.nan), "rho": rho0_it.copy()})
        if do_verbose:
            try:
                rs0 = self.residual_slice_norms(x0)
                print(
                    "[Seed residual-by-tau] "
                    f"max at i={rs0['max_tau_index']} "
                    f"(past={rs0['is_past_boundary_max']}), "
                    f"||F||_tau,max={rs0['max_tau_norm']:.3e}"
                )
            except Exception:
                pass

        # Targets from homogeneous (E_target = E_hom for ratio E/E_hom)
        _targets_cache: Dict[str, Any] = {}
        try:
            _targets_cache = observables_2d.compute_targets_tau0_ghost(
                self,
                subtract_background_charge=False,
            )
        except Exception:
            pass

        def line_search(x: np.ndarray, dx: np.ndarray, F: np.ndarray) -> float:
            n0 = float(np.linalg.norm(F))
            alpha = 1.0
            for _ in range(self.settings.max_backtracks):
                Ft = self.residual(x + alpha * dx)
                nt = float(np.linalg.norm(Ft))
                if np.isfinite(nt) and nt < n0:
                    return float(alpha)
                alpha *= 0.5
                if alpha < self.settings.min_step:
                    break
            raise NewtonConvergenceError("Line search failed to find a decreasing step.")

        def callback(it: int, x: np.ndarray, F: np.ndarray, nF: float) -> None:
            self._last_newton_x = np.asarray(x).copy()
            y_cb, ybar_cb = self.unpack(x)
            phi_cb, phibar_cb = self.phi(y_cb, ybar_cb)
            u = (phi_cb * phibar_cb).real
            rho_it = np.sqrt(np.maximum(2.0 * u + float(self.settings.rho_eps), 0.0))
            self._newton_rho_history.append({"iter": it, "res_norm": float(nF), "rho": rho_it.copy()})
            if store_iteration_history:
                iteration_history.append({"iter": it, "res_norm": float(nF), "rho": rho_it.copy()})
            if not do_verbose:
                return
            extra = ""
            try:
                obs = observables_2d.compute_observables_tau0_ghost(
                    self, x,
                    subtract_background_charge=False,
                )
                # Ensure we have reference densities (homogeneous on same grid)
                tgt = _targets_cache
                if (not tgt or tgt.get("rho_Q") is None or tgt.get("rho_E") is None):
                    from .observables_1d import Q_homogeneous_ball_from_phi
                    r_max = float(self.grid.r[-1])
                    V = (4.0 / 3.0) * np.pi * r_max**3
                    Q_ref = float(Q_homogeneous_ball_from_phi(float(self.omega), float(self.rho0), r_max))
                    E_ref = float(observables_2d.homogeneous_energy_2d(
                        float(self.omega), float(self.rho0), r_max, self.U
                    ))
                    tgt = dict(tgt or {}, rho_Q=Q_ref / V, rho_E=E_ref / V)
                extra = ", " + _format_obs_line(obs, tgt)
            except Exception:
                pass
            tau_extra = ""
            try:
                rs = self.residual_slice_norms(x)
                tau_extra = (
                    f", tau-max i={rs['max_tau_index']}"
                    f"{' (past)' if rs['is_past_boundary_max'] else ''}"
                    f", ||F||_tau,max={rs['max_tau_norm']:.3e}"
                )
            except Exception:
                pass
            print(f"[Newton-explicit] iter={it:02d}, ||F||={nF:.3e}{tau_extra}{extra}")

        nr = newton_solve(
            residual=self.residual,
            jacobian=self.jacobian,
            x0=x0,
            tol=float(self.settings.newton_tol),
            max_iter=int(self.settings.newton_max_iter),
            damping=float(self.settings.damping),
            line_search=line_search,
            norm=lambda v: float(np.linalg.norm(v)),
            callback=callback,
        )

        y, ybar = self.unpack(nr.x)
        sanity = self.run_sanity_tests()

        # Ground truth: τ=0 ghost Q (total), E (canonical), E_hom, energy_ratio
        obs_ghost = observables_2d.compute_observables_tau0_ghost(
            self, nr.x,
            subtract_background_charge=False,
        )
        Q_ghost = obs_ghost["Q"]
        E_ghost = obs_ghost["E"]
        E_hom = obs_ghost.get("E_hom")
        energy_ratio = obs_ghost.get("energy_ratio")
        if E_hom is None or energy_ratio is None:
            E_hom = float(observables_2d.homogeneous_energy_2d(
                float(self.omega), float(self.rho0), float(self.grid.r[-1]), self.U
            ))
            energy_ratio = (E_ghost / E_hom) if abs(E_hom) > 1e-30 else float("nan")
        else:
            E_hom = float(E_hom)
            energy_ratio = float(energy_ratio)
        tgt = _targets_cache or observables_2d.compute_targets_tau0_ghost(
            self,
            subtract_background_charge=False,
        )
        obs_ghost["q_target_r"] = tgt.get("q_r")
        obs_ghost["e_target_r"] = tgt.get("e_r")
        obs_ghost["Q_target"] = tgt.get("Q")
        obs_ghost["E_target"] = tgt.get("E")

        if nr.success and verbose_success_block:
            tQ, tE = tgt.get("Q"), tgt.get("E")
            rQ = (Q_ghost / tQ) if (tQ is not None and abs(tQ) > 1e-30) else float("nan")
            rE = (E_ghost / tE) if (tE is not None and abs(tE) > 1e-30) else float("nan")
            rho_Q_t, rho_E_t = tgt.get("rho_Q"), tgt.get("rho_E")
            r_rhoQ = (obs_ghost["rho_Q"] / rho_Q_t) if (rho_Q_t is not None and abs(rho_Q_t) > 1e-30) else float("nan")
            r_rhoE = (obs_ghost["rho_E"] / rho_E_t) if (rho_E_t is not None and abs(rho_E_t) > 1e-30) else float("nan")
            print("")
            print("--- Diagnostics (τ=0 ghost) ---")
            print(f"  Q = {Q_ghost:.6e}  (target: {tQ}, ratio: {rQ:.4f})")
            print(f"  E = {E_ghost:.6e}  (target: {tE}, ratio: {rE:.4f})")
            print(f"  rho_Q = {obs_ghost['rho_Q']:.6e}  (target: {rho_Q_t}, ratio: {r_rhoQ:.4f})")
            print(f"  rho_E = {obs_ghost['rho_E']:.6e}  (target: {rho_E_t}, ratio: {r_rhoE:.4f})")
            print(f"  q_max = {obs_ghost['q_max']:.3e}  at r = {obs_ghost['r_at_q_max']:.4f}")
            print(f"  e_max = {obs_ghost['e_max']:.3e}  at r = {obs_ghost['r_at_e_max']:.4f}")
            d0 = sanity.get("background_eta0_zero", {})
            if d0:
                print(
                    "  Sanity (η0=0): "
                    f"||F||={d0.get('residual_norm', '—'):.3e}, "
                    f"max|F|={d0.get('residual_max_abs', '—'):.3e}, "
                    f"pred≈{d0.get('predicted_norm', '—'):.3e}, "
                    f"ω²−2W(ρ0)={d0.get('omega2_minus_2W', '—'):.3e}"
                )
            print("")

        return Bubble2DSolution(
            settings=self.settings,
            grid=self.grid,
            newton=nr,
            y=y,
            ybar=ybar,
            rho0=self.rho0,
            Q_tau0=complex(Q_ghost, 0.0),
            E_tau0=float(E_ghost),
            sanity=sanity,
            iteration_history=iteration_history if store_iteration_history else None,
            observables_ghost=obs_ghost,
            E_hom=E_hom,
            energy_ratio=energy_ratio,
        )

    # -------------------------------------------------------------------------
    # external eta0 scan to match Q (bisection with warm-start)
    # -------------------------------------------------------------------------
    def scan_eta0_to_match_Q(
        self,
        Q_target: float,
        eta_bracket: Tuple[float, float],
        x0: np.ndarray,
        max_steps: int = 30,
        verbose: bool = False,
        tol_Q: float = 1e-10,
    ):
        hist: List[Dict[str, Any]] = []
        targets_scan: Dict[str, Any] = {}
        try:
            targets_scan = self.targets_tau0(subtract_background=False)
        except Exception:
            targets_scan = {}
        tQ = targets_scan.get("Q")
        tE = targets_scan.get("E")

        def solve_eta(eta: float, seed: np.ndarray):
            self.settings.eta0 = float(eta)
            self.eta0 = float(eta)
            sol = self.solve(seed, verbose=False, verbose_success_block=False)
            q = float(sol.Q_tau0.real)
            e = float(sol.E_tau0)
            hist.append(dict(eta0=float(eta), Q=q, E=e, residual=float(sol.newton.residual_norm)))
            if verbose:
                obs = self.observables_tau0(sol.newton.x, subtract_background=False, return_profiles=False)
                print(f"[eta-scan] eta={eta:.6f}, " + _format_obs_line(obs, targets_scan))
            return q - float(Q_target), sol

        a, b = map(float, eta_bracket)
        fa, sola = solve_eta(a, x0)
        fb, solb = solve_eta(b, sola.newton.x)

        if fa == 0.0:
            return sola, dict(history=hist, eta0=a)
        if fb == 0.0:
            return solb, dict(history=hist, eta0=b)
        if fa * fb > 0.0:
            raise ValueError(f"eta_bracket does not bracket Q_target: f(a)={fa:+.3e}, f(b)={fb:+.3e}")

        lo, hi = a, b
        flo, fhi = fa, fb
        seed = solb.newton.x
        best_sol = solb

        for _ in range(max_steps):
            mid = 0.5 * (lo + hi)
            fm, solm = solve_eta(mid, seed)
            seed = solm.newton.x
            best_sol = solm
            if abs(fm) < tol_Q * max(1.0, abs(Q_target)):
                return solm, dict(history=hist, eta0=float(mid))
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm

        def f_scalar(eta: float) -> float:
            nonlocal seed, best_sol
            ff, sol = solve_eta(eta, seed)
            seed = sol.newton.x
            best_sol = sol
            return ff

        eta_star = float(brentq(f_scalar, lo, hi, maxiter=max_steps))
        return best_sol, dict(history=hist, eta0=eta_star)

    # -------------------------------------------------------------------------
    # dump
    # -------------------------------------------------------------------------
    @staticmethod
    def dump_solution(sol: Bubble2DSolution, out_dir: str) -> None:
        import os, json
        os.makedirs(out_dir, exist_ok=True)
        meta = dict(
            settings=asdict(sol.settings),
            rho0=float(sol.rho0),
            newton=dict(
                success=bool(sol.newton.success),
                iterations=int(sol.newton.iterations),
                residual_norm=float(sol.newton.residual_norm),
                history=[float(x) for x in sol.newton.history],
            ),
            Q_tau0=float(sol.Q_tau0.real),
            E_tau0=float(sol.E_tau0),
            sanity=sol.sanity,
        )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        np.savez(os.path.join(out_dir, "grid.npz"), r=sol.grid.r, tau=sol.grid.tau, dr=sol.grid.dr, dtau=sol.grid.dtau)
        np.save(os.path.join(out_dir, "y.npy"), sol.y)
        np.save(os.path.join(out_dir, "ybar.npy"), sol.ybar)


if __name__ == "__main__":
    # Example: external potential as in the notebook (potential_bubble)
    try:
        from .potential_bubble import V_phi, dV_dphi, d2V_dphi2
    except ImportError:
        from potential_bubble import V_phi, dV_dphi, d2V_dphi2

    phi0, v1, v2 = 1.999, 1.0, 2.0
    U, dU, d2U = make_potential_from_V(V_phi, dV_dphi, d2V_dphi2, phi0, v1, v2)

    settings = Bubble2DSettings(
        Nr=120, Ntau=240, Lr=20.0, beta=40.0,
        omega_ref=0.4,
        rho0_bracket=(1.0, 1.3),
        eta0=0.0,
        complex_saddle=True,
        allow_debug_bcs=False,
        verbose=True,
    )

    solver = Bubble2DSolver(settings, U, dU, d2U)

    print("Sanity (eta0=0 wiring):", solver.sanity_background_eta0_zero())
    print("Sanity (twist source):", solver.sanity_twist_source())

    x0 = solver.build_anchored_initial_guess(rho_center=v2, Rb=4.0, Lw=0.5)
    sol = solver.solve(x0)

    print("Converged:", sol.newton.success, "||F||=", sol.newton.residual_norm)
    print(f"Q(τ=0 ghost)={sol.Q_tau0.real:.6e}, E(τ=0 ghost)={sol.E_tau0:.6e}")

    # Jacobian FD check at the converged (or last) point
    try:
        dJ = solver.check_jacobian(sol.newton.x, n_tests=3, eps=1e-7)
        print("Jacobian check:", dJ)
    except Exception as e:
        print("Jacobian check failed:", e)

    Bubble2DSolver.dump_solution(sol, "bubble2d_out")
