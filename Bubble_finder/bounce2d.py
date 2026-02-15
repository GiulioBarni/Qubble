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
    """Solve homogeneous stationarity: dU/dρ(ρ0) = 2 ω^2 ρ0."""
    omega = float(omega)

    def f(rho: float) -> float:
        return float(dU(np.array([rho]))[0] - 2.0 * omega * omega * rho)

    a, b = map(float, bracket)
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(f"Bad bracket for rho0: f(a)={fa:+.3e}, f(b)={fb:+.3e}")
    return float(brentq(f, a, b, xtol=rtol, rtol=rtol, maxiter=300))


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
            settings.rho0 = solve_rho0_for_omega(self.omega, dU, settings.rho0_bracket)
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
        where W = dU/d(rho)/(2 rho) and rho = sqrt(u_pos + rho_eps).
        """
        u_pos, dupos_du = self._smooth_pos(u)
        rho = np.sqrt(u_pos + float(self.settings.rho_eps))
        dU = self.dU(rho)
        d2U = self.d2U(rho)
        W = dU / (2.0 * rho)
        Wp = (d2U * rho - dU) / (4.0 * rho**3)
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

                # Factor 2: tau-independent reduction must match 1D bounce ODE (bounce_1d.py).
                # 1D: phi'' + 2/r*phi' = dV/dphi - 2*omega^2*phi.
                # 2D uses W(u)=dU/(2*rho) with rho=sqrt(u); chain rule gives dV/dphi = 2*W*rho.
                # To get dV/dphi - 2*omega^2*phi we need coefficient 2*(omega^2 - W) on y_tot.
                A_coef = 2.0 * (self.omega * self.omega - W[j, i])
                Fy[j, i] = y_tt + y_rr + 2.0 * self.omega * y_t + A_coef * y_tot[j, i]
                Fyb[j, i] = yb_tt + yb_rr - 2.0 * self.omega * yb_t + A_coef * ybar_tot[j, i]

        if self.settings.complex_saddle:
            return self._pack_ri(Fy, Fyb)
        return self._pack_2field(Fy, Fyb)

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
        """rho(r, tau) = sqrt(u_pos + rho_eps) with u = Re(phi_rot * phibar_rot)."""
        phi_rot, phibar_rot = self.phi_rot(y, ybar)
        u = (phi_rot * phibar_rot).real
        u_pos, _ = self._smooth_pos(u)
        return np.sqrt(u_pos + float(self.settings.rho_eps))

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
                # Factor 2: match tau-independent reduction to 1D bounce (see residual)
                A_coef = 2.0 * (self.omega * self.omega - W[j, i])
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
                # Nonlinear term: 2*(ω^2 - W(u)) * y_tot  (factor 2 for 1D bounce match)
                # -----------------------------------------------------------------
                A = float(2.0 * (self.omega * self.omega - W[j, i]))
                Wu = float(2.0 * W_u[j, i])  # d(2*(ω²-W))/du = -2*dW/du

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

                # dA/dvar = - dW/dvar = - Wu * du/dvar
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
        """Charge Q on a tau slice; implementazione in observables_2d.compute_charge_2d."""
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
        subtract_background: bool = True,
    ) -> float | tuple[float, np.ndarray]:
        """Energy on a tau slice; implementazione in observables_2d.compute_energy_2d."""
        return observables_2d.compute_energy_2d(
            self, y, ybar,
            index_tau=index_tau,
            return_profile=return_profile,
            subtract_background=subtract_background,
        )

    def compute_charge_tau0_ghost(
        self,
        y: np.ndarray,
        ybar: np.ndarray,
        *,
        subtract_background: bool = True,
    ) -> float:
        """Charge at τ=0 by ghost reconstruction; implementazione in observables_2d.compute_charge_tau0_ghost_2d."""
        return observables_2d.compute_charge_tau0_ghost_2d(
            self, y, ybar, subtract_background=subtract_background
        )

    def compute_energy_tau0_ghost(self, y: np.ndarray, ybar: np.ndarray) -> float:
        """Energy E(τ=0) by ghost reconstruction; implementazione in observables_2d.compute_energy_tau0_ghost_2d."""
        return observables_2d.compute_energy_tau0_ghost_2d(self, y, ybar)

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
        mismatch = float(self.omega * self.omega - W0[0])
        return dict(residual_norm=nrm, residual_max_abs=mx, omega2_minus_W=mismatch, rho0=self.rho0)

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

        # bulk mismatch term from omega^2 - W(rho0^2)
        W0, *_ = self._W_Wp(np.array([self.rho0**2]))
        mismatch = float(self.omega * self.omega - W0[0])

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
            omega2_minus_W=mismatch,
            interior_reduced_norm=interior_norm,
            boundary_source_err_y=boundary_err_y,
            boundary_source_err_ybar=boundary_err_b,
        )

    def run_sanity_tests(self) -> Dict[str, Any]:
        d0 = self.sanity_background_eta0_zero()
        d1 = self.sanity_twist_source()
        return dict(background_eta0_zero=d0, twist_source=d1)

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
            f"Background: omega^2-W(ρ0)={d['omega2_minus_W']:.3e} with ρ0={d['rho0']:.3e}."
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
    def solve(self, x0: np.ndarray, verbose: Optional[bool] = None, store_iteration_history: bool = False) -> Bubble2DSolution:
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
        # Store initial state (iter 0) so the first curve in the plot is unambiguously the seed
        if store_iteration_history:
            y0, ybar0 = self.unpack(x0)
            phi0, phibar0 = self.phi(y0, ybar0)
            u0 = (phi0 * phibar0).real
            rho0_it = np.sqrt(np.maximum(u0 + float(self.settings.rho_eps), 0.0))
            iteration_history.append({"iter": 0, "res_norm": float(np.nan), "rho": rho0_it.copy()})

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
            if store_iteration_history:
                y_cb, ybar_cb = self.unpack(x)
                phi_cb, phibar_cb = self.phi(y_cb, ybar_cb)
                u = (phi_cb * phibar_cb).real
                rho_it = np.sqrt(np.maximum(u + float(self.settings.rho_eps), 0.0))
                iteration_history.append({"iter": it, "res_norm": float(nF), "rho": rho_it.copy()})
            if not do_verbose:
                return
            Fy, Fyb = self.unpack(F)
            F_abs = np.abs(Fy) + np.abs(Fyb)
            jmax, imax = np.unravel_index(int(np.argmax(F_abs)), F_abs.shape)
            rmax = float(self.grid.r[jmax])
            taumax = float(self.grid.tau[imax])
            Fmax = float(F_abs[jmax, imax])

            extra = ""
            try:
                y, ybar = self.unpack(x)
                Q0_slice = float(self.compute_charge(y, ybar, 0))
                E_tau0 = self.compute_energy_tau0_ghost(y, ybar)
                extra = f", Q(τ[0])={Q0_slice:+.6e}, E(τ=0)={E_tau0:.6e}"
            except Exception:
                pass

            print(f"[Newton-bubble] iter={it:02d}, ||F||={nF:.3e}{extra}")
            print(f"  max|F|={Fmax:.3e} at (r={rmax:.3f}, tau={taumax:.3f})")

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
        Q0_slice = float(self.compute_charge(y, ybar, 0))
        E0_slice = self.compute_energy(y, ybar, 0)
        E0_ghost = self.compute_energy_tau0_ghost(y, ybar)
        sanity = self.run_sanity_tests()

        if nr.success:
            print("")
            print("=" * 70)
            print("Newton converged — diagnostica risultati")
            print("=" * 70)
            print(f"  Convergenza: success, iterazioni = {nr.iterations}")
            print(f"  Residuo finale ||F|| = {nr.residual_norm:.6e}")
            print(f"  Carica Q(τ[0])      = {Q0_slice:+.6e}  (primo slice, tau=-Δτ/2)")
            print(f"  Energia E(τ=0) slice = {E0_slice:.6e}  (su griglia τ[0])")
            print(f"  Energia E(τ=0) ghost = {E0_ghost:.6e}  (ricostruita a τ=0 come Q-ball)")
            d0 = sanity.get("background_eta0_zero", {})
            if d0:
                print(f"  Sanity (η0=0): ||F||={d0.get('residual_norm', '—'):.3e}, ω²−W(ρ0)={d0.get('omega2_minus_W', '—'):.3e}")
            print("=" * 70)
            print("")

        return Bubble2DSolution(
            settings=self.settings,
            grid=self.grid,
            newton=nr,
            y=y,
            ybar=ybar,
            rho0=self.rho0,
            Q_tau0=complex(Q0_slice, 0.0),
            E_tau0=float(E0_ghost),
            sanity=sanity,
            iteration_history=iteration_history if store_iteration_history else None,
        )

    # -------------------------------------------------------------------------
    # external eta0 scan to match Q (bisection with warm-start)
    # -------------------------------------------------------------------------
    def scan_eta0_to_match_Q(self, Q_target: float, eta_bracket: Tuple[float, float], x0: np.ndarray, max_steps: int = 30):
        hist: List[Dict[str, Any]] = []

        def solve_eta(eta: float, seed: np.ndarray):
            self.settings.eta0 = float(eta)
            self.eta0 = float(eta)
            sol = self.solve(seed, verbose=False)
            q = float(sol.Q_tau0.real)
            hist.append(dict(eta0=float(eta), Q=q, residual=float(sol.newton.residual_norm)))
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
            if abs(fm) < 1e-10 * max(1.0, abs(Q_target)):
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
    print("Q(tau0)=", sol.Q_tau0.real, "E(tau0)=", sol.E_tau0)

    # Jacobian FD check at the converged (or last) point
    try:
        dJ = solver.check_jacobian(sol.newton.x, n_tests=3, eps=1e-7)
        print("Jacobian check:", dJ)
    except Exception as e:
        print("Jacobian check failed:", e)

    Bubble2DSolver.dump_solution(sol, "bubble2d_out")
