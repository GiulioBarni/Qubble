# Bubble_finder/negative_modes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .rate_exponent import (
    compute_euclidean_action_half,
    compute_homogeneous_action,
    volume_from_grid,
    make_V_of_s_from_U,
    compute_suppression_exponent_bubble,
)
from .observables_2d import compute_charge_tau0_ghost_2d, compute_charge_2d


# =============================================================================
# Action / microcanonical functional (fixed Q)
# =============================================================================

def _get_omega_rho0_eta0(solver) -> Tuple[float, float, float]:
    omega = getattr(solver, "omega", None)
    if omega is None:
        omega = getattr(getattr(solver, "settings", object()), "omega", None)
    rho0 = getattr(solver, "rho0", None)
    if rho0 is None:
        rho0 = getattr(getattr(solver, "settings", object()), "rho0", None)

    eta0 = getattr(solver, "eta0", None)
    if eta0 is None:
        eta0 = getattr(getattr(solver, "settings", object()), "eta0", None)
    if eta0 is None:
        eta0 = getattr(getattr(solver, "settings", object()), "eta_0", None)

    omega = float(omega)
    rho0 = float(rho0)
    eta0 = float(eta0) if eta0 is not None else 0.0
    return omega, rho0, eta0


def _resolve_tau_index_for_Q(
    solver,
    tau_index_for_Q: Optional[int] = None,
    *,
    fallback_to_interior_if_boundary: bool = True,
) -> Tuple[int, bool, bool]:
    """
    Resolve tau index used for Q diagnostics/projection.
    Returns: (tau_index_used, is_boundary, used_fallback)
    """
    tau = np.asarray(solver.grid.tau, dtype=float).ravel()
    Nt = int(tau.size)
    if Nt <= 0:
        raise ValueError("solver.grid.tau is empty.")

    if tau_index_for_Q is None:
        idx = int(np.argmin(np.abs(tau)))
    else:
        idx = int(tau_index_for_Q)
        if idx < 0:
            idx = Nt + idx
        idx = max(0, min(Nt - 1, idx))

    is_boundary = bool(idx in (0, Nt - 1))
    used_fallback = False
    if is_boundary and fallback_to_interior_if_boundary and Nt >= 3:
        # Prefer the closest interior point to tau~0.
        interior = np.arange(1, Nt - 1, dtype=int)
        idx_int = int(interior[np.argmin(np.abs(tau[interior]))])
        if idx_int != idx:
            idx = idx_int
            used_fallback = True
            is_boundary = False
    return idx, is_boundary, used_fallback


def action_and_functional(
    solver,
    x: np.ndarray,
    *,
    subtract_background_charge: bool = False,
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
    fallback_to_interior_if_boundary: bool = True,
) -> Dict[str, float]:
    """
    Uses the same conventions already implemented in rate_exponent.py + observables_2d.py.

    Returns:
      S_half, S_full (=2*S_half),
      S_hom,
      Q (ghost tau=0),
      F_bounce = S_full - S_hom + eta0*Q
    """
    y, ybar = solver.unpack(x)
    omega, rho0, eta0 = _get_omega_rho0_eta0(solver)

    V_of_s = make_V_of_s_from_U(solver.U)

    S_half = float(compute_euclidean_action_half(y, ybar, solver.grid, omega, eta0, rho0, V_of_s))
    S_full = float(2.0 * S_half)

    V_ball = float(volume_from_grid(solver.grid))
    beta = float(getattr(solver.grid, "beta", getattr(getattr(solver, "settings", object()), "beta", None)))
    if beta is None:
        tau = np.asarray(solver.grid.tau, dtype=float).ravel()
        beta = float(2.0 * abs(tau[0])) if tau.size else 0.0

    S_hom = float(compute_homogeneous_action(beta, V_ball, omega, rho0, V_of_s))

    tau_idx, tau_idx_boundary, tau_idx_fallback = _resolve_tau_index_for_Q(
        solver,
        tau_index_for_Q=tau_index_for_Q,
        fallback_to_interior_if_boundary=fallback_to_interior_if_boundary,
    )
    if q_use_ghost is None:
        # Ghost formula is tied to boundary tau=0 convention.
        q_use_ghost_eff = bool(tau_idx == 0)
    else:
        q_use_ghost_eff = bool(q_use_ghost)

    if q_use_ghost_eff:
        Q = float(compute_charge_tau0_ghost_2d(
            solver, y, ybar,
            subtract_background=subtract_background_charge,
            return_profile=False,
        ))
    else:
        Q = float(compute_charge_2d(
            solver, y, ybar,
            index_tau=int(tau_idx),
            subtract_background=subtract_background_charge,
            return_profile=False,
        ))

    F_bounce = float(compute_suppression_exponent_bubble(S_full, S_hom, eta0, Q))

    return dict(
        S_half=S_half,
        S_full=S_full,
        S_hom=S_hom,
        Q=Q,
        F_bounce=F_bounce,
        beta=beta,
        V_ball=V_ball,
        omega=omega,
        rho0=rho0,
        eta0=eta0,
        tau_index_for_Q=int(tau_idx),
        tau_index_boundary=bool(tau_idx_boundary),
        tau_index_fallback=bool(tau_idx_fallback),
        q_use_ghost=bool(q_use_ghost_eff),
    )


# =============================================================================
# Metric (mass matrix) in solver coordinates
# =============================================================================

def _mass_weights_site_from_action(solver) -> np.ndarray:
    r = np.asarray(solver.grid.r, dtype=float).ravel()

    dr = getattr(solver.grid, "dr", None)
    dt = getattr(solver.grid, "dtau", None)
    if dr is None or dt is None:
        rr = np.asarray(solver.grid.r, dtype=float).ravel()
        tt = np.asarray(solver.grid.tau, dtype=float).ravel()
        dr = float(np.mean(np.diff(rr))) if rr.size > 1 else 1.0
        dt = float(abs(np.mean(np.diff(tt)))) if tt.size > 1 else 1.0
    dr = float(dr)
    dt = float(dt)

    w_r = 4.0 * np.pi * (r ** 2) * dr * dt
    w = np.repeat(w_r[:, None], int(solver.Nt), axis=1).reshape(-1)
    return w.astype(float)


def _detect_layout(solver, x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x)
    Nsite = int(solver.Nr) * int(solver.Nt)
    if x.size == 2 * Nsite:
        return dict(layout="2field", nblock=2)
    if x.size == 4 * Nsite:
        return dict(layout="ri", nblock=4)
    raise ValueError(f"Unrecognized x size={x.size}; expected 2*Nsite or 4*Nsite.")


def build_mass_matrix_diag(
    solver,
    x: np.ndarray,
    *,
    y_metric: bool = True,
) -> np.ndarray:
    """
    Natural action measure is ∫ dτ 4π ∫ dr r² ...

    But solver uses y = r*(phi_rot - rho0) so δphi ~ δy/r.
    Therefore induced metric in y-coordinates divides by r², removing large-r dominance.

    y_metric=True is the *right* choice for spectra in solver coordinates.
    """
    info = _detect_layout(solver, x)
    w_site = _mass_weights_site_from_action(solver)

    if y_metric:
        r = np.asarray(solver.grid.r, dtype=float).ravel()
        r2 = (r ** 2)[:, None]
        r2 = np.repeat(r2, int(solver.Nt), axis=1).reshape(-1)
        r2 = np.maximum(r2, 1e-30)
        w_site = w_site / r2

    if info["layout"] == "ri":
        return np.concatenate([w_site, w_site, w_site, w_site]).astype(float)
    return np.concatenate([w_site, w_site]).astype(float)


# =============================================================================
# Masking
# =============================================================================

def _site_mask(solver, cut_r: int = 1, cut_tau: int = 1) -> np.ndarray:
    Nr, Nt = int(solver.Nr), int(solver.Nt)
    mr = np.ones(Nr, dtype=bool)
    mt = np.ones(Nt, dtype=bool)
    if cut_r > 0:
        mr[:cut_r] = False
        mr[-cut_r:] = False
    if cut_tau > 0:
        mt[:cut_tau] = False
        mt[-cut_tau:] = False
    return (mr[:, None] & mt[None, :]).reshape(-1)


def _lift_site_mask_to_x(solver, x: np.ndarray, site_mask: np.ndarray) -> np.ndarray:
    info = _detect_layout(solver, x)
    if info["layout"] == "ri":
        return np.concatenate([site_mask, site_mask, site_mask, site_mask])
    return np.concatenate([site_mask, site_mask])


def _tau0_site_mask(
    solver,
    cut_r: int = 1,
    *,
    tau_index_for_Q: Optional[int] = None,
    fallback_to_interior_if_boundary: bool = True,
) -> np.ndarray:
    """
    Site mask that keeps only tau~0 column and interior r points.
    """
    Nr, Nt = int(solver.Nr), int(solver.Nt)
    tau = np.asarray(solver.grid.tau, dtype=float).ravel()
    i_tau0, _is_bnd, _used_fb = _resolve_tau_index_for_Q(
        solver,
        tau_index_for_Q=tau_index_for_Q,
        fallback_to_interior_if_boundary=fallback_to_interior_if_boundary,
    )
    mask = np.zeros((Nr, Nt), dtype=bool)
    j0 = int(max(0, cut_r))
    j1 = int(min(Nr, Nr - max(0, cut_r)))
    if j1 > j0:
        mask[j0:j1, i_tau0] = True
    return mask.reshape(-1)


def _Q_of_x(
    solver,
    x: np.ndarray,
    *,
    subtract_background_charge: bool = False,
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
    fallback_to_interior_if_boundary: bool = True,
) -> float:
    xx = np.asarray(x, dtype=float).ravel()
    y, ybar = solver.unpack(xx)
    tau_idx, _is_bnd, _used_fb = _resolve_tau_index_for_Q(
        solver,
        tau_index_for_Q=tau_index_for_Q,
        fallback_to_interior_if_boundary=fallback_to_interior_if_boundary,
    )
    if q_use_ghost is None:
        q_use_ghost_eff = bool(tau_idx == 0)
    else:
        q_use_ghost_eff = bool(q_use_ghost)

    if q_use_ghost_eff:
        return float(
            compute_charge_tau0_ghost_2d(
                solver, y, ybar,
                subtract_background=subtract_background_charge,
                return_profile=False,
            )
        )
    return float(
        compute_charge_2d(
            solver, y, ybar,
            index_tau=int(tau_idx),
            subtract_background=subtract_background_charge,
            return_profile=False,
        )
    )


def grad_Q_tau0_only(
    solver,
    x: np.ndarray,
    *,
    idx_full: Optional[np.ndarray] = None,
    eps_fd: float = 1e-6,
    cut_r: int = 1,
    subtract_background_charge: bool = False,
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
    fallback_to_interior_if_boundary: bool = True,
) -> np.ndarray:
    """
    Sparse numerical gradient of Q in solver coordinates, perturbing only tau~0 DOFs.
    """
    xx = np.asarray(x, dtype=float).ravel()
    if idx_full is None:
        site_mask = _tau0_site_mask(
            solver,
            cut_r=cut_r,
            tau_index_for_Q=tau_index_for_Q,
            fallback_to_interior_if_boundary=fallback_to_interior_if_boundary,
        )
        x_mask = _lift_site_mask_to_x(solver, xx, site_mask)
        idx = np.where(x_mask)[0]
    else:
        idx = np.asarray(idx_full, dtype=int).ravel()
    g = np.zeros_like(xx)
    for ii in idx:
        xp = xx.copy()
        xm = xx.copy()
        xp[ii] += eps_fd
        xm[ii] -= eps_fd
        Qp = _Q_of_x(
            solver,
            xp,
            subtract_background_charge=subtract_background_charge,
            tau_index_for_Q=tau_index_for_Q,
            q_use_ghost=q_use_ghost,
            fallback_to_interior_if_boundary=fallback_to_interior_if_boundary,
        )
        Qm = _Q_of_x(
            solver,
            xm,
            subtract_background_charge=subtract_background_charge,
            tau_index_for_Q=tau_index_for_Q,
            q_use_ghost=q_use_ghost,
            fallback_to_interior_if_boundary=fallback_to_interior_if_boundary,
        )
        g[ii] = (Qp - Qm) / (2.0 * eps_fd)
    return g


def project_to_fixed_Q(
    v: np.ndarray,
    gradQ: np.ndarray,
    Mdiag: np.ndarray,
    *,
    tol: float = 1e-30,
) -> np.ndarray:
    """
    Project v to subspace deltaQ = gradQ^T v = 0 in M-metric:
      p = M^{-1} gradQ
      v_perp = v - alpha p, alpha = (gradQ^T v)/(gradQ^T p)
    """
    vv = np.asarray(v, dtype=float).copy()
    gQ = np.asarray(gradQ, dtype=float).ravel()
    MM = np.asarray(Mdiag, dtype=float).ravel()
    if vv.size != gQ.size or vv.size != MM.size:
        raise ValueError("v, gradQ and Mdiag must have the same size.")
    p = gQ / np.maximum(MM, tol)
    denom = float(np.dot(gQ, p))
    if abs(denom) < tol:
        return vv
    alpha = float(np.dot(gQ, vv) / denom)
    return vv - alpha * p


# =============================================================================
# Sign inference: does Js correspond to +d²S or -d²S?
# =============================================================================

@dataclass
class HessianSignInference:
    sign: float
    ratios: np.ndarray
    fd_second: np.ndarray
    quad_Js: np.ndarray
    meta: Dict[str, Any]


def _second_dir_derivative_S(
    solver,
    x: np.ndarray,
    v: np.ndarray,
    eps_fd: float,
    *,
    functional_key: str = "F_bounce",
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
) -> float:
    """
    d²S/dε²|0 ≈ [S(x+εv) - 2S(x) + S(x-εv)] / ε²
    Using S_full from action_and_functional.
    """
    A0 = action_and_functional(
        solver,
        x,
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
    )
    Ap = action_and_functional(
        solver,
        x + eps_fd * v,
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
    )
    Am = action_and_functional(
        solver,
        x - eps_fd * v,
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
    )
    key = str(functional_key)
    if key not in A0:
        key = "S_full"
    return float((Ap[key] - 2.0 * A0[key] + Am[key]) / (eps_fd ** 2))


def infer_hessian_sign(
    solver,
    x: np.ndarray,
    Js,
    Mdiag: np.ndarray,
    idx: np.ndarray,
    *,
    n_tests: int = 4,
    eps_fd: float = 1e-5,
    seed: int = 0,
    proj_data: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
    sign_functional_key: str = "F_bounce",
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
) -> HessianSignInference:
    """
    Determine whether the symmetric Jacobian Js corresponds to +d²S or -d²S.

    We compare, for random directions v (restricted to idx):
      fd:  d²S/dε²
      q :  v^T Js v
    If fd and q have opposite sign consistently -> take H = -Js.
    """
    rng = np.random.default_rng(seed)
    n = idx.size

    # Dense representation for qform
    if hasattr(Js, "tocsc"):
        Js_loc = Js.tocsr()
    else:
        Js_loc = np.asarray(Js, dtype=float)

    fd_list = []
    q_list = []
    ratio_list = []

    # Build full-size v for FD (needs full x shape)
    x = np.asarray(x, dtype=float).ravel()
    v_full = np.zeros_like(x)

    for _ in range(int(n_tests)):
        u = rng.normal(size=n).astype(float)
        # normalize in M metric: u^T M u = 1
        Mu = Mdiag[idx] * u
        normM = float(np.sqrt(np.dot(u, Mu)))
        if normM < 1e-30:
            continue
        u /= normM

        if proj_data is not None:
            gq_red, p_red, den_red = proj_data
            if abs(den_red) > 1e-30:
                alpha = float(np.dot(gq_red, u) / den_red)
                u -= alpha * p_red

        v_full[:] = 0.0
        v_full[idx] = u

        fd = _second_dir_derivative_S(
            solver,
            x,
            v_full,
            eps_fd=eps_fd,
            functional_key=sign_functional_key,
            tau_index_for_Q=tau_index_for_Q,
            q_use_ghost=q_use_ghost,
        )

        # q = u^T Js u (restricted)
        if hasattr(Js_loc, "dot"):
            Ju = Js_loc[idx][:, idx].dot(u)
        else:
            Ju = Js_loc[np.ix_(idx, idx)] @ u
        if proj_data is not None:
            gq_red, p_red, den_red = proj_data
            if abs(den_red) > 1e-30:
                alpha = float(np.dot(gq_red, Ju) / den_red)
                Ju = Ju - alpha * p_red
        q = float(np.dot(u, Ju))

        fd_list.append(fd)
        q_list.append(q)
        ratio_list.append(fd / (q + 1e-300))

    fd_arr = np.asarray(fd_list, dtype=float)
    q_arr = np.asarray(q_list, dtype=float)
    ratios = np.asarray(ratio_list, dtype=float)

    # Decide sign: if ratios are mostly negative -> need H = -Js
    # (because fd ~ v^T H v and q ~ v^T Js v)
    frac_neg = float(np.mean(ratios < 0.0)) if ratios.size else 0.0
    sign = -1.0 if frac_neg > 0.75 else +1.0

    meta = dict(
        n_tests=int(n_tests),
        eps_fd=float(eps_fd),
        frac_neg_ratio=frac_neg,
        sign_functional_key=str(sign_functional_key),
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
    )
    return HessianSignInference(sign=sign, ratios=ratios, fd_second=fd_arr, quad_Js=q_arr, meta=meta)


# =============================================================================
# Negative modes of the action Hessian (generalized eigenproblem)
# =============================================================================

@dataclass
class NegativeModeResult:
    eigvals: np.ndarray
    eigvecs: np.ndarray
    n_negative: int
    eps: float
    meta: Dict[str, Any]
    sign_inference: Optional[HessianSignInference]


def microcanonical_stationarity_check(
    solver,
    x: np.ndarray,
    *,
    idx: np.ndarray,
    Mdiag_full: np.ndarray,
    eps_fd: float = 1e-6,
    q_eps_fd: float = 1e-6,
    q_cut_r: int = 1,
    subtract_background_charge: bool = False,
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Check microcanonical stationarity:
      - generic direction v: dS, dQ, omega_eff ~ dS/dQ
      - projected direction v_perp (deltaQ=0): dS_perp should be ~0
    """
    xx = np.asarray(x, dtype=float).ravel()
    ii = np.asarray(idx, dtype=int).ravel()
    MM = np.asarray(Mdiag_full, dtype=float).ravel()
    Mred = MM[ii]

    rng = np.random.default_rng(12345)
    v_red = rng.normal(size=ii.size).astype(float)
    nM2 = float(np.dot(v_red, Mred * v_red))
    if nM2 <= 0.0 or (not np.isfinite(nM2)):
        raise ValueError("Invalid M-norm for stationarity check direction.")
    v_red /= np.sqrt(nM2)

    v_full = np.zeros_like(xx)
    v_full[ii] = v_red

    def _S_of(z):
        return float(action_and_functional(
            solver,
            z,
            subtract_background_charge=False,
            tau_index_for_Q=tau_index_for_Q,
            q_use_ghost=q_use_ghost,
        )["S_full"])

    def _Q_of(z):
        return float(_Q_of_x(
            solver,
            z,
            subtract_background_charge=subtract_background_charge,
            tau_index_for_Q=tau_index_for_Q,
            q_use_ghost=q_use_ghost,
        ))

    dS = float((_S_of(xx + eps_fd * v_full) - _S_of(xx - eps_fd * v_full)) / (2.0 * eps_fd))
    dQ = float((_Q_of(xx + eps_fd * v_full) - _Q_of(xx - eps_fd * v_full)) / (2.0 * eps_fd))
    omega_eff = float(dS / dQ) if abs(dQ) > 1e-30 else np.nan

    gQ_full = grad_Q_tau0_only(
        solver,
        xx,
        idx_full=None,
        eps_fd=q_eps_fd,
        cut_r=q_cut_r,
        subtract_background_charge=subtract_background_charge,
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
    )
    gQ_red = np.asarray(gQ_full[ii], dtype=float).ravel()
    p_red = gQ_red / np.maximum(Mred, 1e-30)
    den = float(np.dot(gQ_red, p_red))
    if abs(den) <= 1e-30:
        dS_perp = np.nan
    else:
        alpha = float(np.dot(gQ_red, v_red) / den)
        v_perp_red = v_red - alpha * p_red
        nM2p = float(np.dot(v_perp_red, Mred * v_perp_red))
        if nM2p > 1e-30:
            v_perp_red = v_perp_red / np.sqrt(nM2p)
        v_perp_full = np.zeros_like(xx)
        v_perp_full[ii] = v_perp_red
        dS_perp = float((_S_of(xx + eps_fd * v_perp_full) - _S_of(xx - eps_fd * v_perp_full)) / (2.0 * eps_fd))

    return dict(dS_dir=dS, dQ_dir=dQ, omega_eff=omega_eff, dS_perp=dS_perp)


def _hessian_vec_fd_residual(
    solver,
    x: np.ndarray,
    v: np.ndarray,
    *,
    eps_fd: float = 1e-6,
    H_sign: float = -1.0,
    R0: Optional[np.ndarray] = None,
    fd_mode: str = "forward",
) -> np.ndarray:
    """
    Hessian-vector product from finite differences of residual.

    Forward (default, cheaper):
      Hv ~= H_sign * [R(x+eps v) - R0] / eps, with R0 = R(x).

    Central (optional):
      Hv ~= H_sign * [R(x+eps v) - R(x-eps v)] / (2 eps).
    """
    xx = np.asarray(x, dtype=float).ravel()
    vv = np.asarray(v, dtype=float).ravel()
    mode = str(fd_mode).lower().strip()
    if mode == "central":
        Rp = np.asarray(solver.residual(xx + eps_fd * vv), dtype=float).ravel()
        Rm = np.asarray(solver.residual(xx - eps_fd * vv), dtype=float).ravel()
        return float(H_sign) * (Rp - Rm) / (2.0 * eps_fd)

    Rp = np.asarray(solver.residual(xx + eps_fd * vv), dtype=float).ravel()
    if R0 is None:
        R0 = np.asarray(solver.residual(xx), dtype=float).ravel()
    else:
        R0 = np.asarray(R0, dtype=float).ravel()
    return float(H_sign) * (Rp - R0) / eps_fd


def build_action_hessian_operator(
    solver,
    x: np.ndarray,
    idx: np.ndarray,
    Mdiag_full: np.ndarray,
    *,
    use_analytic_jacobian: bool = True,
    force_symmetric: bool = True,
    infer_sign: bool = True,
    H_sign: float = -1.0,
    sign_tests: int = 4,
    sign_eps_fd: float = 1e-5,
    fixed_Q: bool = False,
    q_eps_fd: float = 1e-6,
    q_cut_r: int = 1,
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
    subtract_background_charge: bool = False,
    fd_fallback: bool = True,
    gradQ_full_precomputed: Optional[np.ndarray] = None,
    verbose: bool = False,
    sign_functional_key: str = "F_bounce",
) -> Dict[str, Any]:
    """
    Source-of-truth builder for the reduced Hessian ingredients.
    Returns reduced metric, optional projector data, sign/sign-inference, and Js.
    """
    x = np.asarray(x, dtype=float).ravel()
    idx = np.asarray(idx, dtype=int).ravel()
    Mdiag_full = np.asarray(Mdiag_full, dtype=float).ravel()
    Mdiag_red = np.asarray(Mdiag_full[idx], dtype=float).ravel()
    inv_sqrt_M = 1.0 / np.sqrt(np.maximum(Mdiag_red, 1e-30))
    tau_idx_used, tau_idx_boundary, tau_idx_fallback = _resolve_tau_index_for_Q(
        solver,
        tau_index_for_Q=tau_index_for_Q,
        fallback_to_interior_if_boundary=True,
    )
    q_use_ghost_eff = bool(tau_idx_used == 0) if q_use_ghost is None else bool(q_use_ghost)

    fixed_Q_requested = bool(fixed_Q)
    fixed_Q_active = bool(fixed_Q_requested)
    gradQ_full = None
    gradQ_red = None
    p_red = None
    den_red = np.nan
    fixed_Q_skip_reason = None
    if fixed_Q_active:
        # If masking removes all Q-slice DOF, projection degenerates by construction.
        site_q_mask = _tau0_site_mask(
            solver,
            cut_r=q_cut_r,
            tau_index_for_Q=tau_idx_used,
            fallback_to_interior_if_boundary=False,
        )
        x_q_mask = _lift_site_mask_to_x(solver, x, site_q_mask)
        idx_q = np.where(x_q_mask)[0]
        if np.intersect1d(idx_q, idx, assume_unique=False).size == 0:
            fixed_Q_active = False
            fixed_Q_skip_reason = "masked_out_q_slice"
            if verbose:
                print("[neg_modes] WARNING: fixed_Q skipped (q-slice removed by mask/cut).")

    if fixed_Q_active:
        if gradQ_full_precomputed is not None:
            gradQ_full = np.asarray(gradQ_full_precomputed, dtype=float).ravel()
        else:
            gradQ_full = grad_Q_tau0_only(
                solver,
                x,
                idx_full=None,
                eps_fd=q_eps_fd,
                cut_r=q_cut_r,
                subtract_background_charge=subtract_background_charge,
                tau_index_for_Q=tau_idx_used,
                q_use_ghost=q_use_ghost_eff,
                fallback_to_interior_if_boundary=True,
            )
        gradQ_red = np.asarray(gradQ_full[idx], dtype=float).ravel()
        p_red = gradQ_red / np.maximum(Mdiag_red, 1e-30)
        den_red = float(np.dot(gradQ_red, p_red))
        scale = max(1e-300, np.linalg.norm(gradQ_red) * np.linalg.norm(p_red))
        if abs(den_red) < (1e-30 * scale):
            fixed_Q_active = False
            fixed_Q_skip_reason = "den_too_small"
            if verbose:
                print("[neg_modes] WARNING: fixed_Q requested but projection disabled (den too small).")
        elif den_red <= 0.0:
            fixed_Q_active = False
            fixed_Q_skip_reason = "den_nonpositive"
            if verbose:
                print("[neg_modes] WARNING: fixed_Q requested but projection disabled (den <= 0).")

    def proj_red_inplace(v: np.ndarray) -> np.ndarray:
        if (not fixed_Q_active) or (gradQ_red is None) or (p_red is None):
            return v
        alpha = float(np.dot(gradQ_red, v) / den_red)
        v -= alpha * p_red
        return v

    Js = None
    used_analytic_jacobian = False
    jacobian_error = None
    asym_ratio = np.nan
    try:
        import scipy.sparse as sp  # type: ignore
        import scipy.sparse.linalg as spla  # type: ignore
        has_scipy = True
    except Exception:
        sp = None  # type: ignore
        spla = None  # type: ignore
        has_scipy = False

    if use_analytic_jacobian:
        try:
            J = solver.jacobian(x)
            if has_scipy and sp is not None and sp.isspmatrix(J):
                J = J.tocsr()
                Js = 0.5 * (J + J.T)
                try:
                    num = float(spla.norm(J - J.T))
                    denJ = float(spla.norm(J))
                    asym_ratio = num / max(denJ, 1e-300)
                except Exception:
                    asym_ratio = np.nan
            else:
                Jd = np.asarray(J.todense() if hasattr(J, "todense") else J, dtype=float)
                Js = 0.5 * (Jd + Jd.T)
                num = float(np.linalg.norm(Jd - Jd.T))
                denJ = float(np.linalg.norm(Jd))
                asym_ratio = num / max(denJ, 1e-300)
            _ = force_symmetric
            used_analytic_jacobian = True
        except Exception as e_j:
            jacobian_error = e_j
            if not fd_fallback:
                raise RuntimeError(f"jacobian() failed and fd_fallback=False: {e_j}")

    si: Optional[HessianSignInference] = None
    sign = float(H_sign)
    if infer_sign and used_analytic_jacobian and (Js is not None):
        proj_data = (gradQ_red, p_red, den_red) if fixed_Q_active and gradQ_red is not None and p_red is not None else None
        si = infer_hessian_sign(
            solver,
            x,
            Js,
            Mdiag_full,
            idx,
            n_tests=sign_tests,
            eps_fd=sign_eps_fd,
            seed=0,
            proj_data=proj_data,
            sign_functional_key=sign_functional_key,
            tau_index_for_Q=tau_idx_used,
            q_use_ghost=q_use_ghost_eff,
        )
        sign = float(si.sign)
    elif (not infer_sign) and use_analytic_jacobian:
        print(f"[neg_modes] WARNING: infer_sign disabled; using H_sign={sign:+.1f}")

    return dict(
        x=x,
        idx=idx,
        Mdiag_red=Mdiag_red,
        inv_sqrt_M=inv_sqrt_M,
        Js=Js,
        sign=sign,
        sign_inference=si,
        used_analytic_jacobian=bool(used_analytic_jacobian),
        jacobian_error=(str(jacobian_error) if jacobian_error is not None else None),
        asym_ratio=float(asym_ratio) if np.isfinite(asym_ratio) else np.nan,
        fixed_Q_requested=bool(fixed_Q_requested),
        fixed_Q_active=bool(fixed_Q_active),
        fixed_Q=bool(fixed_Q_active),
        gradQ_full=gradQ_full,
        gradQ_red=gradQ_red,
        p_red=p_red,
        den_red=(float(den_red) if np.isfinite(den_red) else np.nan),
        proj_red=proj_red_inplace,
        fixed_Q_skip_reason=fixed_Q_skip_reason,
        tau_index_for_Q=int(tau_idx_used),
        tau_index_boundary=bool(tau_idx_boundary),
        tau_index_fallback=bool(tau_idx_fallback),
        q_use_ghost=bool(q_use_ghost_eff),
    )


def lowest_eigs_action_hessian(
    solver,
    x: np.ndarray,
    *,
    k: int = 5,
    eps: float = 1e-12,
    cut_r: int = 1,
    cut_tau: int = 1,
    y_metric: bool = True,
    use_sparse: bool = True,
    H_sign: float = -1.0,
    infer_sign: bool = True,
    sign_tests: int = 4,
    sign_eps_fd: float = 1e-5,
    hv_eps_fd: float = 1e-6,
    fixed_Q: bool = False,
    q_eps_fd: float = 1e-6,
    q_cut_r: int = 1,
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
    subtract_background_charge: bool = False,
    use_analytic_jacobian: bool = True,
    force_symmetric: bool = True,
    use_shift_invert: bool = True,
    sigma: float = 0.0,
    fd_fallback: bool = True,
    fd_mode: str = "forward",
    verbose: bool = False,
    gradQ_full_precomputed: Optional[np.ndarray] = None,
    do_curvature_check: bool = False,
    curv_eps_fd: float = 1e-5,
    curv_full_unmasked: bool = False,
    sign_functional_key: str = "F_bounce",
    do_microcanonical_stationarity_check: bool = False,
) -> NegativeModeResult:
    """
    Compute lowest eigenvalues of the *action* Hessian around x, using:
      - Js = (J + J^T)/2 from solver.jacobian (J = d(residual)/dx)
      - infer global sign by matching v^T Js v to d²S/dε² from action finite differences
      - solve generalized eigenproblem: H v = λ M v with H = sign*Js

    This fixes the issue you observed: BANAL must not look maximally unstable.
    """
    x = np.asarray(x, dtype=float).ravel()

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        has_scipy = True
    except Exception:
        sp = None
        spla = None
        has_scipy = False

    # Geometry / metric / reduced space
    Mdiag_full = build_mass_matrix_diag(solver, x, y_metric=y_metric)
    site_mask = _site_mask(solver, cut_r=cut_r, cut_tau=cut_tau)
    x_mask = _lift_site_mask_to_x(solver, x, site_mask)
    idx = np.where(x_mask)[0]
    if idx.size < max(10, k + 2):
        raise ValueError(f"Too few DOFs after masking: {idx.size} (cut_r={cut_r}, cut_tau={cut_tau}).")
    if np.any(Mdiag_full[idx] <= 0):
        raise ValueError("Non-positive mass weights after masking.")
    op = build_action_hessian_operator(
        solver,
        x,
        idx,
        Mdiag_full,
        use_analytic_jacobian=use_analytic_jacobian,
        force_symmetric=force_symmetric,
        infer_sign=infer_sign,
        H_sign=H_sign,
        sign_tests=sign_tests,
        sign_eps_fd=sign_eps_fd,
        fixed_Q=fixed_Q,
        q_eps_fd=q_eps_fd,
        q_cut_r=q_cut_r,
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
        subtract_background_charge=subtract_background_charge,
        fd_fallback=fd_fallback,
        gradQ_full_precomputed=gradQ_full_precomputed,
        verbose=verbose,
        sign_functional_key=sign_functional_key,
    )
    Mdiag = np.asarray(op["Mdiag_red"], dtype=float).ravel()
    inv_sqrt_M = np.asarray(op["inv_sqrt_M"], dtype=float).ravel()
    fixed_Q_requested = bool(op["fixed_Q_requested"])
    fixed_Q_active = bool(op["fixed_Q_active"])
    gradQ_red = op["gradQ_red"]
    den_red = float(op["den_red"]) if np.isfinite(op["den_red"]) else np.nan
    proj_inplace = op["proj_red"]
    used_analytic_jacobian = bool(op["used_analytic_jacobian"])
    asym_ratio = float(op["asym_ratio"]) if np.isfinite(op["asym_ratio"]) else np.nan
    Js = op["Js"]
    jacobian_error = op["jacobian_error"]
    si = op["sign_inference"]
    sign = float(op["sign"])
    tau_idx_used = int(op.get("tau_index_for_Q", -1))
    tau_idx_boundary = bool(op.get("tau_index_boundary", False))
    tau_idx_fallback = bool(op.get("tau_index_fallback", False))
    q_use_ghost_eff = bool(op.get("q_use_ghost", False))
    fixed_Q_skip_reason = op.get("fixed_Q_skip_reason", None)

    n = int(idx.size)
    k_eff = max(1, min(int(k), max(1, n - 2)))
    sigma_used = None
    used_shift_invert_actual = False

    # Reusable buffers for matvec (avoid per-call allocations where possible).
    tmp_v = np.empty(n, dtype=float)
    tmp_out = np.empty(n, dtype=float)

    # Fast analytic branch
    if used_analytic_jacobian and Js is not None and has_scipy and use_sparse and sp is not None and sp.isspmatrix(Js):
        H_red = (sign * Js).tocsr()[idx][:, idx]
        D = sp.diags(inv_sqrt_M, 0, format="csr")

        if fixed_Q_active:
            def _Aproj(u_vec: np.ndarray) -> np.ndarray:
                uu = np.asarray(u_vec, dtype=float).ravel()
                np.multiply(inv_sqrt_M, uu, out=tmp_v)
                proj_inplace(tmp_v)
                w = np.asarray(H_red @ tmp_v, dtype=float).ravel()
                proj_inplace(w)
                np.multiply(inv_sqrt_M, w, out=tmp_out)
                return tmp_out.copy()
            Aobj = spla.LinearOperator((n, n), matvec=_Aproj, dtype=float)
        else:
            Aobj = D @ H_red @ D

        # For projected operator, shift-invert via LinearOperator can be fragile/slow.
        use_shift_invert_eff = bool(use_shift_invert and (not fixed_Q_active))
        if use_shift_invert_eff:
            sigma_eff = float(sigma)
            try:
                vals, vecs_u = spla.eigsh(Aobj, k=k_eff, sigma=sigma_eff, which="LM")
                used_shift_invert_actual = True
            except Exception:
                if abs(sigma_eff) < 1e-30:
                    sigma_eff_try = -1e-10
                    try:
                        vals, vecs_u = spla.eigsh(Aobj, k=k_eff, sigma=sigma_eff_try, which="LM")
                        used_shift_invert_actual = True
                        sigma_eff = sigma_eff_try
                    except Exception:
                        vals, vecs_u = spla.eigsh(Aobj, k=k_eff, which="SA")
                        used_shift_invert_actual = False
                else:
                    vals, vecs_u = spla.eigsh(Aobj, k=k_eff, which="SA")
                    used_shift_invert_actual = False
            sigma_used = sigma_eff if used_shift_invert_actual else None
        else:
            vals, vecs_u = spla.eigsh(Aobj, k=k_eff, which="SA")

        order = np.argsort(vals)
        vals = np.asarray(vals[order], dtype=float)
        vecs_u = np.asarray(vecs_u[:, order], dtype=float)
        vecs_v = inv_sqrt_M[:, None] * vecs_u

    else:
        # FD fallback branch
        R0 = np.asarray(solver.residual(x), dtype=float).ravel()

        def _A_fd(u_vec: np.ndarray) -> np.ndarray:
            uu = np.asarray(u_vec, dtype=float).ravel()
            np.multiply(inv_sqrt_M, uu, out=tmp_v)
            proj_inplace(tmp_v)

            v_full = np.zeros_like(x)
            v_full[idx] = tmp_v
            Hv_full = _hessian_vec_fd_residual(
                solver,
                x,
                v_full,
                eps_fd=hv_eps_fd,
                H_sign=sign,
                R0=R0,
                fd_mode=fd_mode,
            )
            w = np.asarray(Hv_full[idx], dtype=float).ravel()
            proj_inplace(w)
            np.multiply(inv_sqrt_M, w, out=tmp_out)
            return tmp_out.copy()

        if has_scipy and use_sparse:
            Aop = spla.LinearOperator((n, n), matvec=_A_fd, dtype=float)
            if use_shift_invert:
                sigma_eff = float(sigma)
                try:
                    vals, vecs_u = spla.eigsh(Aop, k=k_eff, sigma=sigma_eff, which="LM")
                    used_shift_invert_actual = True
                except Exception:
                    if abs(sigma_eff) < 1e-30:
                        sigma_eff_try = -1e-10
                        try:
                            vals, vecs_u = spla.eigsh(Aop, k=k_eff, sigma=sigma_eff_try, which="LM")
                            used_shift_invert_actual = True
                            sigma_eff = sigma_eff_try
                        except Exception:
                            vals, vecs_u = spla.eigsh(Aop, k=k_eff, which="SA")
                            used_shift_invert_actual = False
                    else:
                        vals, vecs_u = spla.eigsh(Aop, k=k_eff, which="SA")
                        used_shift_invert_actual = False
                sigma_used = sigma_eff if used_shift_invert_actual else None
            else:
                vals, vecs_u = spla.eigsh(Aop, k=k_eff, which="SA")
            order = np.argsort(vals)
            vals = np.asarray(vals[order], dtype=float)
            vecs_u = np.asarray(vecs_u[:, order], dtype=float)
            vecs_v = inv_sqrt_M[:, None] * vecs_u
        else:
            A_dense = np.zeros((n, n), dtype=float)
            eye = np.eye(n, dtype=float)
            for j in range(n):
                A_dense[:, j] = _A_fd(eye[:, j])
            vals_all, vecs_u_all = np.linalg.eigh(0.5 * (A_dense + A_dense.T))
            vals = np.asarray(vals_all[:k_eff], dtype=float)
            vecs_v = inv_sqrt_M[:, None] * vecs_u_all[:, :k_eff]

    nneg = int(np.sum(vals < -abs(float(eps))))
    meta = dict(
        cut_r=int(cut_r),
        cut_tau=int(cut_tau),
        y_metric=bool(y_metric),
        ndof_full=int(x.size),
        ndof_used=int(idx.size),
        sign=float(sign),
        infer_sign_used=bool(infer_sign),
        sign_inference=(si.meta if si is not None else None),
        hv_eps_fd=float(hv_eps_fd),
        fixed_Q_requested=bool(fixed_Q_requested),
        fixed_Q=bool(fixed_Q_active),
        fixed_Q_active=bool(fixed_Q_active),
        q_eps_fd=float(q_eps_fd),
        q_cut_r=int(q_cut_r),
        tau_index_for_Q=int(tau_idx_used),
        tau_index_boundary=bool(tau_idx_boundary),
        tau_index_fallback=bool(tau_idx_fallback),
        q_use_ghost=bool(q_use_ghost_eff),
        den=(float(den_red) if np.isfinite(den_red) else np.nan),
        proj_cost=("O(N) 1dot+1axpy" if fixed_Q_active else "none"),
        fixed_Q_skip_reason=fixed_Q_skip_reason,
        used_analytic_jacobian=bool(used_analytic_jacobian),
        used_shift_invert=bool(used_shift_invert_actual),
        sigma=(float(sigma_used) if sigma_used is not None else None),
        force_symmetric=bool(force_symmetric),
        asym_ratio=float(asym_ratio) if np.isfinite(asym_ratio) else np.nan,
        n_negative=int(nneg),
        fd_mode=str(fd_mode),
        jacobian_error=(str(jacobian_error) if jacobian_error is not None else None),
        sign_functional_key=str(sign_functional_key),
    )

    if do_curvature_check and vecs_v.size > 0:
        # Cheap embedded check, no full-size eigensolve/recompute.
        v_full = np.zeros_like(x)
        v_full[idx] = np.asarray(vecs_v[:, 0], dtype=float).ravel()
        cc = curvature_check_along_mode(
            solver,
            x,
            v_full,
            eps_fd=curv_eps_fd,
            H_sign=sign,
            y_metric=y_metric,
            fixed_Q=fixed_Q_active,
            q_eps_fd=q_eps_fd,
            q_cut_r=q_cut_r,
            subtract_background_charge=subtract_background_charge,
            quad_method="same_as_spectrum",
            idx=(None if bool(curv_full_unmasked) else idx),
            gradQ_full_precomputed=op.get("gradQ_full", None),
            fd_functional_key=sign_functional_key,
            tau_index_for_Q=tau_idx_used,
            q_use_ghost=q_use_ghost_eff,
        )
        meta["curv_rel_mismatch"] = float(cc.rel_mismatch)
        meta["curv_d2S"] = float(cc.d2S_fd)
        meta["curv_vHv"] = float(cc.quad_form)
        meta["curv_full_unmasked"] = bool(curv_full_unmasked)
        meta["curv_quad_method"] = cc.meta.get("quad_method", "Js")

    if do_microcanonical_stationarity_check:
        try:
            mc = microcanonical_stationarity_check(
                solver,
                x,
                idx=idx,
                Mdiag_full=Mdiag_full,
                eps_fd=curv_eps_fd,
                q_eps_fd=q_eps_fd,
                q_cut_r=q_cut_r,
                subtract_background_charge=subtract_background_charge,
                tau_index_for_Q=tau_idx_used,
                q_use_ghost=q_use_ghost_eff,
            )
            meta["micro_dS"] = float(mc["dS_dir"])
            meta["micro_dQ"] = float(mc["dQ_dir"])
            meta["micro_dS_perp"] = float(mc["dS_perp"])
            meta["micro_omega_eff"] = float(mc["omega_eff"])
        except Exception as e_mc:
            meta["micro_stationarity_error"] = str(e_mc)

    if verbose:
        print(
            "[neg_modes] "
            f"analytic_J={meta['used_analytic_jacobian']} "
            f"shift_invert={meta['used_shift_invert']} "
            f"sigma={meta['sigma']} sign={meta['sign']:+.0f} "
            f"asym_ratio={meta['asym_ratio']:.3e}"
        )
        if fixed_Q_active:
            print(f"[neg_modes] fixed_Q projection precomputed (den={meta['den']:.6e}, ||gQ||={np.linalg.norm(gradQ_red):.6e})")

    return NegativeModeResult(
        eigvals=vals,
        eigvecs=vecs_v,
        n_negative=nneg,
        eps=float(eps),
        meta=meta,
        sign_inference=si,
    )


def print_negative_mode_report_action(
    res: NegativeModeResult,
    *,
    n_show: int = 5,
    label: str = "",
) -> None:
    tag = f" [{label}]" if label else ""
    print(f"Negative-mode report (action Hessian){tag}: n_neg(eig < -{res.eps:g}) = {res.n_negative}")
    print(f"  meta: {res.meta}")
    if "fixed_Q_requested" in res.meta or "fixed_Q_active" in res.meta:
        print(
            "  fixed-Q status: "
            f"requested={res.meta.get('fixed_Q_requested', None)} "
            f"active={res.meta.get('fixed_Q_active', res.meta.get('fixed_Q', None))} "
            f"den={res.meta.get('den', None)}"
        )
        if res.meta.get("fixed_Q_skip_reason", None) is not None:
            print(f"  fixed-Q skip reason: {res.meta.get('fixed_Q_skip_reason')}")
        if bool(res.meta.get("fixed_Q_requested", False)) and not bool(res.meta.get("fixed_Q_active", res.meta.get("fixed_Q", False))):
            print("  [WARNING] fixed_Q requested but inactive (projection disabled).")
    if "sign" in res.meta:
        print(f"  H_sign used: {res.meta['sign']:+.1f}")
    if "asym_ratio" in res.meta and np.isfinite(res.meta["asym_ratio"]) and float(res.meta["asym_ratio"]) > 1e-2:
        print(f"  [WARNING] Jacobian asymmetry is sizable: asym_ratio={res.meta['asym_ratio']:.3e} (using Js is essential)")
    if "tau_index_for_Q" in res.meta:
        print(
            "  Q-slice info: "
            f"tau_index_for_Q={res.meta.get('tau_index_for_Q')} "
            f"boundary={res.meta.get('tau_index_boundary')} "
            f"fallback={res.meta.get('tau_index_fallback')} "
            f"ghost={res.meta.get('q_use_ghost')}"
        )

    if res.sign_inference is not None and res.sign_inference.ratios.size:
        r = res.sign_inference.ratios
        print(f"  sign inference: chosen sign = {res.sign_inference.sign:+.0f} "
              f"(frac ratio<0 = {res.sign_inference.meta.get('frac_neg_ratio', np.nan):.2f})")
        print(f"    ratios fd/q (few): {np.array2string(r[:min(4, r.size)], precision=3)}")

    m = min(int(n_show), res.eigvals.size)
    if m > 0:
        print("  lowest eigenvalues (generalized H v = λ M v):")
        for i in range(m):
            print(f"    {i:3d}: {res.eigvals[i]: .6e}")
    if "micro_dS_perp" in res.meta:
        print(
            "  micro-stationarity: "
            f"dS={res.meta.get('micro_dS', np.nan):+.3e} "
            f"dQ={res.meta.get('micro_dQ', np.nan):+.3e} "
            f"omega_eff=dS/dQ={res.meta.get('micro_omega_eff', np.nan):+.3e} "
            f"dS_perp={res.meta.get('micro_dS_perp', np.nan):+.3e}"
        )


def _qform_same_as_spectrum(
    solver,
    x: np.ndarray,
    v_full: np.ndarray,
    *,
    H_sign: float,
    idx: Optional[np.ndarray],
    fixed_Q: bool,
    q_eps_fd: float,
    q_cut_r: int,
    subtract_background_charge: bool,
    gradQ_full_precomputed: Optional[np.ndarray] = None,
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
) -> float:
    """
    Compute v^T (H v) with H = H_sign * Js, using the same reduced-space
    projection convention used in eigenspectrum construction.
    """
    x = np.asarray(x, dtype=float).ravel()
    v_full = np.asarray(v_full, dtype=float).ravel()
    J = solver.jacobian(x)
    if hasattr(J, "tocsr"):
        Js = 0.5 * (J + J.T)
    else:
        Jd = np.asarray(J.todense() if hasattr(J, "todense") else J, dtype=float)
        Js = 0.5 * (Jd + Jd.T)

    Mdiag_full = build_mass_matrix_diag(solver, x, y_metric=True)
    if idx is None:
        idx_arr = np.arange(x.size, dtype=int)
    else:
        idx_arr = np.asarray(idx, dtype=int).ravel()

    v_red = np.asarray(v_full[idx_arr], dtype=float).copy()
    Mred = np.asarray(Mdiag_full[idx_arr], dtype=float).ravel()

    gQ = None
    p = None
    den = np.nan
    if fixed_Q:
        if gradQ_full_precomputed is not None:
            gQ_full = np.asarray(gradQ_full_precomputed, dtype=float).ravel()
        else:
            gQ_full = grad_Q_tau0_only(
                solver,
                x,
                idx_full=None,
                eps_fd=q_eps_fd,
                cut_r=q_cut_r,
                subtract_background_charge=subtract_background_charge,
                tau_index_for_Q=tau_index_for_Q,
                q_use_ghost=q_use_ghost,
            )
        gQ = np.asarray(gQ_full[idx_arr], dtype=float).ravel()
        p = gQ / np.maximum(Mred, 1e-30)
        den = float(np.dot(gQ, p))
        if abs(den) > 1e-30:
            alpha = float(np.dot(gQ, v_red) / den)
            v_red -= alpha * p

    if hasattr(Js, "tocsr"):
        Hred = (float(H_sign) * Js).tocsr()[idx_arr][:, idx_arr]
        Hv_red = np.asarray(Hred @ v_red, dtype=float).ravel()
    else:
        Hred = float(H_sign) * np.asarray(Js, dtype=float)[np.ix_(idx_arr, idx_arr)]
        Hv_red = np.asarray(Hred @ v_red, dtype=float).ravel()

    if fixed_Q and (gQ is not None) and (p is not None):
        if abs(den) > 1e-30:
            alpha = float(np.dot(gQ, Hv_red) / den)
            Hv_red -= alpha * p

    return float(np.dot(v_red, Hv_red))


# =============================================================================
# Second-derivative diagnostic along a chosen mode (useful!)
# =============================================================================

@dataclass
class CurvatureCheck:
    eps_fd: float
    d2S_fd: float
    quad_form: float
    rel_mismatch: float
    meta: Dict[str, Any]


def normalize_in_mass_metric(
    Mdiag: np.ndarray,
    v: np.ndarray,
    idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Normalize vector v such that v^T M v = 1.
    If idx is provided, normalize only on that subspace.
    Returns (v_normalized, original_norm_M).
    """
    vv = np.asarray(v, dtype=float).copy()
    MM = np.asarray(Mdiag, dtype=float).ravel()
    if vv.size != MM.size:
        raise ValueError("v and Mdiag must have the same size.")
    if idx is None:
        nM2 = float(np.dot(vv, MM * vv))
    else:
        ii = np.asarray(idx, dtype=int).ravel()
        nM2 = float(np.dot(vv[ii], MM[ii] * vv[ii]))
    nM = float(np.sqrt(max(nM2, 0.0)))
    if nM > 0.0:
        vv /= nM
    return vv, nM


def curvature_check_along_mode(
    solver,
    x: np.ndarray,
    v_full: np.ndarray,
    *,
    eps_fd: float = 1e-5,
    H_sign: float = -1.0,
    y_metric: bool = True,
    fixed_Q: bool = False,
    q_eps_fd: float = 1e-6,
    q_cut_r: int = 1,
    subtract_background_charge: bool = False,
    quad_method: str = "same_as_spectrum",
    idx: Optional[np.ndarray] = None,
    gradQ_full_precomputed: Optional[np.ndarray] = None,
    fd_functional_key: str = "F_bounce",
    tau_index_for_Q: Optional[int] = None,
    q_use_ghost: Optional[bool] = None,
) -> CurvatureCheck:
    """
    Compare:
      d²F/dε² from finite differences of selected functional
    with:
      v^T (H) v.

    By default (quad_method="same_as_spectrum"), the quadratic form uses the same
    operator used in spectrum analysis: H = H_sign * Js with Js=(J+J^T)/2.
    Set quad_method="FD_residual" only as a debugging cross-check.

    Note: v_full should be full-size vector in solver coords.
    """
    x = np.asarray(x, dtype=float).ravel()
    v = np.asarray(v_full, dtype=float).ravel()
    if v.size != x.size:
        raise ValueError("v_full must have same size as x.")

    # Normalize consistently in the action metric used for spectra.
    Mdiag_full = build_mass_matrix_diag(solver, x, y_metric=y_metric)
    v, nM = normalize_in_mass_metric(Mdiag_full, v, idx=None)

    gradQ_full = None
    if fixed_Q:
        gradQ_full = grad_Q_tau0_only(
            solver,
            x,
            eps_fd=q_eps_fd,
            cut_r=q_cut_r,
            subtract_background_charge=subtract_background_charge,
            tau_index_for_Q=tau_index_for_Q,
            q_use_ghost=q_use_ghost,
        )
        v = project_to_fixed_Q(v, gradQ_full, Mdiag_full)
        v, _ = normalize_in_mass_metric(Mdiag_full, v, idx=None)

    d2S = _second_dir_derivative_S(
        solver,
        x,
        v,
        eps_fd=eps_fd,
        functional_key=fd_functional_key,
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
    )

    method = str(quad_method).lower().strip()
    if method in {"same_as_spectrum", "js"}:
        q = _qform_same_as_spectrum(
            solver,
            x,
            v,
            H_sign=H_sign,
            idx=idx,
            fixed_Q=fixed_Q,
            q_eps_fd=q_eps_fd,
            q_cut_r=q_cut_r,
            subtract_background_charge=subtract_background_charge,
            gradQ_full_precomputed=(gradQ_full if gradQ_full is not None else gradQ_full_precomputed),
            tau_index_for_Q=tau_index_for_Q,
            q_use_ghost=q_use_ghost,
        )
        operator_consistent = True
    else:
        Hv = _hessian_vec_fd_residual(
            solver,
            x,
            v,
            eps_fd=eps_fd,
            H_sign=H_sign,
            fd_mode="forward",
        )
        if fixed_Q and gradQ_full is not None:
            Hv = project_to_fixed_Q(Hv, gradQ_full, Mdiag_full)
        q = float(np.dot(v, Hv))
        operator_consistent = False

    denom = max(1e-30, abs(d2S), abs(q))
    rel = float(abs(d2S - q) / denom)
    meta = dict(
        H_sign=float(H_sign),
        y_metric=bool(y_metric),
        v_norm_M_before=float(nM),
        fixed_Q=bool(fixed_Q),
        q_eps_fd=float(q_eps_fd),
        q_cut_r=int(q_cut_r),
        quad_method=("Js" if operator_consistent else "FD_residual"),
        operator_consistent_with_spectrum=bool(operator_consistent),
        fd_functional_key=str(fd_functional_key),
        tau_index_for_Q=tau_index_for_Q,
        q_use_ghost=q_use_ghost,
    )
    return CurvatureCheck(eps_fd=float(eps_fd), d2S_fd=float(d2S), quad_form=float(q), rel_mismatch=rel, meta=meta)


def curvature_check_from_masked_mode(
    solver,
    x: np.ndarray,
    res: NegativeModeResult,
    idx: np.ndarray,
    *,
    mode_index: int = 0,
    eps_fd: float = 1e-5,
    y_metric: bool = True,
    fixed_Q: bool = False,
    q_eps_fd: float = 1e-6,
    q_cut_r: int = 1,
    subtract_background_charge: bool = False,
    quad_method: str = "same_as_spectrum",
    gradQ_full_precomputed: Optional[np.ndarray] = None,
) -> Optional[CurvatureCheck]:
    """
    Cheap curvature check from a masked eigenvector by embedding into full-size vector.
    No full-size eigensolve, no GMRES recompute.
    """
    if res.eigvecs.size == 0:
        return None
    x_full = np.asarray(x, dtype=float).ravel()
    idx_arr = np.asarray(idx, dtype=int).ravel()
    if mode_index < 0 or mode_index >= res.eigvecs.shape[1]:
        return None
    v_full = np.zeros_like(x_full)
    v_full[idx_arr] = np.asarray(res.eigvecs[:, mode_index], dtype=float).ravel()
    return curvature_check_along_mode(
        solver,
        x_full,
        v_full,
        eps_fd=eps_fd,
        H_sign=float(res.meta.get("sign", 1.0)),
        y_metric=y_metric,
        fixed_Q=fixed_Q,
        q_eps_fd=q_eps_fd,
        q_cut_r=q_cut_r,
        subtract_background_charge=subtract_background_charge,
        quad_method=quad_method,
        idx=idx_arr,
        gradQ_full_precomputed=gradQ_full_precomputed,
    )


def print_curvature_check(cc: CurvatureCheck, *, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"Curvature check{tag}: rel_mismatch = {cc.rel_mismatch:.3e}")
    print(f"  d2F_fd     = {cc.d2S_fd:+.6e}")
    print(f"  v^T H v    = {cc.quad_form:+.6e}")
    print(f"  meta       = {cc.meta}")


def sanity_check_banal_stability(
    solver,
    sol_banal,
    k: int = 5,
    cut_r: int = 1,
    cut_tau: int = 1,
) -> Optional[NegativeModeResult]:
    """
    Non-crashing sanity check for BANAL stability.
    Prints a loud warning if too many negative modes are found.
    """
    try:
        res = lowest_eigs_action_hessian(
            solver,
            sol_banal.x,
            k=k,
            cut_r=cut_r,
            cut_tau=cut_tau,
            y_metric=True,
            use_sparse=True,
            H_sign=-1.0,
            infer_sign=True,
        )
        print_negative_mode_report_action(res, label="BANAL sanity", n_show=min(k, 10))
        if int(res.n_negative) >= 5:
            print(
                "[SANITY WARNING] BANAL has many negative modes "
                f"(n_negative={res.n_negative}). Sign/metric/masking may still be inconsistent."
            )
        return res
    except Exception as e:
        print(f"[SANITY WARNING] BANAL sanity check failed (non-fatal): {type(e).__name__}: {e}")
        return None


def quickcheck_negative_modes_on_background(solver):
    """
    Lightweight quickcheck for homogeneous background stability.
    Not executed on import.
    """
    try:
        from . import diagnostics_sanity  # type: ignore
        x_hom = diagnostics_sanity.build_background_zero_vec(solver)
    except Exception:
        x_hom = solver._zero_vec()

    res = lowest_eigs_action_hessian(
        solver,
        np.asarray(x_hom, dtype=float).ravel(),
        k=10,
        cut_r=1,
        cut_tau=1,
        y_metric=True,
        use_sparse=True,
        H_sign=-1.0,
        infer_sign=True,
        use_analytic_jacobian=True,
        use_shift_invert=True,
        sigma=0.0,
        fd_fallback=True,
        verbose=True,
    )
    return dict(eigvals=res.eigvals.copy(), meta=dict(res.meta))


__all__ = [
    "action_and_functional",
    "build_mass_matrix_diag",
    "build_action_hessian_operator",
    "lowest_eigs_action_hessian",
    "print_negative_mode_report_action",
    "microcanonical_stationarity_check",
    "normalize_in_mass_metric",
    "curvature_check_along_mode",
    "curvature_check_from_masked_mode",
    "print_curvature_check",
    "sanity_check_banal_stability",
    "quickcheck_negative_modes_on_background",
    "NegativeModeResult",
    "HessianSignInference",
    "CurvatureCheck",
]