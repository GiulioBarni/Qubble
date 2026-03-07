# ansatz_bubble.py
# Bounce-oriented seed construction + residual/penalty-based seed selection + diagnostics
#
# This module intentionally merges the "ansatz_bubble" and "seed_search" roles:
# - build_seed_bubble: constructs a τ-localized, bubble-like 2D seed (plus optional negative-mode kick)
# - rank_seeds / search_best_seed_and_solve: scan parameters, score seeds, try Newton in rank order
# - full_diagnostics / plot_solution_and_diagnostics: plotting and complete checks
#
# Key bounce-finding corrections vs older sphaleron-oriented approach:
#   (1) DO NOT use a global τ-ramp background by default (it attracts sphaleron-like solutions).
#       If needed, you may enable a *gated* ramp only where the bubble is present.
#   (2) Center τ gate at τ≈0 by default (tau_gate_center_frac=1.0).
#   (3) Penalize banal convergence (distance to x_banal) and large deviations from homogeneous energy.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .bounce2d import Bubble2DSolver, Bubble2DSolution, NewtonConvergenceError
from . import observables_2d

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 3.0 * x * x - 2.0 * x * x * x


def tau_gate(
    tau: np.ndarray,
    T: float,
    frac: float,
    *,
    center_frac: float = 1.0,
) -> np.ndarray:
    """
    Smooth gate in τ on the half-box τ ∈ (-T, 0) where T = beta/2.
    - center_frac=1.0 -> centered near τ=0 (turning slice)
    - center_frac=0.0 -> centered near τ=-T (past boundary)
    - frac sets the width scale as a fraction of T.

    Output g(τ) in [0,1], localized near τ_center.
    """
    tau = np.asarray(tau, dtype=float).flatten()
    T = float(T)
    frac = float(frac)
    center_frac = float(center_frac)

    # tau runs from ~ -T to ~ 0 (half-step grid). We map center_frac in [0,1] to [-T, 0].
    tau_center = -T + center_frac * T
    dist = np.abs(tau - tau_center)
    w = max(1e-12, frac * T)

    g = np.zeros_like(tau, dtype=float)
    m = dist < w
    # plateau-ish with smooth fall to 0
    g[m] = 1.0 - smoothstep(dist[m] / w)
    return g


def radial_window(r: np.ndarray, rmax: float, frac: float) -> np.ndarray:
    """
    Smooth window in r: 1 in the bulk, turns off near r=rmax.
    Used ONLY for the kick (negative mode), not for the base bubble embedding.
    """
    r = np.asarray(r, dtype=float).flatten()
    rmax = float(rmax)
    frac = float(frac)
    w0 = (1.0 - frac) * rmax
    w = np.ones_like(r, dtype=float)
    m = r >= w0
    if np.any(m):
        x = (r[m] - w0) / max(1e-12, (rmax - w0))
        w[m] = 1.0 - smoothstep(x)
    return w


def _estimate_phi_hom_from_tail(phi: np.ndarray, phibar: np.ndarray, m: int = 8) -> float:
    """
    Robust tail estimate in solver units: average over last m points of
    sqrt(max(Re(phi*phibar), 0)), i.e. |phi|.
    """
    phi = np.asarray(phi)
    phibar = np.asarray(phibar)
    m = int(max(2, m))
    u = (phi[-m:] * phibar[-m:]).real
    u = np.maximum(u, 0.0)
    return float(np.mean(np.sqrt(u)))


def _estimate_rho_phys_hom_from_tail(phi: np.ndarray, phibar: np.ndarray, m: int = 8) -> float:
    """
    Robust tail estimate in physical-modulus units: average over last m points of
    sqrt(2*max(Re(phi*phibar), 0)), i.e. rho_phys.
    """
    phi = np.asarray(phi)
    phibar = np.asarray(phibar)
    m = int(max(2, m))
    u = (phi[-m:] * phibar[-m:]).real
    u = np.maximum(u, 0.0)
    return float(np.mean(np.sqrt(2.0 * u)))


def _normalize_profile_to_phi_units(
    phi: np.ndarray,
    phibar: np.ndarray,
    phi_ref: float,
    m: int = 8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize a 1D profile into solver units |phi|.
    Heuristic:
      - if tail is closer to phi_ref, profile is already in |phi| (scale=1)
      - else if tail is closer to sqrt(2)*phi_ref, profile is rho_phys (scale=1/sqrt(2))
    """
    phi = np.asarray(phi, dtype=float).copy()
    phibar = np.asarray(phibar, dtype=float).copy()
    if phi.size == 0:
        return phi, phibar, 1.0
    m = int(max(2, min(m, phi.size)))
    tail_mean = float(np.mean(phi[-m:]))
    d_phi = abs(tail_mean - float(phi_ref))
    d_rho = abs(tail_mean - float(np.sqrt(2.0) * phi_ref))
    scale = 1.0 if d_phi <= d_rho else 1.0 / np.sqrt(2.0)
    return phi * scale, phibar * scale, float(scale)


def _pack_real_vector(solver: Bubble2DSolver, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
    """
    Pack (Nr,Nt) arrays to solver vector (RI split if needed).
    """
    return solver.pack(y, ybar)


def _unpack_to_fields(solver: Bubble2DSolver, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return solver.unpack(x)


def _x_banal(solver: Bubble2DSolver) -> np.ndarray:
    """
    Canonical 'banal' state in solver variables: y=ybar=0.
    """
    y0 = np.zeros((solver.Nr, solver.Nt), dtype=complex)
    return solver.pack(y0, y0.copy())


def _xdist(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------
# Ansatz parameters
# ---------------------------------------------------------------------

@dataclass
class AnsatzParams:
    # kick amplitude along negative mode
    eps: float = 0.02
    # tau harmonic (for optional cosine modulation of the kick)
    k: int = 1
    phase: float = 0.0
    # amplitude of the embedded 1D bubble "base"
    amp: float = 1.0
    # τ gate width, fraction of T
    tau_gate_frac: float = 0.15
    # τ gate center: 1 => τ≈0 (turning slice). IMPORTANT for bounce.
    tau_gate_center_frac: float = 1.0
    # kick window near rmax
    r_window_frac: float = 0.15

    # NEW: control background ramp
    include_tau_ramp: bool = False
    ramp_gated: bool = True  # if include_tau_ramp, apply the same g_tau (not global ramp)

    # NEW: allow turning off the cosine modulation (pure gate)
    use_cosine: bool = True
    # NEW: add global twist-compatible background exp(±Δω τ)
    include_twist_background: bool = True
    # If True, gate the twist background by g_tau (default False: keep it global in τ)
    twist_background_gated: bool = False


# ---------------------------------------------------------------------
# Bubble profile hook
# ---------------------------------------------------------------------

def make_bubble_profile_1d_from_arrays(
    r_1d: np.ndarray,
    phi_1d: np.ndarray,
) -> Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Return bubble_profile_1d(omega_tilde, r_grid) from precomputed arrays.
    Ignores omega_tilde (caller handles which arrays to pass).
    """
    r_1d = np.asarray(r_1d, dtype=float)
    phi_1d = np.asarray(phi_1d, dtype=float)
    order = np.argsort(r_1d)
    r_1d = r_1d[order]
    phi_1d = phi_1d[order]

    def bubble_profile_1d(omega_tilde: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        phi = np.interp(r, r_1d, phi_1d, left=phi_1d[0], right=phi_1d[-1])
        return phi.copy(), phi.copy()

    return bubble_profile_1d


def make_bubble_profile_1d_from_solve_bounce(
    solve_bounce_fn: Callable,
    phi0: float,
    v1: float,
    v2: float,
    *,
    d: int = 3,
    extend_to: Optional[float] = None,
    rmax: Optional[float] = None,
    n_grid_points: int = 2000,
) -> Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    bubble_profile_1d(omega_tilde, r_grid) built by calling solve_bounce_fn(phi0,v1,v2,omega_tilde,...)
    with explicit control of d / extend_to / rmax / n_grid_points.
    """
    from scipy.interpolate import interp1d

    def bubble_profile_1d(omega_tilde: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        _rmax = float(rmax) if rmax is not None else float(max(np.max(r), 1.0))
        try:
            out = solve_bounce_fn(
                phi0, v1, v2, float(omega_tilde),
                d=int(d),
                rmax=_rmax,
                extend_to=extend_to,
                n_grid_points=int(n_grid_points),
            )
        except TypeError:
            out = solve_bounce_fn(phi0, v1, v2, float(omega_tilde))
        except Exception:
            out = (None, None, None, None, None)

        if out is None or len(out) < 2 or out[0] is None or out[1] is None:
            # Fallback homogeneous profile if 1D solve fails
            phi = np.full_like(r, float(v1), dtype=float)
            return phi.copy(), phi.copy()

        r_b, phi_b = np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)
        if r_b.size < 2 or phi_b.size < 2:
            tail = float(phi_b[-1]) if phi_b.size else float(v1)
            phi = np.full_like(r, tail, dtype=float)
            return phi.copy(), phi.copy()
        f = interp1d(r_b, phi_b, kind="linear", bounds_error=False, fill_value=(phi_b[0], phi_b[-1]))
        phi = np.asarray(f(r), dtype=float)
        return phi.copy(), phi.copy()

    bubble_profile_1d.profile_d = int(d)
    bubble_profile_1d.profile_extend_to = extend_to
    return bubble_profile_1d


def make_bubble_profile_1d_for_solver(
    solver: Bubble2DSolver,
    solve_bounce_fn: Callable,
    phi0: float,
    v1: float,
    v2: float,
    *,
    d: int,
    alpha_tau: float = 1.0,
    n_grid_points: int = 2000,
) -> Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Build a 1D profile callable whose integration domain covers the full 2D seed radial variable:
      S_max = sqrt(r_max^2 + (|tau_min|/alpha_tau)^2).
    """
    r = np.asarray(solver.grid.r, dtype=float).flatten()
    tau = np.asarray(solver.grid.tau, dtype=float).flatten()
    S_max = float(np.sqrt((float(r[-1]) ** 2) + (abs(float(tau[0])) / max(float(alpha_tau), 1e-12)) ** 2))
    prof = make_bubble_profile_1d_from_solve_bounce(
        solve_bounce_fn,
        phi0,
        v1,
        v2,
        d=int(d),
        extend_to=S_max,
        rmax=S_max,
        n_grid_points=int(n_grid_points),
    )
    prof.profile_d = int(d)
    prof.profile_S_max = S_max
    return prof


# ---------------------------------------------------------------------
# Negative mode: we use the solver Jacobian symmetric part eigenvector (robust default)
# If you want the older "Q-ball negative mode ODE", keep it externally and pass override.
# ---------------------------------------------------------------------

def get_negative_mode_from_jacobian_sym(
    solver: Bubble2DSolver,
    sol: Bubble2DSolution,
    n_eigs: int = 6,
) -> Tuple[np.ndarray, float]:
    """
    Compute an approximate negative mode v_neg from the symmetric part of the Jacobian at sol.
    Returns (v_neg, lambda_min). v_neg is a REAL vector in solver coordinates.
    """
    x = np.asarray(sol.x, dtype=float)
    J = solver.jacobian(x)
    # Symmetric part
    Js = (J + J.T) * 0.5

    # smallest eigenvalues of symmetric matrix
    try:
        import scipy.sparse.linalg as spla
        vals, vecs = spla.eigsh(Js, k=min(n_eigs, Js.shape[0] - 2), which="SA")
        idx = int(np.argmin(vals))
        lam = float(np.real(vals[idx]))
        v = np.asarray(vecs[:, idx], dtype=float)
    except Exception:
        # fallback dense (small problems only)
        A = np.asarray(Js.todense() if hasattr(Js, "todense") else Js, dtype=float)
        vals, vecs = np.linalg.eigh(A)
        lam = float(vals[0])
        v = np.asarray(vecs[:, 0], dtype=float)

    # normalize
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v, lam


# ---------------------------------------------------------------------
# Core seed builder (bounce-oriented)
# ---------------------------------------------------------------------

def build_seed_bubble(
    solver: Bubble2DSolver,
    *,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    params: AnsatzParams,
    # Optional overrides:
    rho_ref_override: Optional[float] = None,
    rho_tilde_override: Optional[float] = None,
    phi1d_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (phi1d, phibar1d) to skip bubble_profile_1d call
    neg_mode_override: Optional[Tuple[np.ndarray, float]] = None,  # (v_neg, lam)
    # Use reference solution to shape / center, optional:
    sol_ref: Optional[Bubble2DSolution] = None,
    # Optional direct override on the solver grid. For backward compatibility the argument
    # name is rho_override_2d, but the expected content is phi_amp_2d = |phi|(r,tau).
    # If provided, we bypass the 1D bubble embedding and negative-mode kick and embed
    # using y = r * (phi_amp - phi_ref).
    rho_override_2d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construct a τ-localized seed aimed at the bounce (turning slice near τ=0).

    Baseline structure:
      - No global τ ramp by default.
      - Embed the 1D bubble profile as δφ(r)=φ_1d(r) - rho_tilde (so it decays at large r).
      - Multiply by τ gate g(τ) localized near τ≈0.
      - Optionally add a kick along a negative mode (windowed near rmax and gated in τ).

    Returns:
      x0 packed for solver.solve and a meta dict.
    """
    r = np.asarray(solver.grid.r, dtype=float).flatten()
    tau = np.asarray(solver.grid.tau, dtype=float).flatten()
    T = float(solver.settings.beta) * 0.5
    rmax = float(r[-1])

    # Set the solver omega to omega_ref for stable rotated frame
    solver.omega = float(omega_ref)

    # Homogeneous reference in solver units |phi|
    phi_ref = float(rho_ref_override) if rho_ref_override is not None else float(solver.rho0)

    # Fast path: if rho_override_2d is given, build y,ybar directly from it and skip
    # all τ-gating, background ramp and negative-mode logic. This is used for custom
    # 2D seeds like the O(4)-symmetric rho(s) construction.
    if rho_override_2d is not None:
        phi_arr = np.asarray(rho_override_2d, dtype=float)
        if phi_arr.shape == (solver.Nt, solver.Nr):
            # convert to (Nr,Nt)
            phi_arr = phi_arr.T.copy()
        if phi_arr.shape != (solver.Nr, solver.Nt):
            raise ValueError(f"rho_override_2d has shape {phi_arr.shape}, expected (Nr,Nt)=({solver.Nr},{solver.Nt})")
        y_seed = (r[:, None] * (phi_arr - phi_ref)).astype(complex)
        ybar_seed = y_seed.copy()
        x0 = solver.pack(y_seed, ybar_seed)
        meta = dict(
            omega_ref=float(omega_ref),
            omega_tilde=float(omega_tilde),
            rho_ref=float(phi_ref),
            rho_tilde=None,
            profile_scale_to_phi=1.0,
            params=dict(**params.__dict__),
            neg_mode_lambda=None,
            note="Seed built from explicit rho_override_2d interpreted as |phi| (no τ-gate / neg-mode logic).",
        )
        return x0, meta

    # 1D profile at omega_tilde (use override if provided to avoid repeated calls)
    profile_scale_to_phi = 1.0
    if phi1d_override is not None:
        phi1d, phibar1d = phi1d_override[0], phi1d_override[1]
        phi1d = np.asarray(phi1d, dtype=float).copy()
        phibar1d = np.asarray(phibar1d, dtype=float).copy()
    else:
        phi1d, phibar1d = bubble_profile_1d(float(omega_tilde), r)
        phi1d, phibar1d, profile_scale_to_phi = _normalize_profile_to_phi_units(
            phi1d, phibar1d, phi_ref=phi_ref, m=8
        )

    # phi_tilde is the homogeneous background at tau=0 in solver units |phi|
    phi_tilde = float(rho_tilde_override) if rho_tilde_override is not None else _estimate_phi_hom_from_tail(phi1d, phibar1d, m=8)

    # τ gate (bounce wants localized around τ=0 by default)
    g_tau = tau_gate(tau, T, params.tau_gate_frac, center_frac=params.tau_gate_center_frac)  # (Nt,)

    # Base bubble embedding in y variables:
    # y = r*(phi_rot - phi_ref). For the localized bubble part we embed r*(phi1d - phi_tilde)
    # and add background shift separately if requested.
    y1d_loc = r * (phi1d - phi_tilde)
    ybar1d_loc = r * (phibar1d - phi_tilde)

    # Background terms:
    # 1) twist-compatible background in total fields:
    #      phi_bg(τ)    = phi_ref * exp(+Δω τ)
    #      phibar_bg(τ) = phi_ref * exp(-Δω τ)
    #    mapped to y-space as r*(phi_bg - phi_ref), r*(phibar_bg - phi_ref).
    # 2) optional affine tau-ramp shift (legacy control).
    Y_bg = np.zeros((tau.size, r.size), dtype=float)   # for y
    YB_bg = np.zeros_like(Y_bg)                        # for ybar

    if params.include_twist_background:
        delta_omega = float(omega_tilde) - float(omega_ref)
        phi_bg_tau = float(phi_ref) * np.exp(delta_omega * tau)
        phibar_bg_tau = float(phi_ref) * np.exp(-delta_omega * tau)
        dphi_bg_tau = (phi_bg_tau - float(phi_ref))
        dphibar_bg_tau = (phibar_bg_tau - float(phi_ref))
        if params.twist_background_gated:
            dphi_bg_tau = dphi_bg_tau * g_tau
            dphibar_bg_tau = dphibar_bg_tau * g_tau
        Y_bg += np.outer(dphi_bg_tau, r)
        YB_bg += np.outer(dphibar_bg_tau, r)

    if params.include_tau_ramp:
        if params.ramp_gated:
            # gated affine shift (local in τ)
            h = g_tau
        else:
            # global ramp (discouraged for bounce search)
            # linear ramp from τ=-T to τ=0
            h = (tau - tau.min()) / max(1e-12, (tau.max() - tau.min()))
            h = np.clip(h, 0.0, 1.0)

        # In y = r*(phi - phi_ref): pure background shift is y_bg = r*(phi_bg - phi_ref)
        # where phi_bg = phi_ref + h*(phi_tilde - phi_ref)
        Y_bg += np.outer(h, r * (phi_tilde - phi_ref))
        YB_bg += np.outer(h, r * (phi_tilde - phi_ref))

    # Bubble base (τ-localized)
    Y0 = Y_bg + params.amp * np.outer(g_tau, y1d_loc)
    YB0 = YB_bg + params.amp * np.outer(g_tau, ybar1d_loc)

    # Negative mode kick (optional)
    lam_neg = None
    v_neg = None
    if params.eps != 0.0:
        if neg_mode_override is not None:
            v_neg, lam_neg = neg_mode_override
            v_neg = np.asarray(v_neg, dtype=float).ravel()
        elif sol_ref is not None:
            v_neg, lam_neg = get_negative_mode_from_jacobian_sym(solver, sol_ref, n_eigs=6)
        else:
            v_neg = None

        if v_neg is not None:
            # Convert v_neg (solver vector) into y,ybar arrays by unpacking a perturbation
            # We build dy,dybar from v_neg by unpacking it like a "field increment".
            dy, dybar = solver.unpack(v_neg)

            # window only on kick
            w_r = radial_window(r, rmax, params.r_window_frac)
            w_tau = g_tau.copy()

            if params.use_cosine:
                cos_tau = np.cos(2.0 * np.pi * float(params.k) * (tau / max(1e-12, T)) + float(params.phase))
                w_tau = w_tau * cos_tau

            # Apply windows in y-space (dy is (Nr,Nt) so transpose carefully)
            # Our Y0 is (Nt,Nr), but solver y is (Nr,Nt).
            dY = (w_tau[:, None] * w_r[None, :]) * (dy.T.real)
            dYB = (w_tau[:, None] * w_r[None, :]) * (dybar.T.real)

            Y = Y0 + float(params.eps) * dY
            YB = YB0 + float(params.eps) * dYB
        else:
            Y, YB = Y0, YB0
    else:
        Y, YB = Y0, YB0

    # pack to solver layout (Nr,Nt)
    y_seed = np.asarray(Y, dtype=float).T.astype(complex)
    ybar_seed = np.asarray(YB, dtype=float).T.astype(complex)
    x0 = solver.pack(y_seed, ybar_seed)

    meta = dict(
        omega_ref=float(omega_ref),
        omega_tilde=float(omega_tilde),
        rho_ref=float(phi_ref),
        rho_tilde=float(phi_tilde),
        profile_scale_to_phi=float(profile_scale_to_phi),
        params=dict(**params.__dict__),
        neg_mode_lambda=(float(lam_neg) if lam_neg is not None else None),
        note=(
            "Bounce-oriented seed in solver units |phi|: τ-localized embedded 1D bubble (around phi_tilde) "
            + ("+ twist background exp(±Δω τ) " if params.include_twist_background and not params.twist_background_gated else "")
            + ("+ gated twist background exp(±Δω τ) " if params.include_twist_background and params.twist_background_gated else "")
            + ("+ gated background shift " if params.include_tau_ramp and params.ramp_gated else "")
            + ("+ global τ ramp " if params.include_tau_ramp and not params.ramp_gated else "")
            + ("+ windowed kick(negmode)" if (params.eps != 0.0 and v_neg is not None) else "")
        ),
    )
    return x0, meta


# ---------------------------------------------------------------------
# Seed scoring (residual + Q + anti-banal + energy penalty)
# ---------------------------------------------------------------------

@dataclass
class SeedScoreWeights:
    wF: float = 1.0
    wQ: float = 5.0
    wBanal: float = 50.0        # penalize closeness to banal (strong)
    wE: float = 2.0             # penalize energy mismatch vs homogeneous (relative)
    banal_tol: float = 1e-3     # if dist_to_banal < banal_tol -> heavy penalty
    energy_rel_tol: float = 0.05 # if |E_M/E_M_hom - 1| > tol -> penalty grows


@dataclass
class SeedCandidate:
    params: AnsatzParams
    meta: Dict[str, Any]
    normF: float
    Q: float
    Q_target: float
    dist_banal: float
    E_M: float
    E_M_hom: float
    energy_rel_dev: float
    score: float
    x0: np.ndarray


def residual_norm(solver: Bubble2DSolver, x0: np.ndarray) -> float:
    F = solver.residual(np.asarray(x0))
    return float(np.linalg.norm(F))


def print_seed_diagnostics(
    solver: Bubble2DSolver,
    x0: np.ndarray,
    *,
    label: str = "seed",
) -> None:
    """
    Cheap diagnostics for a single seed x0 (no assumption that F(x0)=0):
      - ||F(x0)||
      - Q(tau=0) ghost charge
      - max|rho_phys(r, tau≈0) - rho_phys_tail| on the tau≈0 slice
    """
    x0 = np.asarray(x0)
    nF = float(np.linalg.norm(solver.residual(x0)))
    y, ybar = solver.unpack(x0)
    Q = float(observables_2d.compute_charge_tau0_ghost_2d(solver, y, ybar, subtract_background=False))

    # rho from phi,phibar to avoid relying on rho_map internals
    phi, phibar = solver.phi(y, ybar)
    u = (phi * phibar).real
    rho = np.sqrt(2.0 * np.maximum(u, 0.0))
    r = np.asarray(solver.grid.r, dtype=float).flatten()
    tau = np.asarray(solver.grid.tau, dtype=float).flatten()
    idx_tau0 = int(np.argmin(np.abs(tau - 0.0)))

    rho_phys_tail = np.mean(rho[-max(4, rho.shape[0] // 8) :, idx_tau0])
    max_dev = float(np.max(np.abs(rho[:, idx_tau0] - rho_phys_tail)))

    print(f"[{label}] ||F|| = {nF:.3e}, Q(tau=0) = {Q:.6g}, max|rho_phys-rho_phys_tail|(tau≈0) = {max_dev:.3e}")


def score_seed(
    solver: Bubble2DSolver,
    x0: np.ndarray,
    *,
    params: AnsatzParams,
    meta: Dict[str, Any],
    Q_target: float,
    weights: SeedScoreWeights,
) -> SeedCandidate:
    # residual
    nF = residual_norm(solver, x0)

    # tau=0 ghost observables (raw total Q, raw E_M)
    y, ybar = solver.unpack(x0)
    Q = float(observables_2d.compute_charge_tau0_ghost_2d(solver, y, ybar, subtract_background=False))
    E_M = float(observables_2d.compute_energy_minkowski_tau0_ghost_2d(solver, y, ybar))

    r = np.asarray(solver.grid.r, dtype=float)
    E_M_hom = float(observables_2d.homogeneous_E_M_2d(float(solver.omega), float(solver.rho0), float(r[-1]), solver.U))
    energy_rel_dev = float(abs(E_M / E_M_hom - 1.0)) if abs(E_M_hom) > 1e-30 else float("inf")

    # banal distance
    xb = _x_banal(solver)
    dist_b = _xdist(np.asarray(x0, dtype=float), np.asarray(xb, dtype=float))

    # penalties
    dQ = abs(Q - float(Q_target))
    score = weights.wF * nF + weights.wQ * dQ

    # anti-banal: if too close, punish hard; else gentle inverse-distance penalty
    if dist_b < weights.banal_tol:
        score += weights.wBanal * (1.0 / max(dist_b, 1e-12))  # huge
    else:
        score += weights.wBanal * (weights.banal_tol / dist_b)

    # energy penalty: encourage solutions not drifting too far from homogeneous energy
    # (this is a heuristic; tune wE and tol)
    if energy_rel_dev > weights.energy_rel_tol:
        score += weights.wE * (energy_rel_dev - weights.energy_rel_tol) * abs(E_M_hom)

    return SeedCandidate(
        params=params,
        meta=meta,
        normF=float(nF),
        Q=float(Q),
        Q_target=float(Q_target),
        dist_banal=float(dist_b),
        E_M=float(E_M),
        E_M_hom=float(E_M_hom),
        energy_rel_dev=float(energy_rel_dev),
        score=float(score),
        x0=np.asarray(x0),
    )


# ---------------------------------------------------------------------
# Search + Newton tries
# ---------------------------------------------------------------------

def rank_seeds(
    solver: Bubble2DSolver,
    *,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    Q_target: float,
    param_grid: Dict[str, Iterable[Any]],
    weights: Optional[SeedScoreWeights] = None,
    sol_ref_for_negmode: Optional[Bubble2DSolution] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> List[SeedCandidate]:
    """
    Build seeds over param_grid and score them.
    param_grid keys can include:
      eps_list, k_list, phase_list, amp_list, tau_gate_frac_list, tau_gate_center_frac_list,
      r_window_frac_list, include_tau_ramp_list, ramp_gated_list, use_cosine_list
    """
    w = weights or SeedScoreWeights()

    # Extract lists with defaults
    def get_list(name: str, default):
        return list(param_grid.get(name, default))

    eps_list = get_list("eps_list", [0.02])
    k_list = get_list("k_list", [1])
    phase_list = get_list("phase_list", [0.0])
    amp_list = get_list("amp_list", [1.0])
    tau_gate_frac_list = get_list("tau_gate_frac_list", [0.15])
    tau_gate_center_frac_list = get_list("tau_gate_center_frac_list", [1.0])
    r_window_frac_list = get_list("r_window_frac_list", [0.15])
    include_tau_ramp_list = get_list("include_tau_ramp_list", [False])
    ramp_gated_list = get_list("ramp_gated_list", [True])
    use_cosine_list = get_list("use_cosine_list", [True])

    candidates: List[SeedCandidate] = []
    tot = (len(eps_list) * len(k_list) * len(phase_list) * len(amp_list) *
           len(tau_gate_frac_list) * len(tau_gate_center_frac_list) *
           len(r_window_frac_list) * len(include_tau_ramp_list) *
           len(ramp_gated_list) * len(use_cosine_list))

    # Precompute negative mode once (expensive: Jacobian + eigsh) instead of per seed
    neg_mode_override: Optional[Tuple[np.ndarray, float]] = None
    if sol_ref_for_negmode is not None:
        neg_mode_override = get_negative_mode_from_jacobian_sym(solver, sol_ref_for_negmode, n_eigs=6)
        if verbose:
            print(f"[SeedScan] precomputed negative mode (λ={neg_mode_override[1]:.4e})")

    # Precompute 1D bubble profile once (same for all seeds)
    r_grid = np.asarray(solver.grid.r, dtype=float).flatten()
    _phi1d, _phibar1d = bubble_profile_1d(float(omega_tilde), r_grid)
    _phi1d, _phibar1d, _ = _normalize_profile_to_phi_units(
        _phi1d, _phibar1d, phi_ref=float(solver.rho0), m=8
    )
    _phi_tilde = float(_estimate_phi_hom_from_tail(_phi1d, _phibar1d, m=8))
    phi1d_override = (_phi1d, _phibar1d)

    # Build list of all param combinations
    all_params: List[AnsatzParams] = []
    for eps in eps_list:
        for k in k_list:
            for phase in phase_list:
                for amp in amp_list:
                    for tgf in tau_gate_frac_list:
                        for tgc in tau_gate_center_frac_list:
                            for rwf in r_window_frac_list:
                                for inc_ramp in include_tau_ramp_list:
                                    for rg in ramp_gated_list:
                                        for uc in use_cosine_list:
                                            all_params.append(AnsatzParams(
                                                eps=float(eps), k=int(k), phase=float(phase), amp=float(amp),
                                                tau_gate_frac=float(tgf), tau_gate_center_frac=float(tgc),
                                                r_window_frac=float(rwf), include_tau_ramp=bool(inc_ramp),
                                                ramp_gated=bool(rg), use_cosine=bool(uc),
                                            ))

    def _build_and_score(params: AnsatzParams) -> SeedCandidate:
        x0, meta = build_seed_bubble(
            solver,
            omega_ref=float(omega_ref),
            omega_tilde=float(omega_tilde),
            bubble_profile_1d=bubble_profile_1d,
            params=params,
            rho_tilde_override=_phi_tilde,
            phi1d_override=phi1d_override,
            sol_ref=sol_ref_for_negmode if neg_mode_override is None else None,
            neg_mode_override=neg_mode_override,
        )
        return score_seed(solver, x0, params=params, meta=meta, Q_target=float(Q_target), weights=w)

    n_workers = (int(n_jobs) if n_jobs is not None and n_jobs > 1 else 1)
    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        if verbose:
            print(f"[SeedScan] building+scoring {tot} seeds (n_jobs={n_workers})...")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            candidates = list(ex.map(_build_and_score, all_params))
    else:
        if verbose:
            print(f"[SeedScan] building+scoring {tot} seeds...")
        for idx, params in enumerate(all_params, start=1):
            cand = _build_and_score(params)
            candidates.append(cand)
            if verbose and (tot <= 50 or idx <= 10 or idx % 50 == 0 or idx == tot):
                print(f"  [{idx}/{tot}] score={cand.score:.3e}  ||F||={cand.normF:.3e}  "
                      f"|Q-Q*|={abs(cand.Q-cand.Q_target):.3e}  dist_banal={cand.dist_banal:.3e}  "
                      f"Erel={cand.energy_rel_dev:.3e}  params={params}")
    candidates.sort(key=lambda c: c.score)
    return candidates


def search_best_seed_and_solve(
    solver: Bubble2DSolver,
    *,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    Q_target: float,
    param_grid: Dict[str, Iterable[Any]],
    weights: Optional[SeedScoreWeights] = None,
    sol_ref_for_negmode: Optional[Bubble2DSolution] = None,
    n_jobs: Optional[int] = None,
    max_newton_tries: int = 10,
    solve_verbose: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[Bubble2DSolution], Dict[str, Any]]:
    """
    Rank seeds then try Newton in order until success.
    Returns (solution or None, report dict).
    n_jobs: if > 1, score seeds in parallel with that many threads (can speed up the scan).
    """
    cand_list = rank_seeds(
        solver,
        omega_ref=omega_ref,
        omega_tilde=omega_tilde,
        bubble_profile_1d=bubble_profile_1d,
        Q_target=Q_target,
        param_grid=param_grid,
        weights=weights,
        sol_ref_for_negmode=sol_ref_for_negmode,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if verbose:
        print(f"[SeedScan] top 10 by score:")
        for i, c in enumerate(cand_list[:10], start=1):
            print(f"  {i:2d}: score={c.score:.3e}  ||F||={c.normF:.3e}  "
                  f"|Q-Q*|={abs(c.Q-c.Q_target):.3e}  dist_banal={c.dist_banal:.3e}  "
                  f"Erel={c.energy_rel_dev:.3e}  params={c.params}")

    tried = 0
    for rank, cand in enumerate(cand_list[:max_newton_tries], start=1):
        tried += 1
        if verbose:
            print(f"[SeedScan] Newton try {tried}/{max_newton_tries}: rank {rank}  score={cand.score:.3e}  params={cand.params}")
        try:
            sol = solver.solve(cand.x0, verbose=solve_verbose, verbose_success_block=False)
            if getattr(sol, "success", True) and sol.newton.success:
                if verbose:
                    print(f"[SeedScan] SUCCESS at rank {rank}: ||F||={sol.residual_norm:.3e}, iters={sol.iterations}")
                report = dict(
                    candidates=cand_list,
                    best_rank=rank,
                    best_candidate=cand,
                    tried=tried,
                )
                return sol, report
        except (NewtonConvergenceError, RuntimeError) as e:
            if verbose:
                print(f"  -> fail: {type(e).__name__}: {e}")
        except Exception as e:
            if verbose:
                print(f"  -> fail: {type(e).__name__}: {e}")

    if verbose:
        print("[SeedScan] No convergence.")
    return None, dict(candidates=cand_list, best_rank=None, best_candidate=None, tried=tried)


# ---------------------------------------------------------------------
# Simple O(3) seed wrappers
# ---------------------------------------------------------------------

def build_seed_O3_tau(
    solver: Bubble2DSolver,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d,
    *,
    amp: float = 1.0,
):
    """
    O(3)-tau seed: build a 1D profile and place it along tau, identically in r.
    (Useful as a diagnostic "wrong orientation" seed when comparing basins.)
    Returns (x0, meta).
    """
    r = np.asarray(solver.grid.r, dtype=float).flatten()
    tau = np.asarray(solver.grid.tau, dtype=float).flatten()

    # Build a nonnegative radial coordinate from tau so profile center is at tau≈0.
    s_tau = np.abs(tau)
    phi_tau, phibar_tau = bubble_profile_1d(float(omega_tilde), s_tau)
    phi_tau, phibar_tau, _ = _normalize_profile_to_phi_units(
        phi_tau, phibar_tau, phi_ref=float(solver.rho0), m=8
    )
    phi_hom = _estimate_phi_hom_from_tail(phi_tau, phibar_tau, m=8)

    phi_tau_scaled = phi_hom + float(amp) * (phi_tau - phi_hom)
    phi_2d = np.repeat(phi_tau_scaled[None, :], r.size, axis=0)  # (Nr,Nt)

    params = AnsatzParams(
        eps=0.0,
        k=1,
        phase=0.0,
        amp=1.0,
        tau_gate_frac=1.0,
        tau_gate_center_frac=1.0,
        r_window_frac=0.15,
        include_tau_ramp=False,
        ramp_gated=False,
        use_cosine=False,
    )
    x0, meta = build_seed_bubble(
        solver,
        omega_ref=float(omega_ref),
        omega_tilde=float(omega_tilde),
        bubble_profile_1d=bubble_profile_1d,
        params=params,
        rho_tilde_override=phi_hom,
        rho_override_2d=phi_2d,
    )
    meta["note"] = "O(3)-tau seed: profile along tau, copied in r."
    return x0, meta


def build_seed_O3_r(
    solver: Bubble2DSolver,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d,
    *,
    amp: float = 1.0,
):
    """
    O(3)-r seed: build rho_1d(r) from bubble_profile_1d and copy it
    identically on every tau-slice (strictly no tau dependence).
    Returns (x0, meta).
    """
    r = np.asarray(solver.grid.r, dtype=float).flatten()
    tau = np.asarray(solver.grid.tau, dtype=float).flatten()

    # 1D O(3) profile (depends only on r), normalized to solver units |phi|
    phi1d, phibar1d = bubble_profile_1d(float(omega_tilde), r)
    phi1d, phibar1d, _ = _normalize_profile_to_phi_units(
        phi1d, phibar1d, phi_ref=float(solver.rho0), m=8
    )
    phi_hom = _estimate_phi_hom_from_tail(phi1d, phibar1d, m=8)

    # Strictly tau-independent: same radial profile on all tau slices
    phi1d_scaled = phi_hom + float(amp) * (phi1d - phi_hom)
    phi_2d = np.repeat(phi1d_scaled[:, None], tau.size, axis=1)

    # Dummy params for metadata consistency; rho_override_2d bypasses tau-gate/kick logic
    params = AnsatzParams(
        eps=0.0,
        k=1,
        phase=0.0,
        amp=1.0,
        tau_gate_frac=1.0,
        tau_gate_center_frac=1.0,
        r_window_frac=0.15,
        include_tau_ramp=False,
        ramp_gated=False,
        use_cosine=False,
    )
    x0, meta = build_seed_bubble(
        solver,
        omega_ref=float(omega_ref),
        omega_tilde=float(omega_tilde),
        bubble_profile_1d=bubble_profile_1d,
        params=params,
        rho_tilde_override=phi_hom,
        phi1d_override=(phi1d, phibar1d),
        rho_override_2d=phi_2d,
    )


def build_seed_O3_tau_independent(
    solver: Bubble2DSolver,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d,
    *,
    amp: float = 1.0,
):
    """
    Backward-compatible alias for the strictly tau-independent O(3)-r seed.
    """
    return build_seed_O3_r(
        solver,
        omega_ref=omega_ref,
        omega_tilde=omega_tilde,
        bubble_profile_1d=bubble_profile_1d,
        amp=amp,
    )


# ---------------------------------------------------------------------
# O(4)-like seed built from rho_1d(s) with s = sqrt(r^2 + (tau/alpha_tau)^2)
# ---------------------------------------------------------------------

def build_seed_O4_from_rho_of_s(
    solver: Bubble2DSolver,
    omega_ref: float,
    omega_tilde: float,
    bubble_profile_1d,
    *,
    alpha_tau: float = 1.0,
    amp: float = 1.0,
    # optional window near tau=-beta/2 to ease BC:
    use_tau_window: bool = True,
    tau_window_frac: float = 0.25,
    tau_window_kind: str = "cosine",  # or "gauss"
):
    """
    O(4)-like seed built from phi_1d(s) in solver units |phi|, with s = sqrt(r^2 + (tau/alpha_tau)^2).

    Guarantees:
      - phi(r,0) = phi_1d(r)
      - d_tau phi(r,0) = 0 (evenness in tau)
      - localization in tau, r if phi_1d -> phi_hom for large r.

    Optional:
      Multiply only the *deviation from background* by a τ-window w(τ) that goes to 0 near τ=-beta/2
      to reduce boundary mismatch.

    Returns (x0, meta) via build_seed_bubble with rho_override_2d.
    """
    r = np.asarray(solver.grid.r, dtype=float).flatten()
    tau = np.asarray(solver.grid.tau, dtype=float).flatten()

    # 1D profile (phi1d, phibar1d), normalized to solver units |phi|
    phi1d, phibar1d = bubble_profile_1d(float(omega_tilde), r)
    phi1d, phibar1d, _ = _normalize_profile_to_phi_units(
        phi1d, phibar1d, phi_ref=float(solver.rho0), m=8
    )

    # Homogeneous tail phi_hom from 1D profile
    phi_hom = _estimate_phi_hom_from_tail(phi1d, phibar1d, m=8)

    # Build phi(r, tau) = phi1d(s), s = sqrt(r^2 + (tau/alpha_tau)^2)
    R, T = np.meshgrid(r, tau, indexing="ij")  # (Nr,Nt)
    alpha_tau = float(alpha_tau)
    S = np.sqrt(R ** 2 + (T / max(alpha_tau, 1e-12)) ** 2)

    # Interpolate phi1d(S) from phi1d(r)
    S_flat = S.ravel()
    phi_2d = np.interp(
        S_flat,
        r,
        phi1d,
        left=phi1d[0],
        right=phi1d[-1],
    ).reshape(S.shape)

    # Optional tau-window on the deviation from background (in |phi| units)
    if use_tau_window:
        beta = float(solver.settings.beta)
        T_half = 0.5 * beta
        Delta = float(tau_window_frac) * T_half
        tau_arr = tau

        w_tau = np.zeros_like(tau_arr, dtype=float)
        for i, t in enumerate(tau_arr):
            if -Delta <= t <= 0.0:
                w_tau[i] = 1.0
            elif -T_half <= t < -Delta:
                if tau_window_kind == "cosine":
                    num = t + Delta
                    den = max(T_half - Delta, 1e-12)
                    x = np.clip(num / den, -1.0, 1.0)
                    w_tau[i] = 0.5 * (1.0 + np.cos(np.pi * x))
                elif tau_window_kind == "gauss":
                    sigma = max(Delta, 1e-12)
                    x = (t + T_half) / sigma
                    w_tau[i] = float(np.exp(-0.5 * x * x))
                else:
                    raise ValueError("tau_window_kind must be 'cosine' or 'gauss'")
            else:
                w_tau[i] = 0.0
        w_tau_2d = w_tau[None, :]
        delta_phi = phi_2d - phi_hom
        phi_2d = phi_hom + w_tau_2d * delta_phi

    # Amplitude on the deviation from background (keeps far tail near phi_hom)
    phi_2d = phi_hom + float(amp) * (phi_2d - phi_hom)

    # Use build_seed_bubble with rho_override_2d; params mostly irrelevant here
    params = AnsatzParams(
        eps=0.0,
        k=1,
        phase=0.0,
        amp=1.0,
        tau_gate_frac=1.0,
        tau_gate_center_frac=1.0,
        r_window_frac=0.15,
        include_tau_ramp=False,
        ramp_gated=False,
        use_cosine=False,
    )
    return build_seed_bubble(
        solver,
        omega_ref=float(omega_ref),
        omega_tilde=float(omega_tilde),
        bubble_profile_1d=bubble_profile_1d,
        params=params,
        rho_ref_override=None,
        rho_tilde_override=phi_hom,
        phi1d_override=(phi1d, phibar1d),
        sol_ref=None,
        neg_mode_override=None,
        rho_override_2d=phi_2d,
    )
    meta["profile_d"] = int(getattr(bubble_profile_1d, "profile_d", -1))
    meta["S_max"] = float(np.max(S))
    return x0, meta


# ---------------------------------------------------------------------
# Full diagnostics + plots
# ---------------------------------------------------------------------

def slicewise_energies_simple(solver: Bubble2DSolver, sol: Bubble2DSolution) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (tau_sorted, E_static(tau), E_full(tau)) for quick diagnostics.
    E_static = ∫( (∂rφ)(∂rφ̄) + V )  (no tau-kin)
    E_full   = ∫( (∂τφ)(∂τφ̄) + (∂rφ)(∂rφ̄) + V )
    Uses simple finite differences on phi arrays; good enough for diagnostics/plots.
    """
    tau = np.asarray(solver.grid.tau, dtype=float).flatten()
    r = np.asarray(solver.grid.r, dtype=float).flatten()

    phi, phibar = solver.phi(sol.y, sol.ybar)  # (Nr,Nt)
    # transpose to (Nt,Nr) for gradient convenience
    phi_t = phi.T
    phibar_t = phibar.T

    dphi_dt = np.gradient(phi_t, tau, axis=0, edge_order=2)
    dphibar_dt = np.gradient(phibar_t, tau, axis=0, edge_order=2)
    dphi_dr = np.gradient(phi_t, r, axis=1, edge_order=2)
    dphibar_dr = np.gradient(phibar_t, r, axis=1, edge_order=2)

    tau_kin = (dphi_dt * dphibar_dt).real
    spatial = (dphi_dr * dphibar_dr).real
    s = (phi_t * phibar_t).real
    V = solver.U(np.sqrt(2.0 * np.maximum(s, 0.0)))

    pref = 4.0 * np.pi
    E_static = pref * np.trapz((spatial + V) * (r[None, :] ** 2), r, axis=1)
    E_full = pref * np.trapz((tau_kin + spatial + V) * (r[None, :] ** 2), r, axis=1)

    order = np.argsort(tau)
    return tau[order], E_static[order], E_full[order]


def full_diagnostics(solver: Bubble2DSolver, sol: Bubble2DSolution) -> Dict[str, Any]:
    y, ybar = sol.y, sol.ybar

    Q_ghost = float(observables_2d.compute_charge_tau0_ghost_2d(solver, y, ybar, subtract_background=False))
    E_M_ghost = float(observables_2d.compute_energy_minkowski_tau0_ghost_2d(solver, y, ybar))
    r = np.asarray(solver.grid.r, dtype=float)
    E_M_hom = float(observables_2d.homogeneous_E_M_2d(float(solver.omega), float(solver.rho0), float(r[-1]), solver.U))
    Erel = float(E_M_ghost / E_M_hom) if abs(E_M_hom) > 1e-30 else float("nan")

    rho = solver.rho_map(y, ybar)  # (Nr,Nt)
    # crude amp_tau at r index 1
    i_r = min(1, solver.Nr - 1)
    amp_tau = float(np.max(np.abs(rho[i_r, :] - rho[i_r, 0])))

    # E(τ) canonical (H_E slicewise) as in Seed_Basin_Debug: solver.compute_energy + reflection
    _, E_tau = solver.compute_energy(y, ybar, return_profile=True)
    tau_grid = np.asarray(solver.grid.tau, dtype=float).flatten()
    tau_pos = -np.flip(tau_grid)[1:]
    E_pos = np.flip(E_tau)[1:]
    tau_full = np.concatenate([tau_grid, tau_pos])
    E_full = np.concatenate([E_tau, E_pos])
    sidx = np.argsort(tau_full)
    tau_sorted = tau_full[sidx]
    E_tau_canonical = E_full[sidx]

    return dict(
        Q_ghost=Q_ghost,
        E_M_ghost=E_M_ghost,
        E_M_hom=E_M_hom,
        E_M_ratio=Erel,
        amp_tau=amp_tau,
        tau_sorted=tau_sorted,
        E_tau_canonical=E_tau_canonical,
        rho_2d=rho,
        r=r,
        tau=np.asarray(solver.grid.tau, dtype=float),
    )


def plot_solution_and_diagnostics(
    solver: Bubble2DSolver,
    sol: Bubble2DSolution,
    diag: Dict[str, Any],
    *,
    title: str = "",
    savepath: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    tau_sorted = diag["tau_sorted"]
    E_tau_canonical = diag["E_tau_canonical"]
    E_M_ghost = diag["E_M_ghost"]
    rho = diag["rho_2d"]
    r = diag["r"]
    tau = diag["tau"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Sinistra: entrambe le energie nello stesso plot
    axes[0].plot(tau_sorted, -E_tau_canonical, "b-", lw=2, label=r"$E(\tau)$ canonical ($H_E$)")
    axes[0].axhline(E_M_ghost, color="C1", linestyle="--", linewidth=2, label=rf"$E_M(\tau{{=}}0)$ = {E_M_ghost:.6g}")
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$E$")
    axes[0].set_title(r"Energies: $E(\tau)$ and $E_M(\tau=0)$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Destra: rho(r,τ) 2D con τ invertito: τ≈0 in alto
    tau_min = float(np.min(tau))
    tau_max = float(np.max(tau))
    # Con origin="upper", usare extent standard (bottom=tau_min, top=tau_max)
    # mantiene coerente la griglia: in alto compare tau_max (tipicamente tau≈0).
    extent = (float(r[0]), float(r[-1]), tau_min, tau_max)
    im = axes[1].imshow(rho.T, origin="upper", aspect="auto", extent=extent)
    axes[1].set_xlabel("r")
    axes[1].set_ylabel(r"$\tau$")
    axes[1].set_title(r"$\rho(r,\tau)$")
    axes[1].set_ylim(tau_max, tau_min)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

    print("--- Diagnostics ---")
    print(f"success = {sol.success}, iters={sol.iterations}, ||F||={sol.residual_norm:.3e}")
    print(f"Q_ghost(τ=0)      = {diag['Q_ghost']:.8e}")
    print(f"E_M_ghost(τ=0)    = {diag['E_M_ghost']:.8e}")
    print(f"E_M_hom           = {diag['E_M_hom']:.8e}")
    print(f"E_M_ratio         = {diag['E_M_ratio']:.6g}")
    print(f"amp_tau (r idx 1) = {diag['amp_tau']:.6g}")


__all__ = [
    "AnsatzParams",
    "SeedScoreWeights",
    "SeedCandidate",
    "make_bubble_profile_1d_from_arrays",
    "make_bubble_profile_1d_from_solve_bounce",
    "make_bubble_profile_1d_for_solver",
    "build_seed_bubble",
    "build_seed_O3_tau",
    "build_seed_O3_r",
    "build_seed_O3_tau_independent",
    "build_seed_O4_from_rho_of_s",
    "print_seed_diagnostics",
    "rank_seeds",
    "search_best_seed_and_solve",
    "full_diagnostics",
    "plot_solution_and_diagnostics",
]