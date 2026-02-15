# bounce_existence.py — Existence checker (pinned-branch continuation) for Bubble2DSolver
#
# PURPOSE
#   Diagnostic continuation method (Q-ball-decay style) to answer:
#     "Does a non-trivial 2D bounce exist, or is Newton collapsing to the trivial background?"
#
# KEY POINT (COMPLEX SADDLE)
#   This version is written for the *complex* saddle relevant to microcanonical bubble nucleation.
#   It therefore requires:
#
#       solver.settings.complex_saddle == True
#
#   In that case, the solver unknown vector is the real/imag split:
#
#       x_fields = [Re y, Im y, Re ybar, Im ybar]  (real vector, length 4*Nsite)
#
#   The solver residual/Jacobian are the mathematically consistent real system in this basis.
#
# STRATEGY (PINNED BRANCH)
#   Introduce a scalar Lagrange multiplier λ_pin and solve the augmented system:
#       - Replace ONE field equation row k_eq by:   F_k(x_fields) + λ_pin = 0
#       - Add ONE constraint at (pin_site) fixing the local amplitude ρ(pin_site) = ρ_pin
#   Track solutions as ρ_pin is varied (continuation).
#   If λ_pin crosses 0, you have an unpinned solution of the original system (candidate bounce).
#
# IMPORTANT DETAILS FOR BUBBLE NUCLEATION
#   - The pinned amplitude is defined consistently with the solver:
#         u = Re( φ_rot * φbar_rot )
#         ρ = sqrt( smooth_pos(u) + ρ_eps )
#     using the same smooth_pos as the solver (and its derivative) for a stable constraint row.
#   - Use pin_site with j>=2 (avoid tiny r). Default pin_site=(4,0).
#   - We build the constraint in terms of the *physical* u (includes 1/r^2).
#
# API
#   Primary entry point:
#       existence_driver(solver, rho_pin_values, x0_fields, ...)
#
#   The solver must provide:
#       - solver.residual_fields(x_fields) -> (N_fields,) real vector (RI split)
#       - solver.jacobian_fields(x_fields) -> sparse (N_fields, N_fields)
#       - solver.system_size_fields()
#       - solver.get_ri_at_site(x_fields, j, i)
#       - solver.field_variable_indices_at_site(j, i)
#       - solver.field_equation_index(j, i, which)
#       - solver._smooth_pos(u_array)  (returns u_pos, dupos_du)
#       - solver.grid.r, solver.rho0, solver.settings.rho_eps

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class NewtonResultAug:
    x: np.ndarray
    success: bool
    iterations: int
    residual_norm: float
    history: List[float]
    last_step: float
    info: Dict[str, Any]


@dataclass
class PinnedBranchPoint:
    rho_pin: float
    lambda_pin: float
    residual_norm: float
    success: bool
    x_aug: np.ndarray
    info: Dict[str, Any]


@dataclass
class ExistenceScanResult:
    points: List[PinnedBranchPoint]
    zero_crossings: List[Tuple[int, int]]
    best_guess: Optional[PinnedBranchPoint]
    pin_site: Tuple[int, int]
    anchor_site: Tuple[int, int]
    anchor_which: str


# =============================================================================
# Internal helpers (ρ definition consistent with solver)
# =============================================================================

def _require_complex_saddle(solver) -> None:
    if not bool(getattr(solver.settings, "complex_saddle", False)):
        raise ValueError(
            "bounce_existence.py (complex version) requires solver.settings.complex_saddle=True.\n"
            "Create Bubble2DSettings(..., complex_saddle=True) before running existence_driver."
        )


def _rho_and_grads_at_site(
    solver,
    x_fields: np.ndarray,
    site: Tuple[int, int],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute ρ(site) and its gradients wrt local RI variables:
      dρ/d(y_re), dρ/d(y_im), dρ/d(yb_re), dρ/d(yb_im)

    with
      y_tot    = y + r ρ0
      yb_tot   = ybar + r ρ0
      u        = Re( (y_tot/r) * (yb_tot/r) ) = Re(y_tot*yb_tot)/r^2
      u_pos    = smooth_pos(u)
      ρ        = sqrt(u_pos + ρ_eps)
    """
    j0, i0 = site
    r = float(solver.grid.r[j0])
    if r <= 0:
        raise ValueError("Non-positive r at pin site (unexpected).")

    eps_rho = float(solver.settings.rho_eps)

    y_re, y_im, yb_re, yb_im = solver.get_ri_at_site(x_fields, j0, i0)
    rho0 = float(solver.rho0)

    # totals (fluctuation + background)
    yR = float(y_re + r * rho0)
    yI = float(y_im)
    bR = float(yb_re + r * rho0)
    bI = float(yb_im)

    inv_r2 = 1.0 / (r * r)

    # u = Re(y_tot * yb_tot)/r^2 = (yR*bR - yI*bI)/r^2
    u = (yR * bR - yI * bI) * inv_r2

    u_pos, dupos_du = solver._smooth_pos(np.array([u], dtype=float))
    u_pos = float(u_pos[0])
    dupos_du = float(dupos_du[0])

    rho = float(np.sqrt(u_pos + eps_rho))
    drho_du = 0.5 * dupos_du / rho

    grads = {
        "y_re":  drho_du * (bR * inv_r2),
        "y_im":  drho_du * (-bI * inv_r2),
        "yb_re": drho_du * (yR * inv_r2),
        "yb_im": drho_du * (-yI * inv_r2),
    }
    return rho, grads


def _rho_at_site_like_solver(solver, x_fields: np.ndarray, site: Tuple[int, int]) -> float:
    rho, _ = _rho_and_grads_at_site(solver, x_fields, site)
    return float(rho)


def _pin_constraint_and_row(
    solver,
    x_fields: np.ndarray,
    pin_site: Tuple[int, int],
    rho_pin: float,
) -> Tuple[float, sp.csc_matrix]:
    """
    Build g(x_fields)=ρ(pin_site)-ρ_pin and its sparse Jacobian row (1 × N_fields).
    """
    rho, grads = _rho_and_grads_at_site(solver, x_fields, pin_site)
    g = float(rho - float(rho_pin))

    N_fields = solver.system_size_fields()
    row = sp.lil_matrix((1, N_fields), dtype=float)

    j0, i0 = pin_site
    idxs = solver.field_variable_indices_at_site(j0, i0)
    for k, v in grads.items():
        row[0, idxs[k]] = float(v)

    return g, row.tocsc()


# =============================================================================
# Newton for augmented system (fields + lambda)
# =============================================================================

def newton_solve_aug(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], sp.spmatrix],
    x0: np.ndarray,
    *,
    tol: float = 1e-9,
    max_iter: int = 50,
    damping: float = 1.0,
    max_backtracks: int = 25,
    min_step: float = 1e-12,
    callback: Optional[Callable[[int, np.ndarray, np.ndarray, float, float], None]] = None,
) -> NewtonResultAug:
    """
    Newton–Raphson with backtracking line search.
    Assumes a *real* system (RI split) and uses sparse linear solves.
    """
    x = np.asarray(x0, dtype=float).copy()

    hist: List[float] = []
    last_alpha = 1.0
    info: Dict[str, Any] = {}

    for it in range(1, int(max_iter) + 1):
        F = np.asarray(residual(x), dtype=float)
        nF = float(np.linalg.norm(F))
        hist.append(nF)

        if callback is not None:
            callback(it, x, F, nF, last_alpha)

        if not np.isfinite(nF):
            return NewtonResultAug(x=x, success=False, iterations=it, residual_norm=nF, history=hist,
                                   last_step=last_alpha, info={"error": "Non-finite residual."})

        if nF < float(tol):
            return NewtonResultAug(x=x, success=True, iterations=it, residual_norm=nF, history=hist,
                                   last_step=last_alpha, info=info)

        J = jacobian(x)
        if not sp.issparse(J):
            J = sp.csc_matrix(np.asarray(J, dtype=float))
        else:
            J = sp.csc_matrix(J)

        try:
            dx = spla.spsolve(J, -F)
        except Exception as e:
            return NewtonResultAug(x=x, success=False, iterations=it, residual_norm=nF, history=hist,
                                   last_step=last_alpha, info={"error": f"spsolve failed: {e}"})

        alpha = float(damping)
        ok = False
        for _ in range(int(max_backtracks)):
            xt = x + alpha * dx
            Ft = np.asarray(residual(xt), dtype=float)
            nt = float(np.linalg.norm(Ft))
            if np.isfinite(nt) and nt < nF:
                ok = True
                x = xt
                last_alpha = alpha
                break
            alpha *= 0.5
            if alpha < float(min_step):
                break

        if not ok:
            return NewtonResultAug(x=x, success=False, iterations=it, residual_norm=nF, history=hist,
                                   last_step=alpha, info={"error": "Line search failed."})

    return NewtonResultAug(x=x, success=False, iterations=int(max_iter), residual_norm=float(hist[-1]), history=hist,
                           last_step=last_alpha, info={"error": "max_iter reached"})


# =============================================================================
# Augmented pinned system builder
# =============================================================================

def _normalize_anchor_which(anchor_which: str) -> str:
    a = str(anchor_which).strip().lower()
    if a in ("y", "yre", "y_re"):
        return "y_re"
    if a in ("yi", "yim", "y_im"):
        return "y_im"
    if a in ("ybar", "yb", "ybre", "yb_re"):
        return "yb_re"
    if a in ("ybi", "ybim", "yb_im"):
        return "yb_im"
    raise ValueError("anchor_which must be one of: 'y_re','y_im','yb_re','yb_im' (or legacy: 'y','ybar').")


def make_pinned_system(
    solver,
    *,
    rho_pin: float,
    pin_site: Tuple[int, int] = (4, 0),
    anchor_site: Optional[Tuple[int, int]] = None,
    anchor_which: str = "y_re",
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], sp.csc_matrix], int]:
    _require_complex_saddle(solver)

    if anchor_site is None:
        anchor_site = pin_site

    anchor_which = _normalize_anchor_which(anchor_which)

    jA, iA = anchor_site
    k_eq = int(solver.field_equation_index(jA, iA, anchor_which))

    N_fields = int(solver.system_size_fields())

    def residual_aug(x_aug: np.ndarray) -> np.ndarray:
        x_fields = np.asarray(x_aug[:N_fields], dtype=float)
        lam = float(x_aug[N_fields])

        F = np.asarray(solver.residual_fields(x_fields), dtype=float).copy()
        F[k_eq] += lam

        g, _ = _pin_constraint_and_row(solver, x_fields, pin_site, float(rho_pin))
        return np.concatenate([F, np.array([g], dtype=float)])

    def jacobian_aug(x_aug: np.ndarray) -> sp.csc_matrix:
        x_fields = np.asarray(x_aug[:N_fields], dtype=float)

        J = sp.csc_matrix(solver.jacobian_fields(x_fields))

        col = sp.lil_matrix((N_fields, 1), dtype=float)
        col[k_eq, 0] = 1.0
        col = col.tocsc()

        _, row = _pin_constraint_and_row(solver, x_fields, pin_site, float(rho_pin))

        zero = sp.csc_matrix((1, 1), dtype=float)
        return sp.bmat([[J, col], [row, zero]], format="csc")

    return residual_aug, jacobian_aug, k_eq


# =============================================================================
# Continuation scan
# =============================================================================

def _build_ramp(a: float, b: float, n: int) -> np.ndarray:
    n = int(max(n, 0))
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([b], dtype=float)
    return np.linspace(a, b, n + 1, dtype=float)[1:]


def scan_pinned_branch(
    solver,
    rho_pin_values: np.ndarray,
    *,
    x0_fields: np.ndarray,
    pin_site: Tuple[int, int] = (4, 0),
    anchor_site: Optional[Tuple[int, int]] = None,
    anchor_which: str = "y_re",
    lam0: float = 0.0,
    start_near_seed: bool = True,
    homotopy_steps: int = 10,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    damping: Optional[float] = None,
    verbose: bool = True,
    newton_verbose: bool = False,
) -> ExistenceScanResult:
    _require_complex_saddle(solver)

    vals = np.asarray(rho_pin_values, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("rho_pin_values must be a non-empty 1D array.")

    if tol is None:
        tol = float(getattr(solver.settings, "newton_tol", 1e-9))
    if max_iter is None:
        max_iter = int(getattr(solver.settings, "newton_max_iter", 50))
    if damping is None:
        damping = float(getattr(solver.settings, "damping", 1.0))

    x0_fields = np.asarray(solver.to_fields_vector(x0_fields), dtype=float)

    rho_seed = _rho_at_site_like_solver(solver, x0_fields, pin_site)

    reordered = False
    if start_near_seed and vals.size >= 2:
        if abs(vals[-1] - rho_seed) < abs(vals[0] - rho_seed):
            vals = vals[::-1]
            reordered = True

    ramp = _build_ramp(rho_seed, float(vals[0]), homotopy_steps)
    vals_full = np.concatenate([ramp, vals])

    N_fields = int(solver.system_size_fields())
    x_aug = np.concatenate([x0_fields, np.array([float(lam0)], dtype=float)])
    last_good = x_aug.copy()

    points: List[PinnedBranchPoint] = []

    anchor_which_n = _normalize_anchor_which(anchor_which)

    if verbose:
        print(f"[ExistenceScan] pin_site={pin_site}, anchor_site={anchor_site or pin_site}, anchor_which={anchor_which_n}")
        print(f"[ExistenceScan] rho_seed(pin_site) = {rho_seed:.6g}")
        print(f"[ExistenceScan] reordered={reordered}, ramp_steps={homotopy_steps}, total_points={len(vals_full)}")
        print(f"[ExistenceScan] k   rho_pin        lambda        ||F||     ok  iters  alpha")

    for k, rho_pin in enumerate(vals_full):
        res_aug, jac_aug, _ = make_pinned_system(
            solver, rho_pin=float(rho_pin), pin_site=pin_site, anchor_site=anchor_site, anchor_which=anchor_which_n
        )

        seed = last_good

        def cb(it, x, F, nF, alpha):
            if newton_verbose:
                lam = float(x[N_fields])
                print(f"      [Newton] it={it:02d}  ||F||={nF:.3e}  lambda={lam:+.3e}  alpha={alpha:.2e}")

        nr = newton_solve_aug(
            res_aug, jac_aug, seed,
            tol=float(tol),
            max_iter=int(max_iter),
            damping=float(damping),
            callback=cb if newton_verbose else None,
        )

        lam = float(nr.x[N_fields])
        pt = PinnedBranchPoint(
            rho_pin=float(rho_pin),
            lambda_pin=float(lam),
            residual_norm=float(nr.residual_norm),
            success=bool(nr.success),
            x_aug=nr.x.copy(),
            info=dict(iterations=int(nr.iterations), last_step=float(nr.last_step), error=str(nr.info.get("error", ""))),
        )

        if pt.success:
            last_good = pt.x_aug.copy()

        points.append(pt)

        if verbose:
            ok = "OK" if pt.success else "FAIL"
            err = pt.info.get("error", "")
            err = ((" | " + err[:60]) if (not pt.success and err) else "")
            print(f"[ExistenceScan] {k:3d}  {pt.rho_pin:10.6g}  {pt.lambda_pin:+11.3e}  {pt.residual_norm:8.2e}  {ok:4s}  {pt.info['iterations']:4d}  {pt.info['last_step']:.1e}{err}")

    zero_cross: List[Tuple[int, int]] = []
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        if not (a.success and b.success):
            continue
        la, lb = a.lambda_pin, b.lambda_pin
        if not (np.isfinite(la) and np.isfinite(lb)):
            continue
        if la == 0.0:
            zero_cross.append((k, k))
        elif la * lb < 0.0:
            zero_cross.append((k, k + 1))

    best_guess = None
    if zero_cross:
        candidates: List[PinnedBranchPoint] = []
        for ia, ib in zero_cross:
            candidates.append(points[ia])
            candidates.append(points[ib])
        best_guess = min(candidates, key=lambda p: abs(p.lambda_pin))

    return ExistenceScanResult(
        points=points,
        zero_crossings=zero_cross,
        best_guess=best_guess,
        pin_site=pin_site,
        anchor_site=(anchor_site or pin_site),
        anchor_which=anchor_which_n,
    )


# =============================================================================
# Refinement and “unpin verification”
# =============================================================================

def refine_lambda_zero_by_bisect(
    solver,
    *,
    rho_a: float,
    x_aug_a: np.ndarray,
    rho_b: float,
    x_aug_b: np.ndarray,
    pin_site: Tuple[int, int] = (4, 0),
    anchor_site: Optional[Tuple[int, int]] = None,
    anchor_which: str = "y_re",
    max_iter: int = 25,
    tol_lam: float = 1e-8,
    tol_newton: Optional[float] = None,
    max_newton_iter: Optional[int] = None,
    damping: Optional[float] = None,
    verbose: bool = True,
    newton_verbose: bool = False,
) -> Tuple[float, np.ndarray, float]:
    _require_complex_saddle(solver)

    if tol_newton is None:
        tol_newton = float(getattr(solver.settings, "newton_tol", 1e-9))
    if max_newton_iter is None:
        max_newton_iter = int(getattr(solver.settings, "newton_max_iter", 50))
    if damping is None:
        damping = float(getattr(solver.settings, "damping", 1.0))

    N_fields = int(solver.system_size_fields())
    anchor_which_n = _normalize_anchor_which(anchor_which)

    def _newton_cb(it, x, F, nF, alpha):
        if newton_verbose:
            lam = float(x[N_fields])
            print(f"        [Newton] it={it:02d}  ||F||={nF:.3e}  lambda={lam:+.3e}  alpha={alpha:.2e}")

    def solve_at(rho_pin: float, seed: np.ndarray) -> Tuple[NewtonResultAug, float]:
        res_aug, jac_aug, _ = make_pinned_system(
            solver, rho_pin=float(rho_pin), pin_site=pin_site, anchor_site=anchor_site, anchor_which=anchor_which_n
        )
        nr = newton_solve_aug(
            res_aug, jac_aug, seed,
            tol=float(tol_newton),
            max_iter=int(max_newton_iter),
            damping=float(damping),
            callback=_newton_cb if newton_verbose else None,
        )
        lam = float(nr.x[N_fields])
        return nr, lam

    nr_a, lam_a = solve_at(float(rho_a), np.asarray(x_aug_a, dtype=float))
    nr_b, lam_b = solve_at(float(rho_b), np.asarray(x_aug_b, dtype=float))

    if verbose:
        print(f"[Bisect] endpoints: rho_a={rho_a:.6g} lam_a={lam_a:+.3e}, rho_b={rho_b:.6g} lam_b={lam_b:+.3e}")

    if not (nr_a.success and nr_b.success):
        raise RuntimeError("Bisection endpoints did not converge; pick a bracket with successful points.")
    if lam_a == 0.0:
        return float(rho_a), nr_a.x.copy(), float(lam_a)
    if lam_b == 0.0:
        return float(rho_b), nr_b.x.copy(), float(lam_b)
    if lam_a * lam_b > 0:
        raise ValueError("Endpoints do not bracket lambda=0 (same sign).")

    lo_r, hi_r = float(rho_a), float(rho_b)
    lo_x, hi_x = nr_a.x.copy(), nr_b.x.copy()
    lo_l, hi_l = float(lam_a), float(lam_b)

    best_r, best_x, best_l = (lo_r, lo_x.copy(), lo_l) if abs(lo_l) < abs(hi_l) else (hi_r, hi_x.copy(), hi_l)

    for it in range(int(max_iter)):
        mid_r = 0.5 * (lo_r + hi_r)
        seed = lo_x if abs(lo_l) < abs(hi_l) else hi_x

        nr_m, lam_m = solve_at(mid_r, seed)
        if not nr_m.success:
            if abs(lo_l) < abs(hi_l):
                hi_r, hi_x, hi_l = mid_r, nr_m.x.copy(), float(lam_m)
            else:
                lo_r, lo_x, lo_l = mid_r, nr_m.x.copy(), float(lam_m)
            continue

        if abs(lam_m) < abs(best_l):
            best_r, best_x, best_l = mid_r, nr_m.x.copy(), float(lam_m)

        if verbose:
            print(f"[Bisect] it={it:02d} rho={mid_r:.6g} lam={lam_m:+.3e}  ||F||={nr_m.residual_norm:.2e}")

        if abs(lam_m) < float(tol_lam):
            return float(mid_r), nr_m.x.copy(), float(lam_m)

        if lo_l * lam_m <= 0:
            hi_r, hi_x, hi_l = mid_r, nr_m.x.copy(), float(lam_m)
        else:
            lo_r, lo_x, lo_l = mid_r, nr_m.x.copy(), float(lam_m)

    return float(best_r), best_x, float(best_l)


def verify_unpinned_solution(
    solver,
    *,
    x_fields_seed: np.ndarray,
    tol_residual: Optional[float] = None,
    diag_site: Tuple[int, int] = (4, 0),
    verbose: bool = True,
) -> Dict[str, Any]:
    _require_complex_saddle(solver)

    if tol_residual is None:
        tol_residual = 100.0 * float(getattr(solver.settings, "newton_tol", 1e-9))

    x_fields_seed = np.asarray(solver.to_fields_vector(x_fields_seed), dtype=float)

    F = np.asarray(solver.residual_fields(x_fields_seed), dtype=float)
    nF = float(np.linalg.norm(F))
    ok = (nF < float(tol_residual))

    pin_rho = None
    try:
        pin_rho = _rho_at_site_like_solver(solver, x_fields_seed, diag_site)
    except Exception:
        pass

    if verbose:
        print(f"[UnpinCheck] ||F||={nF:.3e}  -> {'PASS' if ok else 'FAIL'}  (tol={tol_residual:.3e})")

    return dict(ok=ok, residual_norm=nF, tol=float(tol_residual), rho_pin_diagnostic=pin_rho)


def existence_driver(
    solver,
    *,
    rho_pin_values: np.ndarray,
    x0_fields: np.ndarray,
    pin_site: Tuple[int, int] = (4, 0),
    anchor_site: Optional[Tuple[int, int]] = None,
    anchor_which: str = "y_re",
    homotopy_steps: int = 10,
    verbose: bool = True,
    newton_verbose: bool = False,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    damping: Optional[float] = None,
) -> Dict[str, Any]:
    scan = scan_pinned_branch(
        solver, rho_pin_values,
        x0_fields=x0_fields,
        pin_site=pin_site,
        anchor_site=anchor_site,
        anchor_which=anchor_which,
        homotopy_steps=homotopy_steps,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
        verbose=verbose,
        newton_verbose=newton_verbose,
    )

    out: Dict[str, Any] = dict(scan=scan, exists=False, refined=None)

    if not scan.zero_crossings:
        if verbose:
            print("[Existence] No lambda sign change found in scan -> no evidence for a bounce on this branch.")
        return out

    ia, ib = scan.zero_crossings[0]
    a = scan.points[ia]
    b = scan.points[ib]
    if not (a.success and b.success):
        return out

    rho_star, x_aug_star, lam_star = refine_lambda_zero_by_bisect(
        solver,
        rho_a=a.rho_pin, x_aug_a=a.x_aug,
        rho_b=b.rho_pin, x_aug_b=b.x_aug,
        pin_site=pin_site,
        anchor_site=anchor_site,
        anchor_which=anchor_which,
        tol_newton=tol,
        max_newton_iter=max_iter,
        damping=damping,
        verbose=verbose,
        newton_verbose=newton_verbose,
    )

    N_fields = int(solver.system_size_fields())
    x_fields_star = x_aug_star[:N_fields]

    unpin = verify_unpinned_solution(solver, x_fields_seed=x_fields_star, diag_site=pin_site, verbose=verbose)

    exists = bool(unpin["ok"]) and (abs(float(lam_star)) < 1e-6)
    out["exists"] = exists
    out["refined"] = dict(rho_star=float(rho_star), lam_star=float(lam_star), x_aug_star=x_aug_star, unpin=unpin)

    if verbose:
        if exists:
            print("[Existence] ✓ Found candidate bounce: lambda≈0 and unpinned residual small.")
        else:
            print("[Existence] ✗ Lambda≈0 bracket found but unpinned residual not small (or lambda not small).")

    return out
