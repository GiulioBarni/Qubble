# File: branching.py  (versione completa)
# NOTE: questo è esattamente quello scritto in /mnt/data/branching.py
"""
Bubble_finder.branching

Find a τ-dependent bounce starting from a sphaleron-like 2D solution.

Why this file exists
--------------------
In this project the 2D Newton solver can have a huge basin of attraction
for a given saddle/root (typically the sphaleron). Naively seeding
    x0 = x_sph ± ε v
and running Newton often reconverges to the same root.

To reliably leave the sphaleron root we solve an augmented system that pins
the solution at a fixed displacement along the negative-mode direction.

Augmented system (fields + 1 constraint)
---------------------------------------
Let F(x)=0 be the packed field equations solved by Newton.
Let v_- be a negative-mode direction around x_sph.

We solve
    R(x) = [ F(x),  <x-x_sph, v_-> - c ] = 0
for fixed c = ±eps.
This removes the sphaleron root (for c != 0) and forces Newton onto a different
branch if one exists.

Public API
----------
- slicewise_energies(solver, sol) -> (tau_sorted, E_static, E_full)
- slicewise_energies_at_tau(solver, y, ybar, tau_abscissa) -> (tau_abscissa, E_static, E_full)
- recenter_solution_in_tau(solver, sol, ...) -> (y_new, ybar_new, info)
- make_centered_tau_and_roll(solver, y, ybar, tau_center=0) -> (tau_plot, y2, ybar2, info)
- find_bounce_from_sphaleron(...) -> (best_bounce_or_None, diagnostics)

The implementation is intentionally minimal and reuses:
  Bubble2DSolver.residual/jacobian
  Bubble2DSolver.newton_solve_aug
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import observables_2d
from .bounce2d import Bubble2DSolver, Bubble2DSolution, NewtonConvergenceError


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _as_float1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()

def _xdist(xa: np.ndarray, xb: np.ndarray) -> float:
    d = _as_float1d(xa) - _as_float1d(xb)
    return float(np.sqrt(np.vdot(d, d).real))

def _tau_sort_index(tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau).flatten()
    if np.all(np.diff(tau) > 0):
        idx = np.arange(len(tau))
        return idx, tau
    idx = np.argsort(tau)
    return idx, tau[idx]

def _wrap_shift(shift: int, n: int) -> int:
    return int(shift % n)


# -----------------------------------------------------------------------------
# Slicewise energies
# -----------------------------------------------------------------------------

def slicewise_energies(
    solver: Bubble2DSolver,
    sol: Bubble2DSolution,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tau_sorted, E_static(tau), E_full(tau)).

    Conventions match the notebook diagnostics:
      E_static(τ) = 4π ∫ dr r^2 [ (∂r φ)(∂r φ̄) + U(ρ) ]
      E_full(τ)   = 4π ∫ dr r^2 [ (∂τ φ)(∂τ φ̄) + (∂r φ)(∂r φ̄) + U(ρ) ]

    τ is returned sorted ascending (needed for np.gradient).
    """
    r = np.asarray(solver.grid.r).flatten()
    tau = np.asarray(solver.grid.tau).flatten()

    if not np.all(np.diff(r) > 0):
        raise ValueError("r grid must be strictly increasing")

    sort_idx, tau_sorted = _tau_sort_index(tau)

    phi_nt, phibar_nt = solver.phi(sol.y, sol.ybar)  # (Nr,Nt)
    phi_nt = phi_nt[:, sort_idx]
    phibar_nt = phibar_nt[:, sort_idx]

    phi_t = np.asarray(phi_nt.T)       # (Nt,Nr)
    phibar_t = np.asarray(phibar_nt.T) # (Nt,Nr)

    dphi_dtau = np.gradient(phi_t, tau_sorted, axis=0, edge_order=2)
    dphibar_dtau = np.gradient(phibar_t, tau_sorted, axis=0, edge_order=2)
    dphi_dr = np.gradient(phi_t, r, axis=1, edge_order=2)
    dphibar_dr = np.gradient(phibar_t, r, axis=1, edge_order=2)

    tau_kin = (dphi_dtau * dphibar_dtau).real
    spatial = (dphi_dr * dphibar_dr).real
    s = (phi_t * phibar_t).real
    V_term = solver.U(np.sqrt(np.maximum(s, 0.0)))

    pref = 4.0 * np.pi
    E_static = pref * np.trapz((spatial + V_term) * r**2, r, axis=1)
    E_full = pref * np.trapz((tau_kin + spatial + V_term) * r**2, r, axis=1)
    return tau_sorted, E_static, E_full


def slicewise_energies_at_tau(
    solver: Bubble2DSolver,
    y: np.ndarray,
    ybar: np.ndarray,
    tau_abscissa: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E_static and E_full along index order using tau_abscissa for d/dτ. Returns (tau_abscissa, E_static, E_full). No sorting."""
    r = np.asarray(solver.grid.r).flatten()
    tau_abscissa = np.asarray(tau_abscissa, dtype=float).flatten()
    if not np.all(np.diff(r) > 0):
        raise ValueError("r grid must be strictly increasing")
    phi_nt, phibar_nt = solver.phi(y, ybar)  # (Nr,Nt)
    phi_t = np.asarray(phi_nt.T)
    phibar_t = np.asarray(phibar_nt.T)
    dphi_dtau = np.gradient(phi_t, tau_abscissa, axis=0, edge_order=2)
    dphibar_dtau = np.gradient(phibar_t, tau_abscissa, axis=0, edge_order=2)
    dphi_dr = np.gradient(phi_t, r, axis=1, edge_order=2)
    dphibar_dr = np.gradient(phibar_t, r, axis=1, edge_order=2)
    tau_kin = (dphi_dtau * dphibar_dtau).real
    spatial = (dphi_dr * dphibar_dr).real
    s = (phi_t * phibar_t).real
    V_term = solver.U(np.sqrt(np.maximum(s, 0.0)))
    pref = 4.0 * np.pi
    E_static = pref * np.trapz((spatial + V_term) * r**2, r, axis=1)
    E_full = pref * np.trapz((tau_kin + spatial + V_term) * r**2, r, axis=1)
    return tau_abscissa, E_static, E_full


# -----------------------------------------------------------------------------
# Recentering in τ (original index space, only np.roll on axis=1)
# -----------------------------------------------------------------------------

def recenter_solution_in_tau(
    solver: Bubble2DSolver,
    sol: Bubble2DSolution,
    tau_turn_target: Optional[float] = None,
    method: str = "rho_peak",
    r_probe_index: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Roll (y,ybar) along τ in original index space so the peak lands at tau_turn_target.

    Uses only np.roll on axis=1; no sort_idx/inv. Supports method="rho_peak", "Estatic_min", "Estatic_max".
    """
    tau = np.asarray(solver.grid.tau).flatten()
    Nt = len(tau)
    if tau_turn_target is None:
        tau_turn_target = 0.5 * (float(tau.min()) + float(tau.max()))
    tau_turn_target = float(tau_turn_target)
    it_target = int(np.argmin(np.abs(tau - tau_turn_target)))

    if method == "rho_peak":
        rho = solver.rho_map(sol.y, sol.ybar)  # (Nr,Nt) original order
        j = int(np.clip(r_probe_index, 0, rho.shape[0] - 1))
        rho_j = np.asarray(rho[j, :]).flatten()
        dev = np.abs(rho_j - float(np.mean(rho_j)))
        it_peak = int(np.argmax(dev))
    elif method in ("Estatic_min", "Estatic_max"):
        sort_idx, tau_sorted = _tau_sort_index(tau)
        _, E_static, _ = slicewise_energies(solver, sol)
        i_in_sorted = int(np.argmin(E_static) if method == "Estatic_min" else np.argmax(E_static))
        it_peak = int(sort_idx[i_in_sorted])
    else:
        raise ValueError("Unknown recenter method")

    shift = _wrap_shift(it_target - it_peak, Nt)
    if shift == 0:
        return sol.y.copy(), sol.ybar.copy(), {
            "shift": 0,
            "tau_target": tau_turn_target,
            "tau_peak": float(tau[it_peak]),
            "method": method,
        }
    y_new = np.roll(sol.y.copy(), shift, axis=1)
    yb_new = np.roll(sol.ybar.copy(), shift, axis=1)
    return y_new, yb_new, {
        "shift": int(shift),
        "tau_target": tau_turn_target,
        "tau_peak": float(tau[it_peak]),
        "method": method,
    }


def make_centered_tau_and_roll(
    solver: Bubble2DSolver,
    y: np.ndarray,
    ybar: np.ndarray,
    tau_center: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Roll y,ybar so tau_center is at index Nt//2; build centered tau_plot = (arange(Nt)-Nt//2)*dt, dt=beta/(Nt-1). Returns (tau_plot, y2, ybar2, info)."""
    tau = np.asarray(solver.grid.tau).flatten()
    Nt = len(tau)
    beta = float(getattr(solver.grid, "beta", 2.0 * np.abs(np.ptp(tau))))
    dt = beta / max(Nt - 1, 1)
    it_center_now = int(np.argmin(np.abs(tau - tau_center)))
    shift = _wrap_shift(Nt // 2 - it_center_now, Nt)
    y2 = np.roll(np.asarray(y, dtype=complex).copy(), shift, axis=1)
    ybar2 = np.roll(np.asarray(ybar, dtype=complex).copy(), shift, axis=1)
    tau_plot = (np.arange(Nt, dtype=float) - Nt // 2) * dt
    info = {"shift": shift, "tau_center": tau_center, "beta": beta, "dt": dt}
    return tau_plot, y2, ybar2, info


# -----------------------------------------------------------------------------
# Negative mode
# -----------------------------------------------------------------------------

def _eig_negative_mode_from_sparse_symmetric(H: sp.spmatrix) -> Tuple[np.ndarray, float]:
    vals, vecs = spla.eigsh(H, k=1, which="SA")
    lam = float(vals[0])
    v = np.asarray(vecs[:, 0]).ravel().real
    nv = float(np.linalg.norm(v))
    if nv < 1e-15:
        raise ValueError("Zero eigenvector")
    v /= nv
    imax = int(np.argmax(np.abs(v)))
    if v[imax] < 0:
        v = -v
    return v, lam

def _eig_negative_mode_from_sparse_nonsym(J: sp.spmatrix, k: int = 5) -> Tuple[np.ndarray, float]:
    n = J.shape[0]
    kk = min(max(1, k), max(1, n - 2))
    vals, vecs = spla.eigs(J, k=kk, which="SR")
    idx = int(np.argmin(vals.real))
    lam = float(vals[idx].real)
    v = np.asarray(vecs[:, idx]).ravel().real
    nv = float(np.linalg.norm(v))
    if nv < 1e-15:
        raise ValueError("Zero eigenvector")
    v /= nv
    imax = int(np.argmax(np.abs(v)))
    if v[imax] < 0:
        v = -v
    return v, lam

def get_negative_mode(
    solver: Bubble2DSolver,
    x_sph: np.ndarray,
    prefer_symmetric: bool = True,
    k: int = 5,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Return (v_neg, lambda_neg, info).

    Default: use symmetric part H=(J+J^T)/2 and take its most negative eigenpair.
    Fallback: use eigs(J) and pick smallest real part.
    """
    info: Dict[str, Any] = {}
    J = solver.jacobian(_as_float1d(x_sph))
    if not sp.issparse(J):
        J = sp.csc_matrix(J)

    if prefer_symmetric:
        H = (J + J.T) * 0.5
        try:
            nJ = float(spla.norm(J))
            nAs = float(spla.norm(J - J.T))
            info["asym_ratio"] = (nAs / nJ) if nJ > 0 else float("nan")
        except Exception:
            info["asym_ratio"] = float("nan")
        try:
            v, lam = _eig_negative_mode_from_sparse_symmetric(H)
            info["mode"] = "symmetric"
            if verbose:
                print(f"  [branching] negative mode (sym): lambda_min = {lam:.6e}, asym_ratio={info['asym_ratio']:.3e}")
            return v, lam, info
        except Exception as e:
            info["symmetric_failed"] = str(e)
            if verbose:
                print(f"  [branching] symmetric negative mode failed ({e}); fallback to eigs(J).")

    v, lam = _eig_negative_mode_from_sparse_nonsym(J, k=k)
    info["mode"] = "nonsym"
    if verbose:
        print(f"  [branching] negative mode (nonsym): min(real) = {lam:.6e}")
    return v, lam, info


# -----------------------------------------------------------------------------
# Augmented pinned solve
# -----------------------------------------------------------------------------

@dataclass
class _AugSolveOut:
    converged: bool
    x_fields: Optional[np.ndarray]
    newton: Any
    message: str

def _solve_pinned_along_mode(
    solver: Bubble2DSolver,
    x_sph: np.ndarray,
    v: np.ndarray,
    c: float,
    x0_fields: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 35,
    damping: float = 1.0,
    verbose: bool = False,
) -> _AugSolveOut:
    """Solve F(x)=0 with constraint <x-x_sph, v>=c using newton_solve_aug."""
    x_sph = _as_float1d(x_sph)
    v = _as_float1d(v)
    v /= float(np.linalg.norm(v) + 1e-30)
    n = x_sph.size

    if x0_fields is None:
        x0_fields = x_sph + c * v
    x0_fields = _as_float1d(x0_fields)

    def residual_aug(x_aug: np.ndarray) -> np.ndarray:
        x_aug = _as_float1d(x_aug)
        x = x_aug[:n]
        lam = float(x_aug[n])
        # KKT: lambda enters field equations along v
        F = _as_float1d(solver.residual_fields(x))
        F_lam = F + lam * v
        g = float(np.dot(x - x_sph, v) - c)
        return np.concatenate([F_lam, np.array([g], dtype=float)])

    def jacobian_aug(x_aug: np.ndarray) -> sp.csc_matrix:
        x_aug = _as_float1d(x_aug)
        x = x_aug[:n]
        J = solver.jacobian_fields(x)
        if not sp.issparse(J):
            J = sp.csc_matrix(J)
        # KKT: [[J(x), v], [v^T, 0]] — top-right column = v
        col_v = sp.csc_matrix(v.reshape(-1, 1))
        top = sp.hstack([J, col_v], format="csc")
        bot = sp.hstack([sp.csc_matrix(v.reshape(1, -1)), sp.csc_matrix((1, 1), dtype=float)], format="csc")
        return sp.vstack([top, bot], format="csc")

    x_aug0 = np.concatenate([x0_fields, np.array([0.0], dtype=float)])

    try:
        newton = solver.newton_solve_aug(
            residual_aug,
            jacobian_aug,
            x_aug0,
            tol=tol,
            max_iter=max_iter,
            damping=0.5,
        )
        ok = bool(getattr(newton, "success", False))
        x_out = _as_float1d(newton.x)[:n]
        return _AugSolveOut(converged=ok, x_fields=x_out if ok else None, newton=newton, message="ok" if ok else "not_converged")
    except Exception as e:
        if verbose:
            print(f"  [branching] pinned solve failed: {e}")
        return _AugSolveOut(converged=False, x_fields=None, newton=None, message=str(e))


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def find_bounce_from_sphaleron(
    solver: Bubble2DSolver,
    sol_sph: Bubble2DSolution,
    eps_list: Optional[np.ndarray] = None,
    max_tries: int = 40,
    tau_turn: Optional[float] = None,
    verbose: bool = True,
    solve_verbose: bool = False,
    amp_tol: float = 1e-3,
    dist_tol_sph: float = 1e-6,
    dist_tol_banal: float = 1e-6,
    prefer_symmetric_mode: bool = True,
    recenter_method: str = "Estatic_min",
    release_after_pin: bool = True,
    pinned_tol: float = 1e-9,
    pinned_max_iter: int = 35,
) -> Tuple[Optional[Bubble2DSolution], Dict[str, Any]]:
    """Find a bounce from a sphaleron-like solution using a pinned negative-mode scan."""

    diag: Dict[str, Any] = {
        "lambda_neg": None,
        "v_neg_norm": None,
        "mode_info": None,
        "x_sph": None,
        "x_banal": None,
        "dist_sph_banal": None,
        "table": [],
        "candidates": [],
        "best": None,
        "best_bounce": None,
        "plot": {},
    }

    r = np.asarray(solver.grid.r).flatten()
    tau = np.asarray(solver.grid.tau).flatten()
    Nr, Nt = len(r), len(tau)

    if tau_turn is None:
        tau_turn = 0.0
    tau_turn = float(tau_turn)

    # packed sphaleron
    x_sph = _as_float1d(solver.pack(sol_sph.y, sol_sph.ybar))
    diag["x_sph"] = x_sph

    # banal reference from homogeneous guess
    y0 = np.zeros((Nr, Nt), dtype=complex)
    x_hom = solver.pack(y0, y0.copy())
    try:
        sol_banal = solver.solve(x_hom, verbose=False, verbose_success_block=False)
        x_banal = _as_float1d(solver.pack(sol_banal.y, sol_banal.ybar))
    except Exception:
        x_banal = _as_float1d(x_hom)
    diag["x_banal"] = x_banal
    diag["dist_sph_banal"] = _xdist(x_sph, x_banal)

    # negative mode
    v_neg, lambda_neg, mode_info = get_negative_mode(
        solver,
        x_sph,
        prefer_symmetric=prefer_symmetric_mode,
        k=5,
        verbose=verbose,
    )
    diag["lambda_neg"] = float(lambda_neg)
    diag["v_neg_norm"] = float(np.linalg.norm(v_neg))
    diag["mode_info"] = mode_info

    if eps_list is None:
        eps_list = np.logspace(-6, -1, 12)
    eps_list = np.atleast_1d(np.asarray(eps_list, dtype=float))

    r0_idx = min(1, Nr - 1)

    if verbose:
        print("\n--- find_bounce_from_sphaleron: pinned negative-mode scan ---")
        print(f"  tau_turn = {tau_turn:.4f}")
        print(f"  dist(x_sph, x_banal) = {diag['dist_sph_banal']:.6e}")
        print(f"  eps_list: {eps_list.tolist()}")
        print(f"  amp_tol={amp_tol:g} dist_tol_sph={dist_tol_sph:g} dist_tol_banal={dist_tol_banal:g}")
        print(f"{'eps':>10} {'sgn':>3} {'pin':>4} {'rel':>4} {'amp_tau':>10} {'dist_sph':>10} {'dist_banal':>10} {'dE_static':>10} {'curv@0':>10} {'bounce':>6}")
        print("-" * 95)

    n_done = 0
    for sgn in (+1.0, -1.0):
        for eps in eps_list:
            if n_done >= int(max_tries):
                break
            n_done += 1

            c = float(sgn * eps)
            pin_out = _solve_pinned_along_mode(
                solver,
                x_sph=x_sph,
                v=v_neg,
                c=c,
                x0_fields=x_sph + c * v_neg,
                tol=pinned_tol,
                max_iter=pinned_max_iter,
                damping=1.0,
                verbose=solve_verbose,
            )

            rec: Dict[str, Any] = {
                "eps": float(eps),
                "sign": "+" if sgn > 0 else "-",
                "pinned_ok": bool(pin_out.converged),
                "released_ok": False,
                "sol": None,
                "pin_newton": pin_out.newton,
            }

            if not pin_out.converged or pin_out.x_fields is None:
                diag["table"].append(rec)
                if verbose:
                    print(f"{eps:>10.2e} {rec['sign']:>3} {'fail':>4} {'--':>4} {float('nan'):>10.4e} {float('nan'):>10.4e} {float('nan'):>10.4e} {float('nan'):>10.4e} {float('nan'):>10.4e} {str(False):>6}")
                continue

            # release or use pinned solution
            sol = None
            ok = False
            if release_after_pin:
                try:
                    sol = solver.solve(pin_out.x_fields, verbose=solve_verbose, verbose_success_block=False)
                    ok = bool(getattr(sol, "success", True))
                except (NewtonConvergenceError, RuntimeError, Exception):
                    sol = None
                    ok = False
            else:
                y_pin, ybar_pin = solver.unpack(pin_out.x_fields)
                sol = Bubble2DSolution(
                    settings=sol_sph.settings,
                    grid=sol_sph.grid,
                    newton=pin_out.newton,
                    y=y_pin,
                    ybar=ybar_pin,
                    rho0=sol_sph.rho0,
                    Q_tau0=getattr(sol_sph, "Q_tau0", 0.0),
                    E_tau0=getattr(sol_sph, "E_tau0", 0.0),
                    sanity=getattr(sol_sph, "sanity", {}),
                    iteration_history=getattr(sol_sph, "iteration_history", None),
                    observables_ghost=getattr(sol_sph, "observables_ghost", None),
                    E_hom=getattr(sol_sph, "E_hom", 0.0),
                    energy_ratio=getattr(sol_sph, "energy_ratio", 0.0),
                )
                ok = True
            rec["released_ok"] = bool(ok)
            rec["sol"] = sol
            rec["sol_pinned"] = sol if not release_after_pin else None
            if not ok or sol is None:
                diag["table"].append(rec)
                if verbose:
                    print(f"{eps:>10.2e} {rec['sign']:>3} {'ok':>4} {'fail':>4} {float('nan'):>10.4e} {float('nan'):>10.4e} {float('nan'):>10.4e} {float('nan'):>10.4e} {float('nan'):>10.4e} {str(False):>6}")
                continue

            # recenter in original index space
            y_rc, yb_rc, rc_info = recenter_solution_in_tau(
                solver,
                sol,
                tau_turn_target=tau_turn,
                method=recenter_method,
                r_probe_index=r0_idx,
            )
            # centered roll: tau_center at Nt//2, tau_plot = (arange(Nt)-Nt//2)*dt
            tau_plot, y2, ybar2, info_roll = make_centered_tau_and_roll(solver, y_rc, yb_rc, tau_center=tau_turn)
            sol_rc = Bubble2DSolution(
                settings=sol.settings,
                grid=sol.grid,
                newton=sol.newton,
                y=y2,
                ybar=ybar2,
                rho0=sol.rho0,
                Q_tau0=sol.Q_tau0,
                E_tau0=sol.E_tau0,
                sanity=sol.sanity,
                iteration_history=sol.iteration_history,
                observables_ghost=sol.observables_ghost,
                E_hom=sol.E_hom,
                energy_ratio=sol.energy_ratio,
            )

            x_sol = _as_float1d(solver.pack(sol_rc.y, sol_rc.ybar))
            dist_sph = _xdist(x_sol, x_sph)
            dist_banal = _xdist(x_sol, x_banal)

            rho = solver.rho_map(y2, ybar2)
            i_center = Nt // 2
            amp_tau = float(np.max(np.abs(rho[r0_idx, :] - rho[r0_idx, i_center])))

            tau_plot, E_static, E_full = slicewise_energies_at_tau(solver, y2, ybar2, tau_plot)
            dE = float(np.max(E_static) - np.min(E_static))
            i_min = int(np.argmin(E_static))
            i_max = int(np.argmax(E_static))
            tmin = float(tau_plot[i_min])
            tmax = float(tau_plot[i_max])
            if 1 <= i_center < Nt - 1:
                curv = float(E_static[i_center + 1] - 2.0 * E_static[i_center] + E_static[i_center - 1])
                bounce_like = curv > 0.0
                sphaleron_like = curv < 0.0
            else:
                curv = float("nan")
                bounce_like = False
                sphaleron_like = False

            try:
                Qg = float(observables_2d.compute_charge_tau0_ghost_2d(solver, sol_rc.y, sol_rc.ybar, subtract_background=False))
                Em = float(observables_2d.compute_energy_minkowski_tau0_ghost_2d(solver, sol_rc.y, sol_rc.ybar))
            except Exception:
                Qg = float("nan")
                Em = float("nan")

            rec.update({
                "dist_sph": float(dist_sph),
                "dist_banal": float(dist_banal),
                "amp_tau": float(amp_tau),
                "delta_E_static": float(dE),
                "where_min_tau": tmin,
                "where_max_tau": tmax,
                "turning_curv": curv,
                "bounce_like": bool(bounce_like),
                "sphaleron_like": bool(sphaleron_like),
                "E_static_tau": E_static,
                "E_full_tau": E_full,
                "tau_sorted": tau_plot,
                "rho_map": rho,
                "Q_ghost": Qg,
                "E_M_ghost": Em,
                "recenter": rc_info,
                "centered_roll": info_roll,
                "sol_recentered": sol_rc,
            })
            diag["table"].append(rec)

            is_tau_dep = (amp_tau > amp_tol) and (dE > 0.0)
            not_sph = (dist_sph > dist_tol_sph)
            not_banal = (dist_banal > dist_tol_banal)
            if is_tau_dep and not_sph and not_banal and bounce_like:
                diag["candidates"].append(rec)

            if verbose:
                print(f"{eps:>10.2e} {rec['sign']:>3} {'ok':>4} {'ok':>4} {amp_tau:>10.4e} {dist_sph:>10.4e} {dist_banal:>10.4e} {dE:>10.4e} {curv:>10.4e} {str(bounce_like):>6}")

    best: Optional[Dict[str, Any]] = None
    if diag["candidates"]:
        def keyfun(rr: Dict[str, Any]):
            Emin = float(np.min(rr["E_static_tau"])) if rr.get("E_static_tau") is not None else float("inf")
            return (rr.get("amp_tau", 0.0), -Emin)
        best = max(diag["candidates"], key=keyfun)
        diag["best"] = {"type": "bounce_candidate", "eps": best["eps"], "sign": best["sign"]}
        diag["best_bounce"] = best["sol_recentered"]
    else:
        fallback = [
            rr for rr in diag["table"]
            if rr.get("released_ok") and rr.get("amp_tau", 0.0) > amp_tol and rr.get("dist_banal", 0.0) > dist_tol_banal
        ]
        if fallback:
            best = max(fallback, key=lambda rr: rr.get("amp_tau", 0.0))
            diag["best"] = {"type": "tau_dependent_fallback", "eps": best.get("eps"), "sign": best.get("sign")}
            diag["best_bounce"] = None
        else:
            diag["best"] = None
            diag["best_bounce"] = None

    if best is not None:
        diag["plot"] = {
            "tau_sorted": best.get("tau_sorted"),
            "E_static": best.get("E_static_tau"),
            "E_full": best.get("E_full_tau"),
            "rho": best.get("rho_map"),
            "r": r,
            "tau": tau,
            "tau_turn": tau_turn,
            "best_type": diag["best"]["type"] if diag.get("best") else None,
        }

    if verbose:
        n_cand = len(diag["candidates"])
        print(f"  Candidates (τ-dep, not banal, bounce_like curv>0 at center): {n_cand}")
        if diag["best"] is None:
            print("  No candidate selected.")
        else:
            print(f"  Best selected ({diag['best']['type']}): eps={diag['best']['eps']:.2e}, sign={diag['best']['sign']}")
            if diag["best_bounce"] is None:
                print("  No bounce candidate found; returning fallback only (see diag['plot']).")
            else:
                print("  Best bounce solution stored in best_bounce; diagnostics in diag.")

    return diag["best_bounce"], diag