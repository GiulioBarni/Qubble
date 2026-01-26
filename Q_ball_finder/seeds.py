"""
Seed selection based on short Newton basin test with multi-term scoring.

This module provides seed selection that evaluates candidates by running
a short Newton iteration (basin test) and scoring with multiple criteria:
- Residual norm (secondary)
- Plateau enforcement (rho0 ≈ target)
- Cloud-like rejection (tau-dependence)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .ansatz import build_qball_escape_ansatz, AnsatzResult
from .bounce2d import QBall2DSolver, QBall2DSettings
from .grid import RadialTimeGrid, pack_fields, unpack_fields, phi_from_y, phi_from_ybar
from .nr_solver import newton_solve, NewtonConvergenceError
from .observables2d import compute_charge
from .potentials import LogisticPotentialParams
from .profiles import QBallProfile, UnstableMode


@dataclass
class BasinTestResult:
    """Result of basin test seed selection."""
    best_x0: np.ndarray
    best_params: dict
    best_score: float
    best_residual_norm: float
    best_rho0: float
    best_tau_variation: float
    best_charge_ratio: float
    table: list[dict]  # each entry: params + metrics


def select_best_qball_escape_seed(
    *,
    params: LogisticPotentialParams,
    omega_qball: float,
    qball_profile: QBallProfile,
    mode: UnstableMode,
    grid: RadialTimeGrid,
    settings: QBall2DSettings,
    eta: float,
    target_charge: float,
    # scan ranges:
    a_plateaus: Tuple[float, ...] = (0.2, 0.3, 0.35, 0.4),
    tau_transitions: Tuple[float, ...] = (-12.0, -15.0, -18.0, -21.0),
    tau_widths: Tuple[float, ...] = (2.0, 3.0, 4.0),
    kick_amps: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    kick_thetas: Tuple[float, ...] = (0.0, np.pi/2, np.pi, 3*np.pi/2),
    kick_tau_widths: Tuple[float, ...] = (4.0, 6.0, 8.0),
    cosh_scales: Tuple[Optional[float], ...] = (None, 3.0),
    flip_signs: Tuple[bool, ...] = (False, True),
    # scoring parameters:
    rho0_target: float = 0.5,
    tau_variation_min: float = 0.2,
    w_rho0: float = 10.0,
    w_cloud: float = 100.0,
    w_Q: float = 2.0,
    # basin test:
    basin_newton_iters: int = 5,
    basin_damping: float = 0.5,
    verbose: bool = False,
) -> BasinTestResult:
    """
    Select the best Q-ball escape ansatz seed using a short Newton basin test
    with multi-term scoring.
    
    Scoring (lower is better):
      score = log(1 + ||F||) + w_rho0 * |rho0 - rho0_target|/rho0_target
              + w_cloud * pen_cloud + w_Q * |Q_ratio - 1|
    
    Where:
      - ||F|| is residual norm after basin test (secondary criterion)
      - rho0 is max(rho) at τ≈0 slice (enforce ≈ rho0_target)
      - pen_cloud penalizes cloud-like (too τ-independent)
      - Q_ratio is charge ratio vs target
    
    Parameters
    ----------
    params
        Model parameters.
    omega_qball
        Q-ball frequency.
    qball_profile
        Q-ball profile for the ansatz (NOT Q-cloud).
    mode
        Unstable mode for kick.
    grid
        Radial-time grid.
    settings
        Solver settings.
    eta
        Chemical potential parameter.
    target_charge
        Target charge to match.
    a_plateaus
        Grid of a_plateau values (target ρ(τ≈0)).
    tau_transitions
        Grid of tau_transition values.
    tau_widths
        Grid of tau_width values.
    kick_amps
        Grid of kick_amp values.
    kick_thetas
        Grid of kick_theta values.
    kick_tau_widths
        Grid of kick_tau_width values.
    cosh_scales
        Grid of cosh_scale values.
    flip_signs
        Grid of flip_sign boolean values.
    rho0_target
        Target max(rho) at τ≈0 (default 0.5).
    tau_variation_min
        Minimum tau variation to avoid cloud-like (default 0.2).
    w_rho0
        Weight for rho0 penalty (default 10.0).
    w_cloud
        Weight for cloud-like penalty (default 100.0).
    w_Q
        Weight for charge mismatch (default 2.0).
    basin_newton_iters
        Number of Newton iterations for basin test (default 5).
    basin_damping
        Damping factor for basin test (default 0.5).
    verbose
        Print progress and results.
    
    Returns
    -------
    BasinTestResult
        Contains best_x0, best_params, best_score, and metrics.
    """
    # Update settings with eta
    from dataclasses import replace
    settings_with_eta = replace(settings, eta0=eta)
    
    # Initialize solver once
    solver = QBall2DSolver(params, omega_qball, settings_with_eta, grid)
    
    # Prepare parameter combinations
    candidates = []
    for a_plateau in a_plateaus:
        for tau_transition in tau_transitions:
            for tau_width in tau_widths:
                for kick_amp in kick_amps:
                    for kick_theta in kick_thetas:
                        for kick_tau_width in kick_tau_widths:
                            for cosh_scale in cosh_scales:
                                for flip_sign in flip_signs:
                                    candidates.append({
                                        "a_plateau": a_plateau,
                                        "tau_transition": tau_transition,
                                        "tau_width": tau_width,
                                        "kick_amp": kick_amp,
                                        "kick_theta": kick_theta,
                                        "kick_tau_width": kick_tau_width,
                                        "cosh_scale": cosh_scale,
                                        "flip_sign": flip_sign,
                                    })
    
    total = len(candidates)
    if verbose:
        print(f"Evaluating {total} Q-ball escape seed candidates with basin test...")
        print(f"  Basin test: {basin_newton_iters} iterations, damping={basin_damping}")
        print(f"  Scoring: rho0_target={rho0_target:.3f}, tau_variation_min={tau_variation_min:.3f}")
        print(f"  Weights: w_rho0={w_rho0:.1f}, w_cloud={w_cloud:.1f}, w_Q={w_Q:.1f}")
    
    quick_results = []
    if verbose:
        print(f"Step 1: Quick residual evaluation for {total} candidates...")
    
    for idx, cand in enumerate(candidates):
        try:
            ansatz = build_qball_escape_ansatz(
                qball_profile=qball_profile,
                mode=mode,
                grid=grid,
                omega_reference=omega_qball,
                a_plateau=cand["a_plateau"],
                tau_transition=cand["tau_transition"],
                tau_width=cand["tau_width"],
                kick_amp=cand["kick_amp"],
                kick_theta=cand["kick_theta"],
                kick_tau_width=cand["kick_tau_width"],
                cosh_scale=cand["cosh_scale"],
                flip_sign=cand["flip_sign"],
                omega_tilde=settings.omega_tilde,
                decrease_towards_zero=False,
            )
            
            x0 = pack_fields(ansatz.y, ansatz.ybar)
            
            F = solver.residual(x0)
            residual_norm = np.linalg.norm(F.ravel())
            
            if not np.isfinite(residual_norm):
                residual_norm = np.inf
            
            quick_results.append({
                "candidate": cand,
                "x0": x0,
                "residual_norm": residual_norm,
            })
            
            if verbose and (idx + 1) % 20 == 0:
                print(f"  Progress: {idx + 1}/{total}")
        
        except Exception as e:
            quick_results.append({
                "candidate": cand,
                "x0": None,
                "residual_norm": np.inf,
                "error": str(e),
            })
            if verbose:
                print(f"  Candidate {idx + 1} failed: {e}")
    
    quick_results_sorted = sorted(quick_results, key=lambda r: r["residual_norm"])
    top5_candidates = quick_results_sorted[:5]
    
    if verbose:
        print(f"\nStep 2: Basin test for top {len(top5_candidates)} candidates (by residual norm)...")
        top5_norms = [r['residual_norm'] for r in top5_candidates[:5]]
        print(f"  Top 5 residual norms: {[f'{n:.6e}' for n in top5_norms]}")
    
    results = []
    for idx, quick_res in enumerate(top5_candidates):
        cand = quick_res["candidate"]
        x0 = quick_res["x0"]
        
        if x0 is None:
            continue
        
        try:
            newton_res = newton_solve(
                residual=lambda vec: solver.residual(vec),
                jacobian=lambda vec: solver.jacobian(vec),
                x0=x0,
                tol=1e-8,
                max_iter=basin_newton_iters,
                damping=basin_damping,
                norm=lambda v: np.linalg.norm(v.ravel()),
                verbose=False,
            )
            
            y_final, ybar_final = unpack_fields(newton_res.x, grid.Nr, grid.Ntau)
            phi_final = phi_from_y(y_final, grid, omega_qball)
            phibar_final = phi_from_ybar(ybar_final, grid, omega_qball)
            
            residual_norm_final = newton_res.residual_norm
            
            rho_2d_final = np.sqrt(np.maximum((phi_final * phibar_final).real, 0.0))
            
            i0 = 0
            i_min = grid.Ntau - 1
            
            rho0 = np.max(rho_2d_final[:, i0])
            rho_min = np.max(rho_2d_final[:, i_min])
            
            tau_variation = np.linalg.norm(rho_2d_final[:, i0] - rho_2d_final[:, i_min]) / max(rho_min, 1e-10)
            
            charge_final = solver.compute_charge(y_final, ybar_final)
            Q_ratio = charge_final / target_charge if target_charge > 0 else 0.0
            
            score_F = np.log(1.0 + residual_norm_final)
            
            pen_rho0 = abs(rho0 - rho0_target) / rho0_target if rho0_target > 0 else abs(rho0 - rho0_target)
            
            pen_cloud = 0.0
            if tau_variation < tau_variation_min:
                pen_cloud = 1.0 - (tau_variation / tau_variation_min)
            
            pen_Q = abs(Q_ratio - 1.0)
            
            score = score_F + w_rho0 * pen_rho0 + w_cloud * pen_cloud + w_Q * pen_Q
            
            result_entry = {
                "a_plateau": cand["a_plateau"],
                "tau_transition": cand["tau_transition"],
                "tau_width": cand["tau_width"],
                "kick_amp": cand["kick_amp"],
                "kick_theta": cand["kick_theta"],
                "kick_tau_width": cand["kick_tau_width"],
                "cosh_scale": cand["cosh_scale"],
                "flip_sign": cand["flip_sign"],
                "residual_norm": residual_norm_final,
                "rho0": rho0,
                "tau_variation": tau_variation,
                "charge": charge_final,
                "Q_ratio": Q_ratio,
                "score": score,
                "x0": x0,  # Store initial x0
            }
            results.append(result_entry)
            
            if verbose:
                print(f"  Top {idx+1}: ||F||={residual_norm_final:.6e}, rho0={rho0:.4f}, tau_var={tau_variation:.4f}, score={score:.6e}")
        
        except NewtonConvergenceError as e:
            # Solution is not converging, exit all scans immediately
            if verbose:
                print(f"\n[seed-selection] Convergence error at candidate {idx+1}: {e}")
                print("[seed-selection] Exiting all scans - solution does not converge")
            raise RuntimeError(
                f"Seed selection aborted: solution does not converge. "
                f"Original error: {e}"
            ) from e
        
        except Exception as e:
            result_entry = {
                "a_plateau": cand["a_plateau"],
                "tau_transition": cand["tau_transition"],
                "tau_width": cand["tau_width"],
                "kick_amp": cand["kick_amp"],
                "kick_theta": cand["kick_theta"],
                "kick_tau_width": cand["kick_tau_width"],
                "cosh_scale": cand["cosh_scale"],
                "flip_sign": cand["flip_sign"],
                "residual_norm": np.inf,
                "rho0": 0.0,
                "tau_variation": 0.0,
                "charge": 0.0,
                "Q_ratio": 0.0,
                "score": np.inf,
                "x0": x0,
                "error": str(e),
            }
            results.append(result_entry)
            if verbose:
                print(f"  Top {idx+1} basin test failed: {e}")
    
    results_sorted = sorted(results, key=lambda r: r.get("score", np.inf))
    
    best_entry = None
    for entry in results_sorted:
        if entry.get("x0") is not None and np.isfinite(entry.get("score", np.inf)):
            best_entry = entry
            break
    
    if best_entry is None:
        # Try to find any with valid x0
        for entry in results_sorted:
            if entry.get("x0") is not None:
                best_entry = entry
                if verbose:
                    print(f"Warning: Using candidate with infinite score")
                break
        
        if best_entry is None:
            raise RuntimeError("All candidates failed! No valid seed found.")
    
    best_x0 = best_entry["x0"]
    best_score = best_entry["score"]
    best_residual_norm = best_entry["residual_norm"]
    best_rho0 = best_entry["rho0"]
    best_tau_variation = best_entry["tau_variation"]
    best_charge_ratio = best_entry["Q_ratio"]
    
    best_params = {
        k: v for k, v in best_entry.items() 
        if k not in ("x0", "score", "residual_norm", "rho0", "tau_variation", "charge", "Q_ratio", "error")
    }
    
    table = []
    for entry in results_sorted:
        table_entry = {
            k: v for k, v in entry.items() if k not in ("x0", "error")
        }
        table.append(table_entry)
    
    if verbose:
        print(f"\nBest Q-ball escape seed found:")
        print(f"  Score = {best_score:.6e}, ||F|| = {best_residual_norm:.6e}")
        print(f"  rho0 = {best_rho0:.6f} (target {rho0_target:.3f}), tau_variation = {best_tau_variation:.6f}")
        print(f"  Q_ratio = {best_charge_ratio:.6f}")
        print(f"  Parameters: {best_params}")
        print(f"\nTop 10 candidates:")
        for i, entry in enumerate(results_sorted[:10]):
            if np.isfinite(entry.get("score", np.inf)):
                cloud_like = "CLOUD" if entry["tau_variation"] < tau_variation_min else "ESCAPE"
                print(f"  {i+1:2d}. score={entry['score']:.6e}, ||F||={entry['residual_norm']:.6e} | "
                      f"a={entry['a_plateau']:.2f}, τ_trans={entry['tau_transition']:.1f}, "
                      f"kick={entry['kick_amp']:.2f} | rho0={entry['rho0']:.4f}, "
                      f"tau_var={entry['tau_variation']:.4f} [{cloud_like}], Q_ratio={entry['Q_ratio']:.3f}")
            else:
                print(f"  {i+1:2d}. FAILED")
    
    return BasinTestResult(
        best_x0=best_x0,
        best_params=best_params,
        best_score=best_score,
        best_residual_norm=best_residual_norm,
        best_rho0=best_rho0,
        best_tau_variation=best_tau_variation,
        best_charge_ratio=best_charge_ratio,
        table=table,
    )


__all__ = ["BasinTestResult", "select_best_qball_escape_seed"]
