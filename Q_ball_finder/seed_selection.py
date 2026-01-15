"""
Deterministic seed selection for 2D Q-ball bounce solver.

This module provides functionality to select the best initial ansatz seed
by minimizing the residual norm ||F(x0)|| over a grid of ansatz parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from .ansatz import build_negative_mode_ansatz, build_qball_like_plateau_ansatz
from .bounce2d import QBall2DSolver, QBall2DSettings
from .grid import RadialTimeGrid, pack_fields, unpack_fields
from .nr_solver import newton_solve, NewtonConvergenceError
from .potentials import LogisticPotentialParams
from .profiles import QBallProfile, UnstableMode


@dataclass
class SeedSelectionResult:
    """Result of seed selection process."""
    best_x0: np.ndarray
    best_params: dict
    best_norm: float
    table: list[dict]  # each entry: params + norm + (optional) quick diagnostics


def select_best_negative_mode_seed(
    params: LogisticPotentialParams,
    omega_qball: float,
    profile: QBallProfile,
    mode: UnstableMode,
    grid: RadialTimeGrid,
    settings: QBall2DSettings,
    *,
    omega_reference: float,
    # parameter grids to explore:
    amplitudes: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    thetas: Tuple[float, ...] = (0.0, np.pi/2, np.pi, 3*np.pi/2),
    tau_centers: Tuple[str, ...] = ("zero", "min"),  # "zero" -> tau_center=0, "min" -> tau_center=grid.tau.min()
    cosh_scales: Tuple[Optional[float], ...] = (None, 3.0),
    envelope_widths: Tuple[Optional[float], ...] = (None,),  # optional: (None, beta/8)
    flip_signs: Tuple[bool, ...] = (False, True),
    decrease_towards_zero_flags: Tuple[bool, ...] = (False, True),
    # numerical control
    norm_kind: str = "l2",
    max_candidates: Optional[int] = None,
    verbose: bool = True,
) -> SeedSelectionResult:
    """
    Select the best initial ansatz seed by minimizing ||residual(x0)|| over a grid of parameters.

    This function evaluates many candidate seeds and returns the one with the smallest
    residual norm. All evaluation is deterministic (no randomness).

    Parameters
    ----------
    params
        Model parameters defining the scalar potential.
    omega_qball
        Frequency of the Q-ball.
    profile
        Q-cloud profile used to build the ansatz.
    mode
        Unstable mode associated with the profile.
    grid
        Radial-time grid for the 2D solver.
    settings
        Solver settings (used for eta0/omega_tilde etc.)
    omega_reference
        Reference frequency for the ansatz construction.
    amplitudes
        Grid of amplitude values to try.
    thetas
        Grid of phase values (in radians) to try.
    tau_centers
        Grid of tau_center modes: "zero" -> tau_center=0.0, "min" -> tau_center=grid.tau.min()
    cosh_scales
        Grid of cosh_scale values (None means no scaling).
    envelope_widths
        Grid of envelope_width values (None means no envelope).
    flip_signs
        Grid of flip_sign boolean values.
    decrease_towards_zero_flags
        Grid of decrease_towards_zero boolean values.
    norm_kind
        Norm type for residual: "l2" (default) uses np.linalg.norm(F.ravel()).
    max_candidates
        Maximum number of candidates to evaluate (None = all).
    verbose
        When True, print progress and top results.

    Returns
    -------
    SeedSelectionResult
        Contains best_x0, best_params, best_norm, and a table of all evaluated seeds.
    """
    # Initialize solver once
    solver = QBall2DSolver(params, omega_qball, settings, grid)

    # Prepare parameter combinations
    candidates = []
    for A in amplitudes:
        for theta in thetas:
            for flip_sign in flip_signs:
                for tau_center_mode in tau_centers:
                    for cosh_scale in cosh_scales:
                        for envelope_width in envelope_widths:
                            for decrease_towards_zero in decrease_towards_zero_flags:
                                # Map tau_center_mode to actual tau_center value and center_at_cloud
                                if tau_center_mode == "zero":
                                    tau_center_val = 0.0
                                    center_at_cloud_eff = False
                                elif tau_center_mode == "min":
                                    tau_center_val = float(grid.tau.min())
                                    center_at_cloud_eff = True
                                else:
                                    raise ValueError(f"Unknown tau_center_mode: {tau_center_mode}")

                                candidates.append({
                                    "amplitude": A,
                                    "theta": theta,
                                    "flip_sign": flip_sign,
                                    "tau_center": tau_center_val,
                                    "tau_center_mode": tau_center_mode,
                                    "cosh_scale": cosh_scale,
                                    "envelope_width": envelope_width,
                                    "center_at_cloud": center_at_cloud_eff,
                                    "decrease_towards_zero": decrease_towards_zero,
                                })

    # Limit number of candidates if requested
    if max_candidates is not None and max_candidates < len(candidates):
        candidates = candidates[:max_candidates]

    total = len(candidates)
    if verbose:
        print(f"Evaluating {total} candidate seeds...")

    # Evaluate all candidates
    results = []
    for idx, cand in enumerate(candidates):
        try:
            # Build ansatz with these parameters
            ansatz = build_negative_mode_ansatz(
                profile=profile,
                mode=mode,
                grid=grid,
                omega_reference=omega_reference,
                amplitude=cand["amplitude"],
                tau_center=cand["tau_center"],
                cosh_scale=cand["cosh_scale"],
                envelope_width=cand["envelope_width"],
                flip_sign=cand["flip_sign"],
                omega_tilde=settings.omega_tilde,
                center_at_cloud=cand["center_at_cloud"],
                decrease_towards_zero=cand["decrease_towards_zero"],
                phase=cand["theta"],
            )

            # Pack into x0 vector
            x0 = pack_fields(ansatz.y, ansatz.ybar)

            # Evaluate residual
            F = solver.residual(x0)

            # Compute norm (handle NaN/inf by assigning +inf)
            if norm_kind == "l2":
                norm = np.linalg.norm(F.ravel())
            else:
                raise ValueError(f"Unknown norm_kind: {norm_kind}")

            # Check for invalid results
            if not np.isfinite(norm):
                norm = np.inf

            # Store result
            result_entry = {
                "amplitude": cand["amplitude"],
                "theta": cand["theta"],
                "flip_sign": cand["flip_sign"],
                "tau_center_mode": cand["tau_center_mode"],
                "cosh_scale": cand["cosh_scale"],
                "envelope_width": cand["envelope_width"],
                "center_at_cloud": cand["center_at_cloud"],
                "decrease_towards_zero": cand["decrease_towards_zero"],
                "norm": norm,
                "x0": x0,  # store for later use
            }
            results.append(result_entry)

            if verbose and (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{total}")

        except Exception as e:
            # If anything goes wrong, assign +inf norm
            result_entry = {
                "amplitude": cand["amplitude"],
                "theta": cand["theta"],
                "flip_sign": cand["flip_sign"],
                "tau_center_mode": cand["tau_center_mode"],
                "cosh_scale": cand["cosh_scale"],
                "envelope_width": cand["envelope_width"],
                "center_at_cloud": cand["center_at_cloud"],
                "decrease_towards_zero": cand["decrease_towards_zero"],
                "norm": np.inf,
                "x0": None,
                "error": str(e),
            }
            results.append(result_entry)
            if verbose:
                print(f"  Candidate {idx + 1} failed: {e}")

    # Sort by norm (ascending)
    results_sorted = sorted(results, key=lambda r: r["norm"])

    # Extract best candidate
    best_entry = results_sorted[0]
    if best_entry["x0"] is None:
        raise RuntimeError("All candidates failed! No valid seed found.")

    best_x0 = best_entry["x0"]
    best_norm = best_entry["norm"]

    # Build best_params dict (without x0)
    best_params = {
        k: v for k, v in best_entry.items() if k not in ("x0", "error", "norm")
    }

    # Build table (without x0 for memory efficiency)
    table = []
    for entry in results_sorted:
        table_entry = {
            k: v for k, v in entry.items() if k not in ("x0", "error")
        }
        table.append(table_entry)

    if verbose:
        print(f"\nBest seed found: ||F|| = {best_norm:.6e}")
        print(f"Best parameters: {best_params}")
        print(f"\nTop 10 seeds:")
        for i, entry in enumerate(results_sorted[:10]):
            cosh_str = str(entry['cosh_scale']) if entry['cosh_scale'] is not None else "None"
            env_str = str(entry['envelope_width']) if entry['envelope_width'] is not None else "None"
            print(f"  {i+1:2d}. ||F|| = {entry['norm']:.6e} | "
                  f"A={entry['amplitude']:.1f}, θ={entry['theta']:.3f}, "
                  f"flip={entry['flip_sign']}, τ_center={entry['tau_center_mode']}, "
                  f"cosh={cosh_str}, env_w={env_str}, "
                  f"center_cloud={entry['center_at_cloud']}, dec_zero={entry['decrease_towards_zero']}")

    return SeedSelectionResult(
        best_x0=best_x0,
        best_params=best_params,
        best_norm=best_norm,
        table=table,
    )


@dataclass
class PlateauSeedResult:
    """Result of plateau seed selection process."""
    best_x0: np.ndarray
    best_params: dict
    best_score: float
    best_norm: float
    table: list[dict]  # each entry: params + metrics


def select_best_plateau_seed(
    params: LogisticPotentialParams,
    omega: float,
    profile: QBallProfile,
    grid: RadialTimeGrid,
    settings: QBall2DSettings,
    *,
    tau_center_list: Tuple[float, ...] = (-10.0, -15.0, -20.0),
    tau_width_list: Tuple[float, ...] = (2.0, 3.0, 5.0),
    amp_scale_list: Tuple[float, ...] = (0.7, 1.0, 1.3),
    phase_list: Tuple[float, ...] = (0.0,),
    max_rho_min: Optional[float] = None,
    mass_min: Optional[float] = None,
    try_newton_step: bool = False,  # Default False to avoid slowdown; enable if needed
    newton_damping: float = 0.2,
    collapse_threshold: float = 0.3,
    verbose: bool = True,
) -> PlateauSeedResult:
    """
    Select the best plateau ansatz seed by evaluating a grid of parameters and
    scoring candidates based on residual norm and nontriviality metrics.
    
    This function builds ansätze with the structure φ(r,τ) = ρ(r) * f(τ) where
    f(τ) transitions from plateau (metastable) at τ << τ_center to vacuum at τ >> τ_center.
    
    Parameters
    ----------
    params
        Model parameters defining the scalar potential.
    omega
        Frequency parameter for the solver.
    profile
        QBallProfile providing the radial profile ρ(r).
    grid
        Radial-time grid for the 2D solver.
    settings
        Solver settings.
    tau_center_list
        Grid of tau_center values (typically negative, e.g., -10 to -20).
    tau_width_list
        Grid of tau_width values (transition width, typically 2-5).
    amp_scale_list
        Grid of amplitude scaling factors.
    phase_list
        Grid of phase values (in radians) for complex field: φ = ρ * exp(i*phase).
    max_rho_min
        Minimum threshold for max(rho) to avoid vacuum seeds. If None, uses
        0.05 * max(profile.phi_abs).
    mass_min
        Minimum threshold for mass integral to avoid vacuum seeds. If None, uses
        0.05 * mass_ref where mass_ref is computed from plateau slice.
    try_newton_step
        If True, perform one damped Newton step to check for collapse.
    newton_damping
        Damping factor for the one-step Newton test (default 0.2).
    collapse_threshold
        Threshold for detecting collapse: if mass_after < collapse_threshold * mass_before,
        mark as collapsed (default 0.3).
    verbose
        When True, print progress and results.
    
    Returns
    -------
    PlateauSeedResult
        Contains best_x0, best_params, best_score, best_norm, and table of candidates.
    """
    # Initialize solver once
    solver = QBall2DSolver(params, omega, settings, grid)
    
    # Set default thresholds if not provided (more lenient)
    rho_max_ref = np.max(profile.phi_abs)
    if max_rho_min is None:
        max_rho_min = 0.01 * rho_max_ref  # More lenient: 1% instead of 5%
    
    # Compute reference mass from plateau slice (f≈1)
    # Use the radial profile at a slice where f(τ) ≈ 1
    rho_r_ref = np.interp(grid.r, profile.r, profile.phi_abs)
    mass_ref = 4.0 * np.pi * np.trapz(grid.r**2 * rho_r_ref**2, grid.r)
    if mass_min is None:
        mass_min = 0.01 * mass_ref  # More lenient: 1% instead of 5%
    
    # Prepare parameter combinations
    candidates = []
    for tau_center in tau_center_list:
        for tau_width in tau_width_list:
            for amp_scale in amp_scale_list:
                for phase in phase_list:
                    candidates.append({
                        "tau_center": tau_center,
                        "tau_width": tau_width,
                        "amp_scale": amp_scale,
                        "phase": phase,
                    })
    
    total = len(candidates)
    if verbose:
        print(f"Evaluating {total} plateau seed candidates...")
        print(f"  Thresholds: max_rho_min={max_rho_min:.6f}, mass_min={mass_min:.6f}")
    
    # Evaluate all candidates
    results = []
    for idx, cand in enumerate(candidates):
        try:
            # Build plateau ansatz
            ansatz = build_qball_like_plateau_ansatz(
                grid=grid,
                omega=omega,
                profile=profile,
                tau_center=cand["tau_center"],
                tau_width=cand["tau_width"],
                amp_scale=cand["amp_scale"],
                phase=cand["phase"],
            )
            
            # Pack into x0 vector
            x0 = pack_fields(ansatz.y, ansatz.ybar)
            
            # Compute nontriviality metrics
            rho_2d = np.sqrt(np.maximum((ansatz.phi * ansatz.phibar).real, 0.0))
            max_rho = np.max(rho_2d)
            
            # Mass at τ index 0 (closest to τ=0)
            rho_slice_0 = rho_2d[:, 0]
            mass_rho = 4.0 * np.pi * np.trapz(grid.r**2 * rho_slice_0**2, grid.r)
            
            # Evaluate residual
            F = solver.residual(x0)
            residual_norm = np.linalg.norm(F.ravel())
            
            if not np.isfinite(residual_norm):
                residual_norm = np.inf
            
            # Check nontriviality (penalize but don't completely reject)
            # We'll penalize low nontriviality but still allow candidates with good residual
            nontriviality_penalty = 0.0
            if max_rho < max_rho_min:
                # Penalty proportional to how far below threshold
                nontriviality_penalty = residual_norm * 10.0 * (max_rho_min - max_rho) / max_rho_min
            if mass_rho < mass_min:
                nontriviality_penalty += residual_norm * 10.0 * (mass_min - mass_rho) / mass_min
            
            if nontriviality_penalty > 0:
                score = residual_norm + nontriviality_penalty
                collapsed = True
                collapse_reason = f"nontriviality (max_rho={max_rho:.6f} < {max_rho_min:.6f} or mass={mass_rho:.2f} < {mass_min:.2f})"
            else:
                # Optional: one-step Newton test
                collapsed = False
                collapse_reason = None
                if try_newton_step:
                    try:
                        # Perform one damped Newton step
                        newton_res = newton_solve(
                            residual=lambda vec: solver.residual(vec),
                            jacobian=lambda vec: solver.jacobian(vec),
                            x0=x0,
                            tol=1e-8,  # irrelevant for single step
                            max_iter=1,
                            damping=newton_damping,
                            norm=lambda v: np.linalg.norm(v.ravel()),
                            verbose=False,
                        )
                        
                        # Check if collapsed
                        y_after, ybar_after = unpack_fields(newton_res.x, grid.Nr, grid.Ntau)
                        from .grid import phi_from_y, phi_from_ybar
                        phi_after = phi_from_y(y_after, grid, omega)
                        phibar_after = phi_from_ybar(ybar_after, grid, omega)
                        
                        rho_2d_after = np.sqrt(np.maximum((phi_after * phibar_after).real, 0.0))
                        max_rho_after = np.max(rho_2d_after)
                        rho_slice_0_after = rho_2d_after[:, 0]
                        mass_rho_after = 4.0 * np.pi * np.trapz(grid.r**2 * rho_slice_0_after**2, grid.r)
                        
                        # Check collapse: use a more lenient threshold
                        collapse_ratio_mass = mass_rho_after / mass_rho if mass_rho > 0 else 0.0
                        collapse_ratio_rho = max_rho_after / max_rho if max_rho > 0 else 0.0
                        
                        if collapse_ratio_mass < collapse_threshold or collapse_ratio_rho < collapse_threshold:
                            collapsed = True
                            collapse_reason = f"newton_collapse (mass_ratio={collapse_ratio_mass:.3f}, rho_ratio={collapse_ratio_rho:.3f})"
                            # Penalize but don't completely reject
                            score = residual_norm * (1.0 + 10.0 * (1.0 - min(collapse_ratio_mass, collapse_ratio_rho)))
                        else:
                            score = residual_norm
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
                        # If Newton step fails, still use residual norm but warn
                        if verbose:
                            print(f"  Warning: Newton step failed for candidate {idx+1}: {e}")
                        score = residual_norm
                else:
                    score = residual_norm
            
            # Store result
            result_entry = {
                "tau_center": cand["tau_center"],
                "tau_width": cand["tau_width"],
                "amp_scale": cand["amp_scale"],
                "phase": cand["phase"],
                "residual_norm": residual_norm,
                "max_rho": max_rho,
                "mass_rho": mass_rho,
                "score": score,
                "collapsed": collapsed,
                "collapse_reason": collapse_reason,
                "x0": x0,
            }
            results.append(result_entry)
            
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{total}")
        
        except Exception as e:
            # If anything goes wrong, assign +inf score
            result_entry = {
                "tau_center": cand["tau_center"],
                "tau_width": cand["tau_width"],
                "amp_scale": cand["amp_scale"],
                "phase": cand["phase"],
                "residual_norm": np.inf,
                "max_rho": 0.0,
                "mass_rho": 0.0,
                "score": np.inf,
                "collapsed": True,
                "collapse_reason": f"exception: {str(e)}",
                "x0": None,
            }
            results.append(result_entry)
            if verbose:
                print(f"  Candidate {idx + 1} failed: {e}")
    
    # Sort by score (ascending), but handle inf scores properly
    def score_key(r):
        score = r.get("score", np.inf)
        if np.isfinite(score):
            return (0, score)  # Finite scores first
        else:
            return (1, r.get("residual_norm", np.inf))  # Then by residual norm
    
    results_sorted = sorted(results, key=score_key)
    
    # Extract best candidate (prefer finite scores, but accept any valid x0)
    best_entry = None
    for entry in results_sorted:
        if entry.get("x0") is not None:
            score_val = entry.get("score", np.inf)
            if np.isfinite(score_val):
                best_entry = entry
                break
    
    if best_entry is None:
        # If no candidate with finite score, try to find any with valid x0
        for entry in results_sorted:
            if entry.get("x0") is not None:
                best_entry = entry
                if verbose:
                    print(f"Warning: Using candidate with infinite score (all candidates had issues)")
                    print(f"  Candidate: {entry.get('tau_center')}, {entry.get('tau_width')}, {entry.get('amp_scale')}")
                    print(f"  Reason: {entry.get('collapse_reason', 'unknown')}")
                break
        
        if best_entry is None:
            # Diagnostic: show what went wrong
            if verbose:
                print("\nDiagnostics:")
                print(f"  Total candidates: {len(results)}")
                valid_x0 = sum(1 for r in results if r.get("x0") is not None)
                print(f"  Candidates with valid x0: {valid_x0}")
                finite_scores = sum(1 for r in results if np.isfinite(r.get("score", np.inf)))
                print(f"  Candidates with finite score: {finite_scores}")
                if len(results) > 0:
                    print(f"  First candidate error: {results[0].get('collapse_reason', 'unknown')}")
            raise RuntimeError("All candidates failed! No valid seed found.")
    
    best_x0 = best_entry["x0"]
    best_score = best_entry["score"]
    best_norm = best_entry["residual_norm"]
    
    # Build best_params dict (without x0)
    best_params = {
        k: v for k, v in best_entry.items() if k not in ("x0", "score", "residual_norm", "max_rho", "mass_rho", "collapsed", "collapse_reason")
    }
    
    # Build table (without x0 for memory efficiency)
    table = []
    for entry in results_sorted:
        table_entry = {
            k: v for k, v in entry.items() if k not in ("x0",)
        }
        table.append(table_entry)
    
    if verbose:
        print(f"\nBest plateau seed found: score = {best_score:.6e}, ||F|| = {best_norm:.6e}")
        print(f"Best parameters: {best_params}")
        print(f"  max_rho = {best_entry['max_rho']:.6f}, mass_rho = {best_entry['mass_rho']:.6f}")
        if best_entry.get('collapsed', False):
            print(f"  Warning: Best candidate had issues: {best_entry.get('collapse_reason', 'unknown')}")
        print(f"\nTop 10 candidates:")
        for i, entry in enumerate(results_sorted[:10]):
            score_val = entry.get('score', np.inf)
            if np.isfinite(score_val):
                collapsed_str = " [COLLAPSED]" if entry.get('collapsed', False) else ""
                print(f"  {i+1:2d}. score={score_val:.6e}, ||F||={entry['residual_norm']:.6e}{collapsed_str} | "
                      f"τ_c={entry['tau_center']:.1f}, τ_w={entry['tau_width']:.1f}, "
                      f"amp={entry['amp_scale']:.2f}, phase={entry['phase']:.3f} | "
                      f"max_ρ={entry['max_rho']:.4f}, mass={entry['mass_rho']:.2f}")
            else:
                reason = entry.get('collapse_reason', 'unknown')
                print(f"  {i+1:2d}. FAILED ({reason})")
    
    return PlateauSeedResult(
        best_x0=best_x0,
        best_params=best_params,
        best_score=best_score,
        best_norm=best_norm,
        table=table,
    )
