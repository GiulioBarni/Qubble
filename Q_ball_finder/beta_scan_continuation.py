"""β continuation with fixed Δτ grids and plateau enforcement."""


import numpy as np
from Q_ball_finder.ansatz import AnsatzResult
from Q_ball_finder.grid import build_grid, pack_fields
from Q_ball_finder.notebook_utils import resample_ansatz, enforce_plateau_after_resample


def compute_ntau_for_beta(beta_i, beta_ref, ntau_ref, min_ntau=100):
    """
    Compute Ntau for beta_i to keep dtau approximately constant.
    
    dtau_target = beta_ref / (2 * Ntau_ref)
    Ntau_i = round(0.5 * beta_i / dtau_target)
    Keep Ntau_i even and >= min_ntau
    """
    dtau_target = beta_ref / (2.0 * ntau_ref)
    ntau_i = int(round(0.5 * beta_i / dtau_target))
    
    # Ensure even
    if ntau_i % 2 != 0:
        ntau_i += 1
    
    # Ensure minimum
    if ntau_i < min_ntau:
        ntau_i = min_ntau
        if ntau_i % 2 != 0:
            ntau_i += 1
    
    return ntau_i


def prepare_warm_start_for_beta_continuation(
    solution_prev,
    beta_new,
    beta_ref,
    ntau_ref,
    nr_ref,
    lr_ref,
    omega,
    min_ntau=100,
):
    """
    Prepare warm start for beta continuation by resampling previous solution.
    
    1. Computes appropriate Ntau for beta_new to keep dtau constant
    2. Builds new grid
    3. Resamples previous solution onto new grid (only y/ybar, reconstructs phi/phibar)
    4. Enforces plateau in new tau region
    
    Args:
        solution_prev: QBall2DSolution from previous beta step
        beta_new: New beta value
        beta_ref: Reference beta (for dtau calculation)
        ntau_ref: Reference Ntau (for dtau calculation)
        nr_ref: Nr to use (typically same as previous)
        lr_ref: Lr to use (typically same as previous)
        omega: Frequency parameter (required for reconstructing phi/phibar)
        min_ntau: Minimum Ntau value
    
    Returns:
        tuple: (x0_initial, grid_new, ntau_new)
            x0_initial: Packed initial guess vector
            grid_new: New RadialTimeGrid
            ntau_new: New Ntau value
    """
    # Compute Ntau for new beta
    ntau_new = compute_ntau_for_beta(beta_new, beta_ref, ntau_ref, min_ntau)
    
    # Build new grid
    grid_new = build_grid(nr_ref, ntau_new, lr_ref, beta_new)
    
    # Create AnsatzResult from previous solution
    ans_old = AnsatzResult(
        phi=solution_prev.phi,
        phibar=solution_prev.phibar,
        y=solution_prev.y,
        ybar=solution_prev.ybar,
    )
    
    # Resample onto new grid (only y/ybar, reconstruct phi/phibar)
    ans_new = resample_ansatz(
        ans_old,
        solution_prev.grid,
        grid_new,
        omega=omega,
        resample_phi_directly=False,
        clamp_tau_to_plateau=True,
    )
    
    # CRITICAL: Enforce plateau in new tau region
    ans_new = enforce_plateau_after_resample(ans_new, ans_old)
    
    # Pack for solver
    x0_initial = pack_fields(ans_new.y, ans_new.ybar)
    
    return x0_initial, grid_new, ntau_new


