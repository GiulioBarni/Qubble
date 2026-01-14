"""
Utility helpers used by the Jupyter notebooks.

The functions defined here are kept lightweight wrappers around the
package internals so that they can be imported directly inside the
notebooks without cluttering them with boilerplate code.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .ansatz import AnsatzResult
from .bounce2d import QBall2DSolution, QBall2DSettings, solve_bounce_for_eta
from .grid import (
    RadialTimeGrid,
    build_grid,
    pack_fields,
    phi_from_y,
    phi_from_ybar,
)
from .profiles import QBallProfile, UnstableMode
from .potentials import LogisticPotentialParams


def solve_fixed_beta_eta(
    params: LogisticPotentialParams,
    omega: float,
    profile: QBallProfile,
    mode: UnstableMode,
    *,
    settings: QBall2DSettings,
    eta: float,
    x0_initial: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> tuple[QBall2DSolution, float, np.ndarray]:
    """
    Run a single Newton solve for the 2D configuration at fixed β and η.

    Parameters
    ----------
    params
        Model parameters defining the scalar potential.
    omega
        Frequency of the metastable Q-ball used for the 2D bounce.
    profile
        One-dimensional Q-cloud profile used to build the ansatz if no
        warm start is supplied.
    mode
        Unstable mode associated with the Q-cloud profile.
    settings
        Solver settings controlling the lattice and Newton iteration.
    eta
        Value of η kept fixed during this Newton solve.
    x0_initial
        Optional packed (y, ybar) vector used as warm start.
    verbose
        When True, forward the verbosity flag to the Newton solver.

    Returns
    -------
    solution
        The converged 2D configuration.
    charge
        Charge computed with the max-slice convention.
    x_vector
        Packed (y, ybar) vector of the final solution, suitable for
        reusing as warm start.
    """
    grid: RadialTimeGrid = build_grid(settings.Nr, settings.Ntau, settings.Lr, settings.beta)
    charge, solution, x_vec = solve_bounce_for_eta(
        params,
        omega,
        profile,
        mode,
        grid,
        settings,
        eta,
        x0_prev=x0_initial,
        verbose=verbose,
    )
    return solution, charge, x_vec


def plot_newton_snapshots(
    snapshot_dir: str | Path,
    cmap: str = "viridis",
    vmax: float | None = None,
    columns: int = 3,
) -> None:
    """
    Visualise the snapshots saved during a Newton iteration.

    Parameters
    ----------
    snapshot_dir
        Directory containing ``*.npz`` files with the snapshot data.
    cmap
        Matplotlib colormap used for the images.
    vmax
        Optional upper bound for the colour scale.
    columns
        Number of columns in the subplot grid.
    """
    snapshot_dir = Path(snapshot_dir)
    files = sorted(snapshot_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")

    n = len(files)
    rows = ceil(n / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.8 * rows), squeeze=False)

    for ax, file in zip(axes.flat, files):
        data = np.load(file)
        phi = data["phi"]
        phibar = data["phibar"]
        rho = np.sqrt(np.maximum((phi * phibar).real, 0.0))

        r = data["r"]
        tau = data["tau"]

        rho_plot = rho.T
        im = ax.imshow(
            rho_plot,
            extent=[r.min(), r.max(), tau.max(), tau.min()],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmax=vmax,
        )
        ax.set_title(file.stem)
        ax.set_xlabel("r")
        ax.set_ylabel(r"$\tau$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(r"$\rho(r,\tau)=\sqrt{\phi\,\bar\phi}$ snapshots", fontsize=16)
    fig.tight_layout()
    plt.show()



def resample_complex_field(
    field_old: np.ndarray,
    r_old: np.ndarray,
    tau_old: np.ndarray,
    r_new: np.ndarray,
    tau_new: np.ndarray,
    *,
    clamp_tau_to_plateau: bool = False,
) -> np.ndarray:
    """
    Interpolate a complex field defined on the (r_old, tau_old) grid onto
    the new coordinates (r_new, tau_new) using a regular-grid interpolator.
    
    Parameters
    ----------
    field_old
        Field array of shape (Nr_old, Ntau_old)
    r_old, tau_old
        Old grid coordinates
    r_new, tau_new
        New grid coordinates
    clamp_tau_to_plateau
        If True, clamp tau_new values to tau_old.min() (plateau) when they
        extend beyond tau_old range. This prevents zero-filling in tau extension
        regions when beta increases. For r beyond r_old.max(), still fills with 0.0.
    
    Returns
    -------
    Field array of shape (Nr_new, Ntau_new)
    """
    R_new, T_new = np.meshgrid(r_new, tau_new, indexing="ij")
    
    if clamp_tau_to_plateau:
        # Clamp tau to plateau (tau_old.min()) to avoid zero-filling artifacts
        # when beta increases and tau_new extends beyond tau_old range
        T_eval = np.clip(T_new, tau_old.min(), tau_old.max())
    else:
        T_eval = T_new
    
    points_new = np.stack([R_new, T_eval], axis=-1)

    interp_re = RegularGridInterpolator(
        (r_old, tau_old), np.real(field_old), bounds_error=False, fill_value=0.0
    )
    interp_im = RegularGridInterpolator(
        (r_old, tau_old), np.imag(field_old), bounds_error=False, fill_value=0.0
    )
    return interp_re(points_new) + 1j * interp_im(points_new)


def resample_ansatz(
    ansatz_old: AnsatzResult,
    grid_old: RadialTimeGrid,
    grid_new: RadialTimeGrid,
    *,
    omega: Optional[float] = None,
    resample_phi_directly: bool = False,
    clamp_tau_to_plateau: bool = True,
) -> AnsatzResult:
    """
    Resample an ``AnsatzResult`` from its original grid onto ``grid_new``.

    This is useful for bootstrapping Newton solves on finer lattices using a
    converged configuration obtained on a coarser grid.
    
    Parameters
    ----------
    ansatz_old
        AnsatzResult to resample
    grid_old
        Old grid
    grid_new
        New grid
    omega
        Frequency parameter (required if resample_phi_directly=False)
    resample_phi_directly
        If True, resample phi/phibar directly (legacy behavior).
        If False (recommended), resample only y/ybar and reconstruct phi/phibar.
    clamp_tau_to_plateau
        If True (default), clamp tau values to plateau when beta increases.
        This prevents zero-filling artifacts in extended tau regions.
    
    Returns
    -------
    Resampled AnsatzResult
    """
    r_old, tau_old = grid_old.r, grid_old.tau
    r_new, tau_new = grid_new.r, grid_new.tau

    if resample_phi_directly:
        # Legacy behavior: resample all fields directly
        phi_new = resample_complex_field(
            ansatz_old.phi, r_old, tau_old, r_new, tau_new,
            clamp_tau_to_plateau=clamp_tau_to_plateau
        )
        phibar_new = resample_complex_field(
            ansatz_old.phibar, r_old, tau_old, r_new, tau_new,
            clamp_tau_to_plateau=clamp_tau_to_plateau
        )
        y_new = resample_complex_field(
            ansatz_old.y, r_old, tau_old, r_new, tau_new,
            clamp_tau_to_plateau=clamp_tau_to_plateau
        )
        ybar_new = resample_complex_field(
            ansatz_old.ybar, r_old, tau_old, r_new, tau_new,
            clamp_tau_to_plateau=clamp_tau_to_plateau
        )
    else:
        # Recommended: resample only y/ybar and reconstruct phi/phibar
        if omega is None:
            raise ValueError("omega must be provided when resample_phi_directly=False")
        
        y_new = resample_complex_field(
            ansatz_old.y, r_old, tau_old, r_new, tau_new,
            clamp_tau_to_plateau=clamp_tau_to_plateau
        )
        ybar_new = resample_complex_field(
            ansatz_old.ybar, r_old, tau_old, r_new, tau_new,
            clamp_tau_to_plateau=clamp_tau_to_plateau
        )
        
        # Reconstruct phi/phibar from resampled y/ybar
        phi_new = phi_from_y(y_new, grid_new, omega)
        phibar_new = phi_from_ybar(ybar_new, grid_new, omega)

    return AnsatzResult(phi=phi_new, phibar=phibar_new, y=y_new, ybar=ybar_new)


def enforce_plateau_after_resample(
    ans_new: AnsatzResult,
    ans_old: AnsatzResult,
    k_slices: Optional[int] = None,
) -> AnsatzResult:
    """
    Enforce plateau in the last K tau slices after resampling.
    
    When beta increases, resampling fills new tau region with zeros (fill_value=0.0).
    This creates an artificial "frontier" that can drift the decay time.
    
    After resampling, overwrite the last K columns with a nonzero plateau background.
    
    Parameters
    ----------
    ans_new
        AnsatzResult after resampling (will be modified in-place)
    ans_old
        AnsatzResult from previous converged solution
    k_slices
        Number of slices to overwrite (default: max(10, int(0.05*Ntau_new)))
    
    Returns
    -------
    Modified ans_new (same object, modified in-place)
    """
    ntau_new = ans_new.y.shape[1]
    
    if k_slices is None:
        k_slices = max(10, int(0.05 * ntau_new))
    
    k_slices = min(k_slices, ntau_new)
    
    # Use the last slice from old solution as plateau
    # Tile it across the last K slices
    if ans_old.y.shape[1] > 0:
        y_bg = ans_old.y[:, -1]  # Last slice from old solution
        ybar_bg = ans_old.ybar[:, -1]
    else:
        # Fallback: use zeros (shouldn't happen in practice)
        y_bg = np.zeros(ans_new.y.shape[0], dtype=complex)
        ybar_bg = np.zeros(ans_new.ybar.shape[0], dtype=complex)
    
    # Overwrite last K slices
    ans_new.y[:, -k_slices:] = y_bg[:, None]
    ans_new.ybar[:, -k_slices:] = ybar_bg[:, None]
    
    return ans_new


def warm_start_from_solution(
    prev_solution: QBall2DSolution,
    grid_new: RadialTimeGrid,
    omega: float,
    *,
    enforce_plateau: bool = True,
) -> np.ndarray:
    """
    Prepare warm start from a previous converged solution for beta continuation.
    
    This function:
    1. Creates AnsatzResult from previous solution
    2. Resamples y/ybar onto new grid (with tau clamping to plateau)
    3. Reconstructs phi/phibar from resampled y/ybar
    4. Optionally enforces plateau in extended tau region
    5. Returns packed x0_initial vector
    
    Parameters
    ----------
    prev_solution
        QBall2DSolution from previous converged step
    grid_new
        New grid (typically with different beta/Ntau)
    omega
        Frequency parameter for reconstructing phi/phibar
    enforce_plateau
        If True (default), enforce plateau in extended tau region
    
    Returns
    -------
    Packed initial guess vector x0_initial = pack_fields(y_new, ybar_new)
    """
    # Create AnsatzResult from previous solution
    ans_old = AnsatzResult(
        phi=prev_solution.phi,
        phibar=prev_solution.phibar,
        y=prev_solution.y,
        ybar=prev_solution.ybar,
    )
    
    # Resample onto new grid (only y/ybar, reconstruct phi/phibar)
    ans_new = resample_ansatz(
        ans_old,
        prev_solution.grid,
        grid_new,
        omega=omega,
        resample_phi_directly=False,
        clamp_tau_to_plateau=True,
    )
    
    # Optionally enforce plateau in extended tau region
    if enforce_plateau:
        ans_new = enforce_plateau_after_resample(ans_new, ans_old)
    
    # Pack for solver
    x0_initial = pack_fields(ans_new.y, ans_new.ybar)
    
    return x0_initial


__all__ = [
    "solve_fixed_beta_eta",
    "plot_newton_snapshots",
    "resample_complex_field",
    "resample_ansatz",
    "enforce_plateau_after_resample",
    "warm_start_from_solution",
]

