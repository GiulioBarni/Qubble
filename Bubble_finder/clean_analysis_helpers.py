"""
Helper utilities for the clean bubble analysis notebook.

This module centralizes reusable logic so the notebook remains compact:
- plotting/style setup
- 1D bounce caching/loading
- 1D omega scans
- 2D seed construction with a unified interface
- eta0 scan + Newton wrapper
- repeated plotting helpers
- static-energy diagnostics vs tau
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .bounce_1d import solve_bounce
from .observables_1d import (
    compute_charge,
    compute_free_energy_grandcanonical,
    compute_energy_minkowski_1d_spherical,
    compute_charge_1d_volume_corrected,
    compute_energy_physical_1d_volume_corrected,
)
from .potential_bubble import V_phi


@dataclass
class ModelParams:
    phi0: float
    v1: float
    v2: float


def configure_matplotlib(
    use_tex: bool = True,
    fontsize: int = 12,
    dpi: int = 140,
) -> None:
    """Global plot style with LaTeX-like serif/math fonts."""
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize + 1,
            "legend.fontsize": fontsize - 1,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": bool(use_tex),
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
        }
    )


def _cache_file(cache_dir: Path, omega: float, d: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"bounce1d_d{d}_omega_{omega:.8f}.npz"


def solve_or_load_bounce_1d(
    cache_dir: Optional[Path],
    model: ModelParams,
    omega: float,
    d: int,
    *,
    r0: float = 1e-6,
    rmax: float = 250.0,
    n_grid_points: int = 1400,
    max_iter: int = 160,
    force_recompute: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Load 1D bounce from cache or compute and cache it.

    If ``cache_dir`` is ``None``, skip disk I/O and always solve fresh.
    """
    if cache_dir is None:
        r, phi, phi0_center, phi_false, phi_true = solve_bounce(
            model.phi0,
            model.v1,
            model.v2,
            float(omega),
            d=int(d),
            r0=r0,
            rmax=rmax,
            n_grid_points=n_grid_points,
            max_iter=max_iter,
            verbose=verbose,
        )
        if r is None or phi is None:
            raise RuntimeError(f"Failed to solve 1D bounce for d={d}, omega={omega:.6g}")
        return {
            "r": np.asarray(r, dtype=float),
            "phi": np.asarray(phi, dtype=float),
            "phi0_center": float(phi0_center),
            "phi_false": float(phi_false),
            "phi_true": float(phi_true),
            "omega": float(omega),
            "d": int(d),
            "from_cache": False,
        }

    cache_path = _cache_file(cache_dir, omega=omega, d=d)
    if cache_path.exists() and not force_recompute:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "r": data["r"],
            "phi": data["phi"],
            "phi0_center": float(data["phi0_center"]),
            "phi_false": float(data["phi_false"]),
            "phi_true": float(data["phi_true"]),
            "omega": float(data["omega"]),
            "d": int(data["d"]),
            "from_cache": True,
        }

    r, phi, phi0_center, phi_false, phi_true = solve_bounce(
        model.phi0,
        model.v1,
        model.v2,
        float(omega),
        d=int(d),
        r0=r0,
        rmax=rmax,
        n_grid_points=n_grid_points,
        max_iter=max_iter,
        verbose=verbose,
    )
    if r is None or phi is None:
        raise RuntimeError(f"Failed to solve 1D bounce for d={d}, omega={omega:.6g}")

    np.savez_compressed(
        cache_path,
        r=np.asarray(r, dtype=float),
        phi=np.asarray(phi, dtype=float),
        phi0_center=float(phi0_center),
        phi_false=float(phi_false),
        phi_true=float(phi_true),
        omega=float(omega),
        d=int(d),
    )
    return {
        "r": np.asarray(r, dtype=float),
        "phi": np.asarray(phi, dtype=float),
        "phi0_center": float(phi0_center),
        "phi_false": float(phi_false),
        "phi_true": float(phi_true),
        "omega": float(omega),
        "d": int(d),
        "from_cache": False,
    }


def compute_1d_observables(
    model: ModelParams,
    bounce: Dict[str, Any],
    *,
    r_max_ref: float | None = None,
) -> Dict[str, float]:
    """Compute charge and energies for one cached/computed 1D profile."""
    r = np.asarray(bounce["r"], dtype=float)
    phi = np.asarray(bounce["phi"], dtype=float)
    omega = float(bounce["omega"])
    phi_false = float(bounce["phi_false"])

    V_of_rho = lambda rho: V_phi(rho, model.phi0, model.v1, model.v2)

    if r_max_ref is None:
        Q = compute_charge(r, phi, omega)
        E_M = compute_energy_minkowski_1d_spherical(r, phi, omega, V_of_rho)
    else:
        Q = compute_charge_1d_volume_corrected(
            r, phi, omega, r_max_ref=float(r_max_ref), phi_false_tail=phi_false
        )
        E_M = compute_energy_physical_1d_volume_corrected(
            r,
            phi,
            omega,
            V_of_rho,
            r_max_ref=float(r_max_ref),
            rho_tail=phi_false,
        )

    F_omega = compute_free_energy_grandcanonical(r, phi, omega, V_of_rho)
    return {
        "Q": float(Q),
        "E_M": float(E_M),
        "F_omega": float(F_omega),
    }


def run_1d_scan(
    cache_dir: Optional[Path],
    model: ModelParams,
    omega_values: Iterable[float],
    *,
    dimensions: Tuple[int, ...] = (3, 4),
    r0: float = 1e-6,
    rmax: float = 250.0,
    n_grid_points: int = 1200,
    max_iter: int = 150,
    verbose: bool = False,
) -> Dict[int, list[Dict[str, Any]]]:
    """Compute/load a full omega scan for selected O(d) dimensions."""
    out: Dict[int, list[Dict[str, Any]]] = {d: [] for d in dimensions}
    for om in omega_values:
        for d in dimensions:
            try:
                res = solve_or_load_bounce_1d(
                    cache_dir,
                    model,
                    omega=float(om),
                    d=d,
                    r0=r0,
                    rmax=rmax,
                    n_grid_points=n_grid_points,
                    max_iter=max_iter,
                    verbose=verbose,
                )
                out[d].append(res)
            except Exception:
                continue
    return out


def _interp_profile(r_src: np.ndarray, phi_src: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Stable linear interpolation with flat tails."""
    f = interp1d(
        np.asarray(r_src, dtype=float),
        np.asarray(phi_src, dtype=float),
        kind="linear",
        bounds_error=False,
        fill_value=(float(phi_src[0]), float(phi_src[-1])),
    )
    return np.asarray(f(np.asarray(x_eval, dtype=float)), dtype=float)


def _normalize_profile_to_solver_phi_units(
    phi_profile: np.ndarray,
    phi_ref: float,
    tail_points: int = 8,
) -> Tuple[np.ndarray, float]:
    """
    Normalize a 1D profile into solver units |phi|.

    If the profile tail is closer to sqrt(2)*phi_ref than to phi_ref, interpret it as
    physical modulus rho_phys and rescale by 1/sqrt(2).
    """
    arr = np.asarray(phi_profile, dtype=float).copy()
    if arr.size == 0:
        return arr, 1.0

    m = int(max(2, min(int(tail_points), arr.size)))
    tail_mean = float(np.mean(arr[-m:]))
    d_phi = abs(tail_mean - float(phi_ref))
    d_rho = abs(tail_mean - float(np.sqrt(2.0) * float(phi_ref)))
    scale = 1.0 if d_phi <= d_rho else (1.0 / np.sqrt(2.0))
    return arr * scale, float(scale)


def build_seed(
    solver: Any,
    seed_type: str,
    profiles_1d: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified 2D seed builder.

    seed_type options:
    - "O4_seed"
    - "O3_static_seed"
    - "O1_tau_seed"
    - "homogeneous_seed"
    """
    seed_type = seed_type.strip()
    r = np.asarray(solver.grid.r, dtype=float)
    tau = np.asarray(solver.grid.tau, dtype=float)
    R = r[:, None]
    T = np.abs(tau)[None, :]

    if seed_type == "O4_seed":
        p = profiles_1d["O4"]
        rho_e = np.sqrt(R * R + T * T)
        phi_raw = _interp_profile(p["r"], p["phi"], rho_e)
        phi_seed, _ = _normalize_profile_to_solver_phi_units(phi_raw, float(solver.rho0))
    elif seed_type == "O3_static_seed":
        p = profiles_1d["O3"]
        phi_r = _interp_profile(p["r"], p["phi"], r)
        phi_r, _ = _normalize_profile_to_solver_phi_units(phi_r, float(solver.rho0))
        phi_seed = np.repeat(phi_r[:, None], tau.size, axis=1)
    elif seed_type == "O1_tau_seed":
        p = profiles_1d["O1"]
        phi_t = _interp_profile(p["r"], p["phi"], np.abs(tau))
        phi_t, _ = _normalize_profile_to_solver_phi_units(phi_t, float(solver.rho0))
        phi_seed = np.repeat(phi_t[None, :], r.size, axis=0)
    elif seed_type == "homogeneous_seed":
        phi_seed = np.full((r.size, tau.size), float(solver.rho0), dtype=float)
    else:
        raise ValueError(f"Unknown seed_type={seed_type}")

    y = R * (phi_seed - float(solver.rho0))
    ybar = y.copy()
    x0 = solver.pack(y, ybar)
    return x0, phi_seed


def run_newton_with_eta_scan(
    solver: Any,
    x0: np.ndarray,
    Q_target: float,
    *,
    eta_bracket: Tuple[float, float] = (-0.15, 0.15),
    max_bracket_expansions: int = 8,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Run 2D Newton solve while matching target charge through eta0 scan."""
    a, b = float(eta_bracket[0]), float(eta_bracket[1])
    last_error: Exception | None = None
    for _ in range(max_bracket_expansions):
        try:
            sol, meta = solver.scan_eta0_to_match_Q(
                Q_target=Q_target,
                eta_bracket=(a, b),
                x0=x0,
                verbose=verbose,
                max_steps=26,
                tol_Q=1e-8,
            )
            meta["eta_bracket_used"] = (a, b)
            return sol, meta
        except Exception as exc:
            last_error = exc
            a *= 1.6
            b *= 1.6
    raise RuntimeError(f"Could not complete eta0 scan. Last error: {last_error}")


def summarize_solution(sol: Any) -> Dict[str, float]:
    """Compact diagnostics dictionary for a Bubble2DSolution."""
    q = float(np.real(sol.Q_tau0))
    return {
        "success": float(bool(sol.success)),
        "iterations": float(sol.iterations),
        "residual_norm": float(sol.residual_norm),
        "Q_tau0": q,
        "E_tau0": float(sol.E_tau0),
        "rho0": float(sol.rho0),
        "E_hom": float(sol.E_hom),
        "energy_ratio": float(sol.energy_ratio),
    }


def compute_static_energy_minus_hom_vs_tau(
    solver: Any,
    y: np.ndarray,
    ybar: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """
    Compute E_static(tau) - E_hom(static).

    E_static includes:
    - radial gradient term (d_r phi)(d_r phibar)
    - potential term U(rho)
    and excludes time-derivative/kinetic terms.
    """
    r = np.asarray(solver.grid.r, dtype=float)
    tau = np.asarray(solver.grid.tau, dtype=float)
    phi, phibar = solver.phi(y, ybar)

    rho_hom_phys = float(np.sqrt(2.0) * solver.rho0)
    U_hom = float(np.asarray(solver.U(np.array([rho_hom_phys]))).flat[0])
    E_hom_static = float(4.0 * np.pi * np.trapz(r * r * U_hom, x=r))

    E_static = np.zeros(tau.size, dtype=float)
    for i in range(tau.size):
        dphi_dr = np.gradient(phi[:, i], r, edge_order=2)
        dphibar_dr = np.gradient(phibar[:, i], r, edge_order=2)
        dphi_dr[0] = 0.0
        dphibar_dr[0] = 0.0
        u = np.maximum((phi[:, i] * phibar[:, i]).real, 0.0)
        rho_phys = np.sqrt(2.0 * u + float(getattr(solver.settings, "rho_eps", 0.0)))
        U_vals = np.asarray(solver.U(rho_phys), dtype=float)
        dens = (dphi_dr * dphibar_dr).real + U_vals
        E_static[i] = float(4.0 * np.pi * np.trapz(r * r * dens, x=r))

    return {
        "tau": tau,
        "E_static": E_static,
        "E_hom_static": E_hom_static,
        "delta_E_static": E_static - E_hom_static,
    }


def plot_seed_maps(
    r: np.ndarray,
    tau: np.ndarray,
    seed_map: np.ndarray,
    title_prefix: str,
) -> Tuple[Any, Any]:
    """Figure with seed map and the two canonical slices."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    im = axes[0].pcolormesh(r, tau, seed_map.T, shading="auto", cmap="viridis")
    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$\tau$")
    # Display tau from 0 (top) to -beta/2 (bottom).
    axes[0].set_ylim(float(np.max(tau)), float(np.min(tau)))
    axes[0].set_title(f"{title_prefix}: 2D seed")
    fig.colorbar(im, ax=axes[0], label=r"$\rho_{\mathrm{seed}}(r,\tau)$")

    axes[1].plot(tau, seed_map[0, :], label=r"seed at $r=0$")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(r"$\rho$")
    axes[1].set_title(r"Slice at $r=0$")
    axes[1].legend()

    axes[2].plot(r, seed_map[:, 0], label=r"seed at $\tau=0$")
    axes[2].set_xlabel(r"$r$")
    axes[2].set_ylabel(r"$\rho$")
    axes[2].set_title(r"Slice at $\tau=0$")
    axes[2].legend()

    fig.tight_layout()
    return fig, axes


def plot_solution_maps(
    r: np.ndarray,
    tau: np.ndarray,
    rho_map: np.ndarray,
    title_prefix: str,
) -> Tuple[Any, Any]:
    """Figure with final 2D solution map and main slices."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    im = axes[0].pcolormesh(r, tau, rho_map.T, shading="auto", cmap="magma")
    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$\tau$")
    # Display tau from 0 (top) to -beta/2 (bottom).
    axes[0].set_ylim(float(np.max(tau)), float(np.min(tau)))
    axes[0].set_title(f"{title_prefix}: final 2D solution")
    fig.colorbar(im, ax=axes[0], label=r"$\rho(r,\tau)$")

    axes[1].plot(tau, rho_map[0, :], label=r"2D at $r=0$")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(r"$\rho$")
    axes[1].set_title(r"Slice at $r=0$")
    axes[1].legend()

    axes[2].plot(r, rho_map[:, 0], label=r"2D at $\tau=0$")
    axes[2].set_xlabel(r"$r$")
    axes[2].set_ylabel(r"$\rho$")
    axes[2].set_title(r"Slice at $\tau=0$")
    axes[2].legend()
    fig.tight_layout()
    return fig, axes
