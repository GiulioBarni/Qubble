"""Minkowski evolution after analytic continuation from the 2D Euclidean saddle.

Evolves (φ, φ̄) on a spherical grid with velocity-Verlet. Initial data follow the
τ = 0 ghost reconstruction used in bounce2d and observables_2d.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.integrate import simpson
except Exception:  # pragma: no cover
    simpson = None


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class MinkowskiEvolutionConfig:
    """Configuration for the post-tunneling Minkowski evolution."""

    dt: Optional[float] = None
    cfl_prefactor: float = 0.20
    t_max: float = 10.0

    # Radial boundary handling
    enforce_origin_regular: bool = True
    outer_boundary_mode: str = "copy_neumann"  # "copy_neumann", "fixed", "none"
    phi_boundary_value: Optional[complex] = None
    phibar_boundary_value: Optional[complex] = None

    # Optional outer absorbing layer
    sponge_fraction: float = 0.10
    sponge_strength: float = 0.00   # physical damping rate gamma_max; 0 disables sponge
    sponge_power: float = 2.0

    # Potential reconstruction
    rho_eps: float = 1.0e-12

    # Optional projection back to the physical subspace
    enforce_conjugacy: bool = False

    # Output control
    store_every: int = 1
    snapshot_times: Optional[Tuple[float, ...]] = None
    verbose: bool = True


@dataclass
class MinkowskiSnapshot:
    """Single snapshot of the Minkowski evolution."""

    t: float
    phi: np.ndarray
    phibar: np.ndarray
    dotphi: np.ndarray
    dotphibar: np.ndarray
    rho_phys: np.ndarray
    beta: np.ndarray
    phase_angle: np.ndarray


@dataclass
class MinkowskiHistory:
    """Time-series output of the Minkowski evolution."""

    t: np.ndarray
    energy: np.ndarray
    charge: np.ndarray
    r: np.ndarray
    snapshots: List[MinkowskiSnapshot]
    metadata: Dict[str, Any]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


Array = np.ndarray
PotentialFn = Callable[[Array], Array]


def _integrate_spherical(r: Array, density: Array) -> float:
    """Return 4*pi * integral dr r^2 density(r)."""
    r = np.asarray(r, dtype=float).flatten()
    density = np.asarray(density)
    integrand = r * r * density
    if simpson is not None and r.size >= 3:
        return float(4.0 * np.pi * simpson(integrand, x=r))
    return float(4.0 * np.pi * np.trapz(integrand, x=r))


def _as_1d_real_grid(x: Array, name: str) -> Array:
    x = np.asarray(x, dtype=float).flatten()
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    return x


def _grid_spacing_summary(r: Array) -> Tuple[float, float, bool]:
    """Return (dr_mean, dr_max_dev, is_uniform)."""
    if r.size < 2:
        return 1.0, 0.0, True
    dr = np.diff(r)
    dr_mean = float(np.mean(dr))
    dev = float(np.max(np.abs(dr - dr_mean)))
    uniform = bool(dev <= 1.0e-10 * max(1.0, abs(dr_mean)))
    return dr_mean, dev, uniform


def _safe_divide(num: Array, den: Array, fill: float | complex = 0.0) -> Array:
    num = np.asarray(num)
    den = np.asarray(den)
    out = np.full_like(num, fill_value=fill, dtype=np.result_type(num, np.complex128))
    np.divide(num, den, out=out, where=(np.abs(den) > 0.0))
    return out


# -----------------------------------------------------------------------------
# Field reconstruction and polar diagnostics
# -----------------------------------------------------------------------------


def rho_phys_from_fields(phi: Array, phibar: Array, rho_eps: float = 1.0e-12) -> Array:
    """
    Physical modulus entering the potential.

    Projectively:
        rho_phys = sqrt(max(2 * Re(phi * phibar), 0) + rho_eps)

    This matches the 2D solver convention rho_phys = sqrt(2*u_pos + rho_eps),
    with u = Re(phi * phibar).
    """
    u = (np.asarray(phi) * np.asarray(phibar)).real
    u_pos = np.maximum(u, 0.0)
    return np.sqrt(u_pos * 2.0 + float(rho_eps))



def beta_from_fields(phi: Array, phibar: Array, rho_floor: float = 1.0e-30) -> Array:
    """
    Reconstruct the generalized beta variable from phi and phibar.

    We define beta by
        phi    = (rho_phys / sqrt(2)) * exp(+beta)
        phibar = (rho_phys / sqrt(2)) * exp(-beta)
    so formally
        beta = 0.5 * log(phi / phibar).

    This is the natural continuation of the Euclidean variable used in the
    project. For a physical Minkowski solution with phibar ≈ conj(phi), beta is
    approximately purely imaginary.

    The logarithm is branch-dependent; this routine is therefore intended for
    diagnostics, not for defining the evolution equations.
    """
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    ratio = _safe_divide(phi, phibar, fill=1.0 + 0.0j)
    beta = 0.5 * np.log(ratio)
    # Clean isolated singular points where both fields are tiny.
    mask_tiny = (np.abs(phi) * np.abs(phibar)) < float(rho_floor)
    beta = np.asarray(beta, dtype=np.complex128)
    beta[mask_tiny] = 0.0
    return beta



def phase_angle_from_fields(phi: Array) -> Array:
    """
    Physical phase angle extracted from phi alone.

    This is meaningful when the Minkowski solution remains close to the
    physical subspace phibar = conj(phi). The angle is unwrapped along the
    radial direction for readability.
    """
    phi = np.asarray(phi, dtype=np.complex128)
    ang = np.angle(phi)
    if ang.ndim == 1:
        return np.unwrap(ang)
    return np.unwrap(ang, axis=0)



def compute_polar_diagnostics(phi: Array, phibar: Array, rho_eps: float = 1.0e-12) -> Dict[str, Array]:
    """Return rho_phys, beta, and the usual phase angle extracted from phi."""
    return {
        "rho_phys": rho_phys_from_fields(phi, phibar, rho_eps=rho_eps),
        "beta": beta_from_fields(phi, phibar),
        "phase_angle": phase_angle_from_fields(phi),
    }


# -----------------------------------------------------------------------------
# Extraction of Minkowski initial data from the Euclidean 2D solver
# -----------------------------------------------------------------------------


def _tau_derivative_centered_from_solver(
    solver: Any,
    y: Array,
    ybar: Array,
    i_tau: int,
) -> Tuple[Array, Array]:
    """
    Compute BC-aware d/dtau of solver variables y and ybar at a single tau index.

    This uses the same ghost/twist logic as observables_2d and is therefore much
    more reliable than a naive finite difference taken directly on phi and phibar
    at the boundary tau = 0.
    """
    tau = _as_1d_real_grid(solver.grid.tau, "solver.grid.tau")
    dt = float(getattr(solver, "dt", getattr(solver.grid, "dtau", None)))
    if dt is None:
        raise ValueError("Could not determine tau spacing from solver.dt or solver.grid.dtau.")

    y_im1, y_ip1, yb_im1, yb_ip1 = solver._tau_neighbors(y, ybar, int(i_tau))

    if tau.size >= 2 and tau[1] > tau[0]:
        y_t = (y_ip1 - y_im1) / (2.0 * dt)
        yb_t = (yb_ip1 - yb_im1) / (2.0 * dt)
    else:
        y_t = (y_im1 - y_ip1) / (2.0 * dt)
        yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)
    return y_t, yb_t



def _phi_tau_from_solver_slice(
    solver: Any,
    y: Array,
    ybar: Array,
    i_tau: int,
) -> Tuple[Array, Array, Array, Array]:
    """
    Reconstruct phi, phibar, d_tau phi, and d_tau phibar on a given tau slice.

    This follows exactly the field reconstruction used in the 2D solver and in
    observables_2d.
    """
    r = _as_1d_real_grid(solver.grid.r, "solver.grid.r")
    tau = _as_1d_real_grid(solver.grid.tau, "solver.grid.tau")
    omega = float(getattr(solver, "omega"))
    rho0 = float(getattr(solver, "rho0"))

    phi, phibar = solver.phi(y, ybar)
    y_t, yb_t = _tau_derivative_centered_from_solver(solver, y, ybar, i_tau)

    y_tot = y[:, i_tau] + r * rho0
    yb_tot = ybar[:, i_tau] + r * rho0

    inv_r = np.zeros_like(r, dtype=float)
    inv_r[r != 0.0] = 1.0 / r[r != 0.0]

    exp_p = np.exp(+omega * tau[i_tau]) * inv_r
    exp_m = np.exp(-omega * tau[i_tau]) * inv_r

    phi_tau = exp_p * (y_t + omega * y_tot)
    phibar_tau = exp_m * (yb_t - omega * yb_tot)

    return phi[:, i_tau], phibar[:, i_tau], phi_tau, phibar_tau



def extract_initial_data_from_solver(
    solver: Any,
    solution: Any,
    *,
    index_tau: Optional[int] = None,
    enforce_conjugacy: bool = False,
    verbose: bool = True,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Extract Minkowski initial data from a 2D Euclidean solution.

    Parameters
    ----------
    solver:
        Bubble2DSolver-like object providing grid, phi(...), and _tau_neighbors(...).
    solution:
        Object exposing .y and .ybar arrays with shape (Nr, Nt).
    index_tau:
        Tau index of the initial slice. If None, choose the grid point closest to
        tau = 0.
    enforce_conjugacy:
        If True, project the extracted initial data to the physical subspace
        phibar = conj(phi) and dotphibar = conj(dotphi).
    verbose:
        Print basic diagnostics.

    Returns
    -------
    r, phi0, phibar0, dotphi0, dotphibar0
    """
    y = np.asarray(solution.y)
    ybar = np.asarray(solution.ybar)
    tau = _as_1d_real_grid(solver.grid.tau, "solver.grid.tau")
    r = _as_1d_real_grid(solver.grid.r, "solver.grid.r")

    if y.shape != ybar.shape:
        raise ValueError("solution.y and solution.ybar must have the same shape.")
    if y.shape != (r.size, tau.size):
        raise ValueError(
            f"Expected solution arrays of shape {(r.size, tau.size)}, got {y.shape}."
        )

    i0 = int(np.argmin(np.abs(tau))) if index_tau is None else int(index_tau)
    phi0, phibar0, phi_tau0, phibar_tau0 = _phi_tau_from_solver_slice(solver, y, ybar, i0)

    # Analytic continuation t = -i tau => d_t = +i d_tau.
    dotphi0 = 1j * np.asarray(phi_tau0, dtype=np.complex128)
    dotphibar0 = 1j * np.asarray(phibar_tau0, dtype=np.complex128)

    if enforce_conjugacy:
        phi0 = 0.5 * (phi0 + np.conj(phibar0))
        phibar0 = np.conj(phi0)
        dotphi0 = 0.5 * (dotphi0 + np.conj(dotphibar0))
        dotphibar0 = np.conj(dotphi0)

    if verbose:
        conj_mismatch = float(np.max(np.abs(phibar0 - np.conj(phi0))))
        vel_conj_mismatch = float(np.max(np.abs(dotphibar0 - np.conj(dotphi0))))
        print(
            f"[extract_initial_data_from_solver] tau index = {i0}, tau = {tau[i0]:+.6e}, "
            f"max|phibar0-conj(phi0)| = {conj_mismatch:.3e}, "
            f"max|dotphibar0-conj(dotphi0)| = {vel_conj_mismatch:.3e}"
        )

    return r, phi0, phibar0, dotphi0, dotphibar0



def extract_initial_data_from_fields(
    phi: Array,
    phibar: Array,
    tau_grid: Array,
    r_grid: Array,
    *,
    enforce_conjugacy: bool = False,
    verbose: bool = True,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Extract Minkowski initial data from already reconstructed Euclidean fields.

    Finite-difference τ derivative; invalid if τ = 0 is a ghost point — use extract_initial_data_from_solver() of the provided tau grid. If tau = 0
    is a boundary of the half-box, prefer extract_initial_data_from_solver().
    """
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    tau_grid = _as_1d_real_grid(tau_grid, "tau_grid")
    r_grid = _as_1d_real_grid(r_grid, "r_grid")

    if phi.shape != phibar.shape:
        raise ValueError("phi and phibar must have the same shape.")
    if phi.shape != (r_grid.size, tau_grid.size):
        raise ValueError("phi/phibar shape must be (Nr, Nt) matching r_grid and tau_grid.")

    i0 = int(np.argmin(np.abs(tau_grid)))
    phi0 = phi[:, i0].copy()
    phibar0 = phibar[:, i0].copy()

    if tau_grid.size >= 3 and 0 < i0 < tau_grid.size - 1:
        dtau = tau_grid[i0 + 1] - tau_grid[i0 - 1]
        phi_tau0 = (phi[:, i0 + 1] - phi[:, i0 - 1]) / dtau
        phibar_tau0 = (phibar[:, i0 + 1] - phibar[:, i0 - 1]) / dtau
    elif tau_grid.size >= 2 and i0 == 0:
        dtau = tau_grid[1] - tau_grid[0]
        phi_tau0 = (phi[:, 1] - phi[:, 0]) / dtau
        phibar_tau0 = (phibar[:, 1] - phibar[:, 0]) / dtau
    elif tau_grid.size >= 2 and i0 == tau_grid.size - 1:
        dtau = tau_grid[-1] - tau_grid[-2]
        phi_tau0 = (phi[:, -1] - phi[:, -2]) / dtau
        phibar_tau0 = (phibar[:, -1] - phibar[:, -2]) / dtau
    else:
        phi_tau0 = np.zeros_like(phi0)
        phibar_tau0 = np.zeros_like(phibar0)

    dotphi0 = 1j * phi_tau0
    dotphibar0 = 1j * phibar_tau0

    if enforce_conjugacy:
        phi0 = 0.5 * (phi0 + np.conj(phibar0))
        phibar0 = np.conj(phi0)
        dotphi0 = 0.5 * (dotphi0 + np.conj(dotphibar0))
        dotphibar0 = np.conj(dotphi0)

    if verbose:
        print(
            f"[extract_initial_data_from_fields] tau index = {i0}, tau = {tau_grid[i0]:+.6e}"
        )

    return r_grid.copy(), phi0, phibar0, dotphi0, dotphibar0


# -----------------------------------------------------------------------------
# Spatial derivatives and equations of motion
# -----------------------------------------------------------------------------


def radial_derivative(r: Array, f: Array) -> Array:
    """
    First radial derivative with centered differences in the interior.

    The origin is treated with the regularity condition df/dr = 0 at r = 0 when
    the grid starts exactly at the origin.
    """
    r = _as_1d_real_grid(r, "r")
    f = np.asarray(f, dtype=np.complex128)
    if f.shape != (r.size,):
        raise ValueError("radial_derivative expects a 1D field defined on r.")

    df = np.zeros_like(f, dtype=np.complex128)
    n = r.size
    if n == 1:
        return df
    if n == 2:
        slope = (f[1] - f[0]) / (r[1] - r[0])
        if np.isclose(r[0], 0.0):
            df[0] = 0.0
        else:
            df[0] = slope
        df[1] = slope
        return df

    df[1:-1] = (f[2:] - f[:-2]) / (r[2:] - r[:-2])

    if np.isclose(r[0], 0.0):
        df[0] = 0.0
    else:
        df[0] = (f[1] - f[0]) / (r[1] - r[0])

    df[-1] = (f[-1] - f[-2]) / (r[-1] - r[-2])
    return df



def _radial_fv_geometry(r: Array) -> Dict[str, Array]:
    """
    Build finite-volume geometry for a 1D spherical radial grid.

    Returns a dictionary with:
    - `r_edges`: N+1 shell edges, inferred from center coordinates.
    - `cell_volumes`: N shell volumes, 4*pi/3 * (r_{i+1/2}^3 - r_{i-1/2}^3).
    - `r_half`: N-1 interior edge radii between adjacent centers.
    - `edge_areas`: N-1 interior edge areas, 4*pi*r_{i+1/2}^2.
    - `dr_centers`: N-1 center spacings.

    This geometry is reused consistently both for:
    - the finite-volume Laplacian in the equations of motion,
    - and the edge-based gradient contribution in the discrete Hamiltonian.
    """
    r = _as_1d_real_grid(r, "r")
    n = r.size
    if n == 0:
        raise ValueError("Radial grid must be non-empty.")
    if n >= 2 and np.any(np.diff(r) <= 0.0):
        raise ValueError("Radial grid must be strictly increasing.")

    r_edges = np.zeros(n + 1, dtype=float)
    if n == 1:
        dr_ref = max(abs(float(r[0])), 1.0)
        left = float(r[0] - 0.5 * dr_ref)
        right = float(r[0] + 0.5 * dr_ref)
        if np.isclose(r[0], 0.0):
            left = 0.0
        r_edges[0] = max(0.0, left)
        r_edges[1] = max(r_edges[0] + 1.0e-15, right)
    else:
        r_edges[1:-1] = 0.5 * (r[:-1] + r[1:])
        left = float(r[0] - 0.5 * (r[1] - r[0]))
        right = float(r[-1] + 0.5 * (r[-1] - r[-2]))
        if np.isclose(r[0], 0.0):
            left = 0.0
        r_edges[0] = max(0.0, left)
        r_edges[-1] = max(r_edges[-2] + 1.0e-15, right)

    if np.any(np.diff(r_edges) <= 0.0):
        raise ValueError("Inferred radial shell edges are not strictly increasing.")

    cell_volumes = (4.0 * np.pi / 3.0) * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    cell_volumes = np.asarray(cell_volumes, dtype=float)
    cell_volumes[cell_volumes <= 0.0] = 1.0e-30

    if n >= 2:
        dr_centers = np.diff(r)
        r_half = 0.5 * (r[:-1] + r[1:])
        edge_areas = 4.0 * np.pi * r_half * r_half
    else:
        dr_centers = np.zeros(0, dtype=float)
        r_half = np.zeros(0, dtype=float)
        edge_areas = np.zeros(0, dtype=float)

    return {
        "r_edges": r_edges,
        "cell_volumes": cell_volumes,
        "r_half": r_half,
        "edge_areas": edge_areas,
        "dr_centers": dr_centers,
    }


def _radial_edge_gradients(r: Array, f: Array) -> Array:
    """Return edge gradients (f[i+1]-f[i])/(r[i+1]-r[i]) on interior radial edges."""
    r = _as_1d_real_grid(r, "r")
    f = np.asarray(f, dtype=np.complex128)
    if f.shape != (r.size,):
        raise ValueError("Field must be a 1D array on the radial grid.")
    if r.size <= 1:
        return np.zeros(0, dtype=np.complex128)
    dr = np.diff(r)
    if np.any(dr <= 0.0):
        raise ValueError("Radial grid must be strictly increasing.")
    return (f[1:] - f[:-1]) / dr


def _gradient_energy_from_edges(
    r: Array,
    phi: Array,
    phibar: Array,
) -> Tuple[float, Array, Dict[str, Array]]:
    """
    Compute the discrete gradient energy from edge gradients.

    The discrete term is:
        E_grad = sum_{i+1/2} A_{i+1/2} * Re(g_phi * g_phibar) * dr_i,
    with A_{i+1/2} = 4*pi*r_{i+1/2}^2 and g=(delta field)/(delta r).
    """
    r = _as_1d_real_grid(r, "r")
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    if phi.shape != (r.size,) or phibar.shape != (r.size,):
        raise ValueError("phi and phibar must be 1D arrays on the radial grid.")

    geom = _radial_fv_geometry(r)
    if r.size <= 1:
        return 0.0, np.zeros(0, dtype=float), geom

    g_phi = _radial_edge_gradients(r, phi)
    g_phibar = _radial_edge_gradients(r, phibar)
    edge_density = (g_phi * g_phibar).real
    edge_energy = geom["edge_areas"] * edge_density * geom["dr_centers"]
    return float(np.sum(edge_energy)), np.asarray(edge_energy, dtype=float), geom


def _gradient_energy_density_cells(
    r: Array,
    phi: Array,
    phibar: Array,
) -> Tuple[Array, float]:
    """
    Return a cell-centered gradient-energy density consistent with edge energy.

    Edge contributions are split half/half between adjacent cells. The resulting
    per-cell energies are divided by shell volumes to obtain densities.
    """
    r = _as_1d_real_grid(r, "r")
    n = r.size
    e_grad_total, edge_energy, geom = _gradient_energy_from_edges(r, phi, phibar)
    cell_energy = np.zeros(n, dtype=float)
    if n >= 2:
        cell_energy[0] += 0.5 * edge_energy[0]
        cell_energy[-1] += 0.5 * edge_energy[-1]
    if n >= 3:
        cell_energy[1:-1] += 0.5 * (edge_energy[:-1] + edge_energy[1:])
    grad_density = cell_energy / geom["cell_volumes"]
    return grad_density, e_grad_total


def spherical_laplacian(r: Array, f: Array) -> Array:
    """
    Finite-volume spherical radial Laplacian consistent with edge energy.

    We discretize:
        nabla_r^2 f = (1/r^2) d_r (r^2 d_r f),
    by shell fluxes. For each cell i:
        V_i * lap_i = F_{i+1/2} - F_{i-1/2},
        F_{i+1/2} = 4*pi*r_{i+1/2}^2 * g_{i+1/2},
        g_{i+1/2} = (f_{i+1}-f_i)/(r_{i+1}-r_i).

    At the origin the left flux is exactly zero by area factor A(r=0)=0.
    At the outer side this corresponds to a natural zero-flux boundary for the
    variational structure when no extra ghost flux is supplied.
    """
    r = _as_1d_real_grid(r, "r")
    f = np.asarray(f, dtype=np.complex128)
    if f.shape != (r.size,):
        raise ValueError("spherical_laplacian expects a 1D field defined on r.")

    n = r.size
    lap = np.zeros_like(f, dtype=np.complex128)
    if n <= 1:
        return lap

    geom = _radial_fv_geometry(r)
    g = _radial_edge_gradients(r, f)
    flux = geom["edge_areas"] * g
    vol = geom["cell_volumes"]

    # Left boundary cell: only outgoing interior edge contributes.
    lap[0] = flux[0] / vol[0]

    # Interior cells: conservative flux difference.
    if n >= 3:
        lap[1:-1] = (flux[1:] - flux[:-1]) / vol[1:-1]

    # Right boundary cell: natural zero-flux closure at the outer edge.
    lap[-1] = -flux[-1] / vol[-1]
    return lap



def potential_force_coefficient(
    phi: Array,
    phibar: Array,
    dU: PotentialFn,
    rho_eps: float = 1.0e-12,
) -> Array:
    """
    Return the coefficient C(r) such that
        dV/dphibar = C * phi,
        dV/dphi    = C * phibar,
    for V = U(rho_phys) with rho_phys = sqrt(2 * Re(phi * phibar) + rho_eps).

    Since
        drho_phys / dphibar = phi / rho_phys,
    one has
        dV/dphibar = (dU/drho_phys) * phi / rho_phys.

    Therefore the correct coefficient is
        C = dU(rho_phys) / rho_phys.

    This differs by a factor of 2 from formulas appropriate to conventions where
    rho = sqrt(phi * phibar) instead of sqrt(2 * phi * phibar).
    """
    rho = rho_phys_from_fields(phi, phibar, rho_eps=rho_eps)
    dU_val = np.asarray(dU(rho), dtype=np.complex128)
    coeff = _safe_divide(dU_val, rho, fill=0.0)
    coeff[~np.isfinite(coeff)] = 0.0
    return coeff



def minkowski_acceleration(
    phi: Array,
    phibar: Array,
    r_grid: Array,
    dU: PotentialFn,
    rho_eps: float = 1.0e-12,
) -> Tuple[Array, Array]:
    """
    Compute the Minkowski accelerations
        d_t^2 phi    = nabla_r^2 phi    - dV/dphibar,
        d_t^2 phibar = nabla_r^2 phibar - dV/dphi.
    """
    r = _as_1d_real_grid(r_grid, "r_grid")
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    if phi.shape != (r.size,) or phibar.shape != (r.size,):
        raise ValueError("phi and phibar must be 1D arrays on the radial grid.")

    lap_phi = spherical_laplacian(r, phi)
    lap_phibar = spherical_laplacian(r, phibar)
    coeff = potential_force_coefficient(phi, phibar, dU=dU, rho_eps=rho_eps)

    accel_phi = lap_phi - coeff * phi
    accel_phibar = lap_phibar - coeff * phibar
    return accel_phi, accel_phibar


# -----------------------------------------------------------------------------
# Boundary conditions and damping
# -----------------------------------------------------------------------------


def build_sponge_profile(r: Array, fraction: float, strength: float, power: float = 2.0) -> Array:
    """
    Build a smooth outer damping profile gamma(r).

    The sponge is zero in the bulk and rises smoothly in the last `fraction` of
    the box up to `strength` at the boundary.
    """
    r = _as_1d_real_grid(r, "r")
    fraction = float(fraction)
    strength = float(strength)
    power = float(power)

    gamma = np.zeros_like(r, dtype=float)
    if strength <= 0.0 or fraction <= 0.0:
        return gamma

    r_max = float(r[-1])
    r_start = (1.0 - fraction) * r_max
    if r_max <= 0.0:
        return gamma

    mask = r >= r_start
    if np.any(mask):
        x = (r[mask] - r_start) / max(r_max - r_start, 1.0e-30)
        gamma[mask] = strength * np.power(np.clip(x, 0.0, 1.0), power)
    return gamma



def apply_boundary_conditions(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
    r_grid: Array,
    *,
    enforce_origin_regular: bool = True,
    outer_boundary_mode: str = "copy_neumann",
    phi_boundary_value: Optional[complex] = None,
    phibar_boundary_value: Optional[complex] = None,
) -> Tuple[Array, Array, Array, Array]:
    """
    Apply simple boundary conditions after each step.

    Supported outer modes:
    - "copy_neumann": copy the penultimate point into the last point.
    - "fixed": pin the last point to the provided boundary values.
    - "none": do nothing.
    """
    r = _as_1d_real_grid(r_grid, "r_grid")
    phi = np.asarray(phi, dtype=np.complex128).copy()
    phibar = np.asarray(phibar, dtype=np.complex128).copy()
    dotphi = np.asarray(dotphi, dtype=np.complex128).copy()
    dotphibar = np.asarray(dotphibar, dtype=np.complex128).copy()

    if phi.shape != (r.size,) or phibar.shape != (r.size,):
        raise ValueError("Boundary application expects 1D fields on the radial grid.")

    if enforce_origin_regular and r.size >= 2 and np.isclose(r[0], 0.0):
        phi[0] = phi[1]
        phibar[0] = phibar[1]
        dotphi[0] = dotphi[1]
        dotphibar[0] = dotphibar[1]

    mode = str(outer_boundary_mode).lower()
    if mode == "copy_neumann" and r.size >= 2:
        phi[-1] = phi[-2]
        phibar[-1] = phibar[-2]
        dotphi[-1] = dotphi[-2]
        dotphibar[-1] = dotphibar[-2]
    elif mode == "fixed":
        if phi_boundary_value is None or phibar_boundary_value is None:
            raise ValueError("Fixed outer boundary mode requires phi_boundary_value and phibar_boundary_value.")
        phi[-1] = complex(phi_boundary_value)
        phibar[-1] = complex(phibar_boundary_value)
        dotphi[-1] = 0.0
        dotphibar[-1] = 0.0
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown outer_boundary_mode = {outer_boundary_mode!r}.")

    return phi, phibar, dotphi, dotphibar



def enforce_physical_conjugacy(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
) -> Tuple[Array, Array, Array, Array]:
    """
    Project the state back to the physical subspace phibar = conj(phi).

    This is optional. It can be useful if the initial data are physical and one
    wants to suppress purely numerical drift away from the physical manifold.
    """
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    dotphi = np.asarray(dotphi, dtype=np.complex128)
    dotphibar = np.asarray(dotphibar, dtype=np.complex128)

    phi_proj = 0.5 * (phi + np.conj(phibar))
    dotphi_proj = 0.5 * (dotphi + np.conj(dotphibar))
    return phi_proj, np.conj(phi_proj), dotphi_proj, np.conj(dotphi_proj)


# -----------------------------------------------------------------------------
# Observables in Minkowski time
# -----------------------------------------------------------------------------


def compute_energy_density(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
    r_grid: Array,
    U: PotentialFn,
    rho_eps: float = 1.0e-12,
) -> Array:
    """
    Cell-centered energy density consistent with the FV/edge Hamiltonian.

    We keep the density output for plotting, but the total energy is computed by
    shell-volume summation in `compute_energy` to remain fully consistent with the
    finite-volume spatial discretization.
    """
    r = _as_1d_real_grid(r_grid, "r_grid")
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    dotphi = np.asarray(dotphi, dtype=np.complex128)
    dotphibar = np.asarray(dotphibar, dtype=np.complex128)

    if phi.shape != (r.size,):
        raise ValueError("compute_energy_density expects 1D fields on r_grid.")

    kinetic = (dotphi * dotphibar).real
    gradient, _ = _gradient_energy_density_cells(r, phi, phibar)
    rho = rho_phys_from_fields(phi, phibar, rho_eps=rho_eps)
    potential = np.asarray(U(rho), dtype=float).reshape(r.shape)
    return kinetic + gradient + potential



def compute_energy(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
    r_grid: Array,
    U: PotentialFn,
    rho_eps: float = 1.0e-12,
) -> float:
    """
    Discrete Minkowski Hamiltonian consistent with FV spatial operator.

    The kinetic and potential parts are summed on shell volumes, while the
    gradient part is the edge-based contribution built from the same objects
    used by the finite-volume Laplacian.
    """
    r = _as_1d_real_grid(r_grid, "r_grid")
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    dotphi = np.asarray(dotphi, dtype=np.complex128)
    dotphibar = np.asarray(dotphibar, dtype=np.complex128)
    if phi.shape != (r.size,) or phibar.shape != (r.size,):
        raise ValueError("compute_energy expects phi/phibar on r_grid.")
    if dotphi.shape != (r.size,) or dotphibar.shape != (r.size,):
        raise ValueError("compute_energy expects dotphi/dotphibar on r_grid.")

    geom = _radial_fv_geometry(r)
    kinetic = (dotphi * dotphibar).real
    rho = rho_phys_from_fields(phi, phibar, rho_eps=rho_eps)
    potential = np.asarray(U(rho), dtype=float).reshape(r.shape)
    e_grad_total, _, _ = _gradient_energy_from_edges(r, phi, phibar)
    e_kin_pot = float(np.sum(geom["cell_volumes"] * (kinetic + potential)))
    return e_kin_pot + e_grad_total



def compute_charge_density(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
) -> Array:
    """
    Minkowski Noether charge density in the conventions of the project.

    The correct continuation of the Euclidean convention q_E = Re(phibar*d_tau phi
    - phi*d_tau phibar) is
        q_M = -i * (phibar*d_t phi - d_t phibar*phi),
    whose real part is kept explicitly for numerical stability.

    For a physical field
        phi = (rho_phys/sqrt(2)) * exp(i theta),
        phibar = conj(phi),
    this gives
        q_M = rho_phys^2 * d_t theta.
    """
    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    dotphi = np.asarray(dotphi, dtype=np.complex128)
    dotphibar = np.asarray(dotphibar, dtype=np.complex128)
    return (-1j * (phibar * dotphi - dotphibar * phi)).real



def compute_charge(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
    r_grid: Array,
) -> float:
    """Spherical U(1) charge Q = 4*pi * integral dr r^2 q(r)."""
    return _integrate_spherical(_as_1d_real_grid(r_grid, "r_grid"), compute_charge_density(phi, phibar, dotphi, dotphibar))



def compute_observables(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
    r_grid: Array,
    U: PotentialFn,
    rho_eps: float = 1.0e-12,
) -> Dict[str, Any]:
    """Bundle the most common observables and polar diagnostics for a single time slice."""
    rho = rho_phys_from_fields(phi, phibar, rho_eps=rho_eps)
    beta = beta_from_fields(phi, phibar)
    phase = phase_angle_from_fields(phi)
    e_r = compute_energy_density(phi, phibar, dotphi, dotphibar, r_grid, U, rho_eps=rho_eps)
    q_r = compute_charge_density(phi, phibar, dotphi, dotphibar)
    return {
        "rho_phys": rho,
        "beta": beta,
        "phase_angle": phase,
        "energy_density": e_r,
        "charge_density": q_r,
        "energy": compute_energy(phi, phibar, dotphi, dotphibar, r_grid, U, rho_eps=rho_eps),
        "charge": _integrate_spherical(r_grid, q_r),
    }


def summarize_conservation(
    t: Array,
    energy: Array,
    charge: Array,
) -> Dict[str, float]:
    """Return compact absolute/relative drift diagnostics for E and Q."""
    t = np.asarray(t, dtype=float).flatten()
    energy = np.asarray(energy, dtype=float).flatten()
    charge = np.asarray(charge, dtype=float).flatten()
    if t.size == 0 or energy.size == 0 or charge.size == 0:
        raise ValueError("summarize_conservation expects non-empty arrays.")
    if not (t.size == energy.size == charge.size):
        raise ValueError("t, energy, and charge must have the same length.")

    e0 = float(energy[0])
    q0 = float(charge[0])
    e_abs = float(np.max(np.abs(energy - e0)))
    q_abs = float(np.max(np.abs(charge - q0)))
    e_rel = e_abs / max(abs(e0), 1.0e-30)
    q_rel = q_abs / max(abs(q0), 1.0e-30)
    return {
        "t_end": float(t[-1]),
        "e0": e0,
        "q0": q0,
        "max_abs_energy_drift": e_abs,
        "max_rel_energy_drift": e_rel,
        "max_abs_charge_drift": q_abs,
        "max_rel_charge_drift": q_rel,
    }


def compare_energy_drift_across_dt(
    r_grid: Array,
    phi0: Array,
    phibar0: Array,
    dotphi0: Array,
    dotphibar0: Array,
    U: PotentialFn,
    dU: PotentialFn,
    dt_values: Sequence[float],
    config_template: Optional[MinkowskiEvolutionConfig] = None,
) -> List[Dict[str, float]]:
    """
    Compare drift for several dt:
    run the same setup for multiple dt values and report conservation drifts.
    """
    cfg0 = MinkowskiEvolutionConfig() if config_template is None else config_template
    reports: List[Dict[str, float]] = []
    for dt in dt_values:
        cfg_dt = replace(cfg0, dt=float(dt), verbose=False)
        hist = run_minkowski_evolution(
            r_grid,
            phi0,
            phibar0,
            dotphi0,
            dotphibar0,
            U,
            dU,
            config=cfg_dt,
        )
        rep = summarize_conservation(hist.t, hist.energy, hist.charge)
        rep["dt"] = float(dt)
        reports.append(rep)
    return reports


# -----------------------------------------------------------------------------
# Time stepping
# -----------------------------------------------------------------------------


def velocity_verlet_step(
    phi: Array,
    phibar: Array,
    dotphi: Array,
    dotphibar: Array,
    r_grid: Array,
    dt: float,
    dU: PotentialFn,
    *,
    rho_eps: float = 1.0e-12,
    sponge_profile: Optional[Array] = None,
    enforce_origin_regular: bool = True,
    outer_boundary_mode: str = "copy_neumann",
    phi_boundary_value: Optional[complex] = None,
    phibar_boundary_value: Optional[complex] = None,
    enforce_conjugacy: bool = False,
) -> Tuple[Array, Array, Array, Array]:
    """Advance the Minkowski evolution by one second-order velocity-Verlet step."""
    r = _as_1d_real_grid(r_grid, "r_grid")
    dt = float(dt)

    phi = np.asarray(phi, dtype=np.complex128)
    phibar = np.asarray(phibar, dtype=np.complex128)
    dotphi = np.asarray(dotphi, dtype=np.complex128)
    dotphibar = np.asarray(dotphibar, dtype=np.complex128)

    acc_phi, acc_phibar = minkowski_acceleration(phi, phibar, r, dU=dU, rho_eps=rho_eps)

    phi_new = phi + dt * dotphi + 0.5 * dt * dt * acc_phi
    phibar_new = phibar + dt * dotphibar + 0.5 * dt * dt * acc_phibar

    dotphi_half = dotphi + 0.5 * dt * acc_phi
    dotphibar_half = dotphibar + 0.5 * dt * acc_phibar

    phi_new, phibar_new, dotphi_half, dotphibar_half = apply_boundary_conditions(
        phi_new,
        phibar_new,
        dotphi_half,
        dotphibar_half,
        r,
        enforce_origin_regular=enforce_origin_regular,
        outer_boundary_mode=outer_boundary_mode,
        phi_boundary_value=phi_boundary_value,
        phibar_boundary_value=phibar_boundary_value,
    )

    acc_new_phi, acc_new_phibar = minkowski_acceleration(phi_new, phibar_new, r, dU=dU, rho_eps=rho_eps)

    dotphi_new = dotphi_half + 0.5 * dt * acc_new_phi
    dotphibar_new = dotphibar_half + 0.5 * dt * acc_new_phibar

    if sponge_profile is not None:
        gamma = np.asarray(sponge_profile, dtype=float)
        if gamma.shape != r.shape:
            raise ValueError("sponge_profile must have the same shape as r_grid.")
        damp = np.exp(-gamma * dt)
        dotphi_new *= damp
        dotphibar_new *= damp

    phi_new, phibar_new, dotphi_new, dotphibar_new = apply_boundary_conditions(
        phi_new,
        phibar_new,
        dotphi_new,
        dotphibar_new,
        r,
        enforce_origin_regular=enforce_origin_regular,
        outer_boundary_mode=outer_boundary_mode,
        phi_boundary_value=phi_boundary_value,
        phibar_boundary_value=phibar_boundary_value,
    )

    if enforce_conjugacy:
        phi_new, phibar_new, dotphi_new, dotphibar_new = enforce_physical_conjugacy(
            phi_new, phibar_new, dotphi_new, dotphibar_new
        )

    return phi_new, phibar_new, dotphi_new, dotphibar_new


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def run_minkowski_evolution(
    r_grid: Array,
    phi0: Array,
    phibar0: Array,
    dotphi0: Array,
    dotphibar0: Array,
    U: PotentialFn,
    dU: PotentialFn,
    config: Optional[MinkowskiEvolutionConfig] = None,
) -> MinkowskiHistory:
    """
    Evolve the post-tunneling initial data in Minkowski time.

    Parameters
    ----------
    r_grid, phi0, phibar0, dotphi0, dotphibar0:
        Initial conditions on the radial grid.
    U, dU:
        Potential and first derivative with respect to rho_phys.
    config:
        Evolution settings.

    Returns
    -------
    MinkowskiHistory
        Time series of energy and charge, plus optional snapshots.
    """
    cfg = MinkowskiEvolutionConfig() if config is None else config

    r = _as_1d_real_grid(r_grid, "r_grid")
    phi = np.asarray(phi0, dtype=np.complex128).copy()
    phibar = np.asarray(phibar0, dtype=np.complex128).copy()
    dotphi = np.asarray(dotphi0, dtype=np.complex128).copy()
    dotphibar = np.asarray(dotphibar0, dtype=np.complex128).copy()

    for arr, name in ((phi, "phi0"), (phibar, "phibar0"), (dotphi, "dotphi0"), (dotphibar, "dotphibar0")):
        if arr.shape != (r.size,):
            raise ValueError(f"{name} must have shape {(r.size,)}, got {arr.shape}.")

    dr_mean, dr_dev, dr_uniform = _grid_spacing_summary(r)
    dt = float(cfg.dt) if cfg.dt is not None else float(cfg.cfl_prefactor * dr_mean)
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")

    n_steps = int(np.ceil(float(cfg.t_max) / dt))
    t_arr = np.linspace(0.0, n_steps * dt, n_steps + 1)

    if cfg.verbose:
        print(
            f"[run_minkowski_evolution] dr_mean = {dr_mean:.6e}, dt = {dt:.6e}, "
            f"n_steps = {n_steps}, t_max(actual) = {t_arr[-1]:.6e}"
        )
        if not dr_uniform:
            print(
                f"[run_minkowski_evolution] WARNING: radial grid is not perfectly uniform "
                f"(max |dr-dr_mean| = {dr_dev:.3e})."
            )

    sponge_profile = build_sponge_profile(
        r,
        fraction=cfg.sponge_fraction,
        strength=cfg.sponge_strength,
        power=cfg.sponge_power,
    )

    phi, phibar, dotphi, dotphibar = apply_boundary_conditions(
        phi,
        phibar,
        dotphi,
        dotphibar,
        r,
        enforce_origin_regular=cfg.enforce_origin_regular,
        outer_boundary_mode=cfg.outer_boundary_mode,
        phi_boundary_value=cfg.phi_boundary_value,
        phibar_boundary_value=cfg.phibar_boundary_value,
    )
    if cfg.enforce_conjugacy:
        phi, phibar, dotphi, dotphibar = enforce_physical_conjugacy(phi, phibar, dotphi, dotphibar)

    energy = np.zeros(n_steps + 1, dtype=float)
    charge = np.zeros(n_steps + 1, dtype=float)
    energy[0] = compute_energy(phi, phibar, dotphi, dotphibar, r, U, rho_eps=cfg.rho_eps)
    charge[0] = compute_charge(phi, phibar, dotphi, dotphibar, r)

    snapshots: List[MinkowskiSnapshot] = []
    snapshot_times = tuple(sorted(cfg.snapshot_times or ()))
    next_snapshot = 0

    def maybe_store_snapshot(current_t: float, step_index: int) -> None:
        nonlocal next_snapshot, snapshots
        save_by_stride = (cfg.store_every > 0 and step_index % int(cfg.store_every) == 0)
        save_by_time = False
        while next_snapshot < len(snapshot_times) and current_t >= snapshot_times[next_snapshot] - 1.0e-15:
            save_by_time = True
            next_snapshot += 1
        if not (save_by_stride or save_by_time):
            return
        pol = compute_polar_diagnostics(phi, phibar, rho_eps=cfg.rho_eps)
        snapshots.append(
            MinkowskiSnapshot(
                t=float(current_t),
                phi=phi.copy(),
                phibar=phibar.copy(),
                dotphi=dotphi.copy(),
                dotphibar=dotphibar.copy(),
                rho_phys=np.asarray(pol["rho_phys"]).copy(),
                beta=np.asarray(pol["beta"]).copy(),
                phase_angle=np.asarray(pol["phase_angle"]).copy(),
            )
        )

    maybe_store_snapshot(0.0, 0)

    for istep in range(1, n_steps + 1):
        phi, phibar, dotphi, dotphibar = velocity_verlet_step(
            phi,
            phibar,
            dotphi,
            dotphibar,
            r,
            dt,
            dU,
            rho_eps=cfg.rho_eps,
            sponge_profile=sponge_profile,
            enforce_origin_regular=cfg.enforce_origin_regular,
            outer_boundary_mode=cfg.outer_boundary_mode,
            phi_boundary_value=cfg.phi_boundary_value,
            phibar_boundary_value=cfg.phibar_boundary_value,
            enforce_conjugacy=cfg.enforce_conjugacy,
        )

        energy[istep] = compute_energy(phi, phibar, dotphi, dotphibar, r, U, rho_eps=cfg.rho_eps)
        charge[istep] = compute_charge(phi, phibar, dotphi, dotphibar, r)
        maybe_store_snapshot(float(t_arr[istep]), istep)

    metadata: Dict[str, Any] = {
        "dt": dt,
        "dr_mean": dr_mean,
        "dr_uniform": dr_uniform,
        "rho_eps": float(cfg.rho_eps),
        "energy_discretization": "finite_volume_edge_hamiltonian_v1",
        "outer_boundary_mode": str(cfg.outer_boundary_mode),
        "sponge_fraction": float(cfg.sponge_fraction),
        "sponge_strength": float(cfg.sponge_strength),
        "enforce_conjugacy": bool(cfg.enforce_conjugacy),
    }

    if cfg.verbose:
        rep = summarize_conservation(t_arr, energy, charge)
        conj_final = float(np.max(np.abs(phibar - np.conj(phi))))
        print(
            "[run_minkowski_evolution] "
            f"max|Delta E| = {rep['max_abs_energy_drift']:.3e}, "
            f"max|Delta E/E0| = {rep['max_rel_energy_drift']:.3e}"
        )
        print(
            "[run_minkowski_evolution] "
            f"max|Delta Q| = {rep['max_abs_charge_drift']:.3e}, "
            f"max|Delta Q/Q0| = {rep['max_rel_charge_drift']:.3e}"
        )
        print(f"[run_minkowski_evolution] final max|phibar-conj(phi)| = {conj_final:.3e}")
        if cfg.sponge_strength > 0.0:
            print("[run_minkowski_evolution] NOTE: the sponge breaks exact energy and charge conservation.")
        if cfg.outer_boundary_mode != "none":
            print("[run_minkowski_evolution] NOTE: outer boundary handling can also induce small conservation drift.")
        print(
            "[run_minkowski_evolution] NOTE: FV Laplacian/energy are mutually consistent; "
            "residual energy drift is mainly from time stepping and boundary effects."
        )

    return MinkowskiHistory(
        t=t_arr,
        energy=energy,
        charge=charge,
        r=r,
        snapshots=snapshots,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Wrappers
# -----------------------------------------------------------------------------


def run_minkowski_evolution_from_solver(
    solver: Any,
    solution: Any,
    U: PotentialFn,
    dU: PotentialFn,
    config: Optional[MinkowskiEvolutionConfig] = None,
    *,
    index_tau: Optional[int] = None,
    extract_enforce_conjugacy: bool = False,
) -> MinkowskiHistory:
    """
    
    extract initial data from the Euclidean 2D solution and evolve them.
    """
    r, phi0, phibar0, dotphi0, dotphibar0 = extract_initial_data_from_solver(
        solver,
        solution,
        index_tau=index_tau,
        enforce_conjugacy=extract_enforce_conjugacy,
        verbose=(config.verbose if config is not None else True),
    )
    return run_minkowski_evolution(
        r,
        phi0,
        phibar0,
        dotphi0,
        dotphibar0,
        U,
        dU,
        config=config,
    )


__all__ = [
    "MinkowskiEvolutionConfig",
    "MinkowskiHistory",
    "MinkowskiSnapshot",
    "rho_phys_from_fields",
    "beta_from_fields",
    "phase_angle_from_fields",
    "compute_polar_diagnostics",
    "extract_initial_data_from_solver",
    "extract_initial_data_from_fields",
    "radial_derivative",
    "spherical_laplacian",
    "minkowski_acceleration",
    "compute_energy_density",
    "compute_energy",
    "compute_charge_density",
    "compute_charge",
    "compute_observables",
    "summarize_conservation",
    "compare_energy_drift_across_dt",
    "velocity_verlet_step",
    "run_minkowski_evolution",
    "run_minkowski_evolution_from_solver",
]
