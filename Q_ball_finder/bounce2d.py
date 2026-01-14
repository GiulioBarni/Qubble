"""
2D Q-ball bounce solver built from the notebook's "Second try" section.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

from .ansatz import AnsatzResult, build_negative_mode_ansatz
from .grid import (
    RadialTimeGrid,
    build_grid,
    pack_fields,
    phi_from_y,
    phi_from_ybar,
    unpack_fields,
)
from .nr_solver import NewtonResult, newton_solve
from .observables2d import compute_charge, compute_energy
from .potentials import (
    LogisticPotentialParams,
    logistic_potential_chi,
    logistic_potential_rho,
)
from .profiles import QBallProfile, UnstableMode, compute_unstable_mode, solve_qball_profile


@dataclass
class QBall2DSettings:
    Nr: int = 150
    Ntau: int = 300
    Lr: float = 20.0
    beta: float = 40.0
    omega_tilde: Optional[float] = None
    eta0: Optional[float] = None
    ansatz_amplitude: float = 2.0
    tau_center: Optional[float] = None
    cosh_scale: Optional[float] = None
    envelope_width: Optional[float] = None
    flip_mode_sign: bool = False
    ansatz_center_at_cloud: bool = False
    ansatz_decrease_towards_zero: bool = False
    newton_tol: float = 1e-8
    newton_max_iter: int = 35
    damping: float = 1.0
    energy_reference: Optional[float] = None
    newton_verbose: bool = False


@dataclass
class QBall2DSolution:
    settings: QBall2DSettings
    grid: RadialTimeGrid
    profile: QBallProfile
    unstable_mode: UnstableMode
    ansatz: Optional[AnsatzResult]
    newton: NewtonResult
    y: np.ndarray
    ybar: np.ndarray
    phi: np.ndarray
    phibar: np.ndarray
    charge: Optional[float] = None  # Charge Q at slice i=0 (Eq. 3.9)
    charge_iter: Optional[float] = None  # Charge computed during Newton iteration
    energy: Optional[float] = None  # Energy E at slice i=0 (Eq. 2.4)
    energy_ratio: Optional[float] = None  # energy / E_Q




class QBall2DSolver:
    def __init__(
        self,
        params: LogisticPotentialParams,
        omega: float,
        settings: QBall2DSettings,
        grid: RadialTimeGrid,
    ):
        self.params = params
        self.omega = float(omega)
        self.settings = settings
        self.grid = grid
        self.Nr = grid.Nr
        self.Ntau = grid.Ntau
        self.dr2 = grid.dr * grid.dr
        self.dt = grid.dtau
        self.dt2 = self.dt * self.dt
        self.last_iteration_charge: Optional[float] = None

        self.omega_tilde = settings.omega_tilde if settings.omega_tilde is not None else self.omega
        self.eta0 = (
            settings.eta0
            if settings.eta0 is not None
            else settings.beta * (self.omega_tilde - self.omega)
        )

        _, self.Vprime, self.Vsecond = logistic_potential_rho(params)

    # ----------------------------------
    # Helpers
    # ----------------------------------

    def _unpack(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return unpack_fields(vec, self.Nr, self.Ntau)

    def _pack(self, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
        return pack_fields(y, ybar)

    # Backwards-compatible aliases used in notebooks / external scripts
    def unpack(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._unpack(vec)

    def pack(self, y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
        return self._pack(y, ybar)

    def _compute_phi(self, y: np.ndarray, ybar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = phi_from_y(y, self.grid, self.omega)
        phibar = phi_from_ybar(ybar, self.grid, self.omega)
        return phi, phibar

    def compute_charge(self, y: np.ndarray, ybar: np.ndarray) -> float:
        """Compute charge using Eq.(3.9): Q = ∫ d^3x [ phibar ∂τ phi - phi ∂τ phibar ]."""
        return compute_charge(y, ybar, self.grid, self.omega, self.eta0, index_tau=0)

    # ----------------------------------
    # Residual & Jacobian
    # ----------------------------------

    def residual(self, vec: np.ndarray) -> np.ndarray:
        y, ybar = self._unpack(vec)
        Nr, Nt = self.Nr, self.Ntau
        omega = self.omega
        dt = self.dt

        phi, phibar = self._compute_phi(y, ybar)
        s = (phi * phibar).real
        s = np.maximum(s, 0.0)
        V0 = self.Vprime(s)

        Fy = np.zeros_like(y, dtype=complex)
        Fyb = np.zeros_like(ybar, dtype=complex)

        for j in range(Nr):
            for i in range(Nt):
                # tau neighbors with ghost points
                if i == 0:
                    y_im1 = ybar[j, 0]
                    y_ip1 = y[j, 1]
                    yb_im1 = y[j, 0]
                    yb_ip1 = ybar[j, 1]
                elif i == Nt - 1:
                    y_im1 = y[j, Nt - 2]
                    y_ip1 = np.exp(-self.eta0) * ybar[j, Nt - 1]
                    yb_im1 = ybar[j, Nt - 2]
                    yb_ip1 = np.exp(+self.eta0) * y[j, Nt - 1]
                else:
                    y_im1 = y[j, i - 1]
                    y_ip1 = y[j, i + 1]
                    yb_im1 = ybar[j, i - 1]
                    yb_ip1 = ybar[j, i + 1]

                y_t = (y_im1 - y_ip1) / (2.0 * dt)
                y_tt = (y_im1 + y_ip1 - 2.0 * y[j, i]) / self.dt2

                yb_t = (yb_im1 - yb_ip1) / (2.0 * dt)
                yb_tt = (yb_im1 + yb_ip1 - 2.0 * ybar[j, i]) / self.dt2

                if j == 0:
                    y_jm1 = 0.0
                    yb_jm1 = 0.0
                else:
                    y_jm1 = y[j - 1, i]
                    yb_jm1 = ybar[j - 1, i]

                if j == Nr - 1:
                    y_jp1 = y[j, i]
                    yb_jp1 = ybar[j, i]
                else:
                    y_jp1 = y[j + 1, i]
                    yb_jp1 = ybar[j + 1, i]

                y_rr = (y_jp1 - 2.0 * y[j, i] + y_jm1) / self.dr2
                yb_rr = (yb_jp1 - 2.0 * ybar[j, i] + yb_jm1) / self.dr2

                Fy[j, i] = (
                    y_tt + y_rr + 2.0 * omega * y_t + omega * omega * y[j, i] - V0[j, i] * y[j, i]
                )
                Fyb[j, i] = (
                    yb_tt
                    + yb_rr
                    - 2.0 * omega * yb_t
                    + omega * omega * ybar[j, i]
                    - V0[j, i] * ybar[j, i]
                )

        return self._pack(Fy, Fyb)

    def jacobian(self, vec: np.ndarray) -> sp.csc_matrix:
        y, ybar = self._unpack(vec)
        Nr, Nt = self.Nr, self.Ntau
        dt = self.dt
        dr2 = self.dr2
        omega = self.omega
        eta0 = self.eta0

        phi, phibar = self._compute_phi(y, ybar)
        s = (phi * phibar).real
        s = np.maximum(s, 0.0)
        V0 = self.Vprime(s)
        dV0_ds = self.Vsecond(s)

        Nsite = Nr * Nt
        J = sp.lil_matrix((2 * Nsite, 2 * Nsite), dtype=complex)

        r = self.grid.r
        r2 = r * r

        for j in range(Nr):
            for i in range(Nt):
                idx = j * Nt + i
                row_y = idx
                row_yb = Nsite + idx

                # --- tau contributions for y ---
                if i == 0:
                    diag_y_time = -2.0 / (dt * dt)
                    col_ip1 = j * Nt + (i + 1)
                    J[row_y, col_ip1] += (1.0 / (dt * dt) - omega / dt)
                    col_ybar = Nsite + idx
                    J[row_y, col_ybar] += (1.0 / (dt * dt) + omega / dt)
                elif i == Nt - 1:
                    diag_y_time = -2.0 / (dt * dt)
                    col_im1 = j * Nt + (i - 1)
                    J[row_y, col_im1] += (1.0 / (dt * dt) + omega / dt)
                    col_ybar = Nsite + idx
                    J[row_y, col_ybar] += np.exp(-eta0) * (1.0 / (dt * dt) - omega / dt)
                else:
                    diag_y_time = -2.0 / (dt * dt)
                    col_im1 = j * Nt + (i - 1)
                    col_ip1 = j * Nt + (i + 1)
                    J[row_y, col_im1] += (1.0 / (dt * dt) + omega / dt)
                    J[row_y, col_ip1] += (1.0 / (dt * dt) - omega / dt)

                # --- tau contributions for ybar ---
                if i == 0:
                    diag_yb_time = -2.0 / (dt * dt)
                    col_ip1 = j * Nt + (i + 1)
                    J[row_yb, Nsite + col_ip1] += (1.0 / (dt * dt) + omega / dt)
                    J[row_yb, idx] += (1.0 / (dt * dt) - omega / dt)
                elif i == Nt - 1:
                    diag_yb_time = -2.0 / (dt * dt)
                    col_im1 = j * Nt + (i - 1)
                    J[row_yb, Nsite + col_im1] += (1.0 / (dt * dt) - omega / dt)
                    J[row_yb, idx] += np.exp(eta0) * (1.0 / (dt * dt) + omega / dt)
                else:
                    diag_yb_time = -2.0 / (dt * dt)
                    col_im1 = j * Nt + (i - 1)
                    col_ip1 = j * Nt + (i + 1)
                    J[row_yb, Nsite + col_im1] += (1.0 / (dt * dt) - omega / dt)
                    J[row_yb, Nsite + col_ip1] += (1.0 / (dt * dt) + omega / dt)

                # --- radial contributions ---
                if j == 0:
                    diag_y_rad = -2.0 / dr2
                    col_jp1 = (j + 1) * Nt + i
                    J[row_y, col_jp1] += 1.0 / dr2
                    diag_yb_rad = -2.0 / dr2
                    J[row_yb, Nsite + col_jp1] += 1.0 / dr2
                elif j == Nr - 1:
                    diag_y_rad = -1.0 / dr2
                    col_jm1 = (j - 1) * Nt + i
                    J[row_y, col_jm1] += 1.0 / dr2
                    diag_yb_rad = -1.0 / dr2
                    J[row_yb, Nsite + col_jm1] += 1.0 / dr2
                else:
                    diag_y_rad = -2.0 / dr2
                    col_jm1 = (j - 1) * Nt + i
                    col_jp1 = (j + 1) * Nt + i
                    J[row_y, col_jm1] += 1.0 / dr2
                    J[row_y, col_jp1] += 1.0 / dr2
                    diag_yb_rad = -2.0 / dr2
                    J[row_yb, Nsite + col_jm1] += 1.0 / dr2
                    J[row_yb, Nsite + col_jp1] += 1.0 / dr2

                y_val = y[j, i]
                ybar_val = ybar[j, i]
                r_sq = r2[j]

                ds_dy = ybar_val / r_sq
                ds_dybar = y_val / r_sq

                dV_dy = dV0_ds[j, i] * ds_dy
                dV_dybar = dV0_ds[j, i] * ds_dybar

                diag_y = (
                    diag_y_time
                    + diag_y_rad
                    + omega * omega
                    - V0[j, i]
                    - y_val * dV_dy
                )
                diag_yb = (
                    diag_yb_time
                    + diag_yb_rad
                    + omega * omega
                    - V0[j, i]
                    - ybar_val * dV_dybar
                )

                J[row_y, idx] += diag_y
                J[row_yb, Nsite + idx] += diag_yb

                J[row_y, Nsite + idx] += -y_val * dV_dybar
                J[row_yb, idx] += -ybar_val * dV_dy

        return J.tocsc()

    # ----------------------------------
    # Newton solve
    # ----------------------------------

    def solve(self, x0: np.ndarray, settings: QBall2DSettings, *, verbose: bool = False) -> NewtonResult:
        self.last_iteration_charge = None

        def iteration_callback(iter_idx: int, x_vec: np.ndarray, residual_vec: np.ndarray, res_norm: float) -> None:
            y_iter, ybar_iter = self._unpack(x_vec)
            charge_iter = self.compute_charge(y_iter, ybar_iter)
            self.last_iteration_charge = charge_iter
            if verbose or settings.newton_verbose:
                print(
                    f"[Newton-explicit] iter={iter_idx:02d}, ||F||={res_norm:.3e}, charge={charge_iter:.6e}"
                )

        return newton_solve(
            residual=lambda vec: self.residual(vec),
            jacobian=lambda vec: self.jacobian(vec),
            x0=x0,
            tol=settings.newton_tol,
            max_iter=settings.newton_max_iter,
            damping=settings.damping,
            norm=lambda v: np.linalg.norm(v.ravel()),
            verbose=verbose or settings.newton_verbose,
            callback=iteration_callback if (verbose or settings.newton_verbose) else None,
        )




def solve_qball_bounce_2d(
    params: LogisticPotentialParams,
    omega: float,
    *,
    profile: Optional[QBallProfile] = None,
    mode: Optional[UnstableMode] = None,
    settings: Optional[QBall2DSettings] = None,
) -> QBall2DSolution:
    settings = settings or QBall2DSettings()

    if profile is None:
        profile = solve_qball_profile(params, omega)
    if mode is None:
        mode = compute_unstable_mode(profile)

    grid = build_grid(settings.Nr, settings.Ntau, settings.Lr, settings.beta)

    ansatz = build_negative_mode_ansatz(
        profile,
        mode,
        grid,
        omega_reference=omega,
        amplitude=settings.ansatz_amplitude,
        tau_center=settings.tau_center,
        cosh_scale=settings.cosh_scale,
        envelope_width=settings.envelope_width,
        flip_sign=settings.flip_mode_sign,
        omega_tilde=settings.omega_tilde,
        center_at_cloud=settings.ansatz_center_at_cloud,
        decrease_towards_zero=settings.ansatz_decrease_towards_zero,
    )

    x0 = pack_fields(ansatz.y, ansatz.ybar)
    solver = QBall2DSolver(params, omega, settings, grid)
    newton_res = solver.solve(x0, settings, verbose=settings.newton_verbose)

    y_final, ybar_final = unpack_fields(newton_res.x, grid.Nr, grid.Ntau)
    phi_final = phi_from_y(y_final, grid, omega)
    phibar_final = phi_from_ybar(ybar_final, grid, omega)
    charge_final_exact = solver.compute_charge(y_final, ybar_final)
    charge_final = solver.last_iteration_charge
    if charge_final is None:
        charge_final = charge_final_exact

    # Compute energy using Eq.(2.4) at slice i=0
    energy = compute_energy(y_final, ybar_final, grid, omega, solver.eta0, params, index_tau=0)
    
    solution = QBall2DSolution(
        settings=settings,
        grid=grid,
        profile=profile,
        unstable_mode=mode,
        ansatz=ansatz,
        newton=newton_res,
        y=y_final,
        ybar=ybar_final,
        phi=phi_final,
        phibar=phibar_final,
        charge=charge_final_exact,
        charge_iter=charge_final,
        energy=energy,
        energy_ratio=energy / settings.energy_reference if settings.energy_reference is not None else None,
    )
    
    return solution


@dataclass
class EtaScanHistoryEntry:
    eta: float
    charge: float


@dataclass
class EtaScanResult:
    eta: float
    charge: float
    solution: QBall2DSolution
    history: List[EtaScanHistoryEntry]
    bracket: Tuple[float, float]


def _build_initial_guess_vector(
    profile: QBallProfile,
    mode: UnstableMode,
    grid: RadialTimeGrid,
    settings: QBall2DSettings,
    omega_reference: float,
) -> Tuple[np.ndarray, AnsatzResult]:
    ansatz = build_negative_mode_ansatz(
        profile,
        mode,
        grid,
        omega_reference=omega_reference,
        amplitude=settings.ansatz_amplitude,
        tau_center=settings.tau_center,
        cosh_scale=settings.cosh_scale,
        envelope_width=settings.envelope_width,
        flip_sign=settings.flip_mode_sign,
        omega_tilde=settings.omega_tilde,
        center_at_cloud=settings.ansatz_center_at_cloud,
        decrease_towards_zero=settings.ansatz_decrease_towards_zero,
    )
    x0 = pack_fields(ansatz.y, ansatz.ybar)
    return x0, ansatz


def solve_bounce_for_eta(
    params: LogisticPotentialParams,
    omega: float,
    profile: QBallProfile,
    mode: UnstableMode,
    grid: RadialTimeGrid,
    base_settings: QBall2DSettings,
    eta: float,
    *,
    x0_prev: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[float, QBall2DSolution, np.ndarray]:
    settings = replace(base_settings, eta0=eta)
    solver = QBall2DSolver(params, omega, settings, grid)

    if x0_prev is None:
        x0, ansatz = _build_initial_guess_vector(profile, mode, grid, settings, omega_reference=omega)
    else:
        x0 = x0_prev
        ansatz = None

    newton_res = solver.solve(x0, settings, verbose=verbose or settings.newton_verbose)
    y_final, ybar_final = unpack_fields(newton_res.x, grid.Nr, grid.Ntau)
    phi_final = phi_from_y(y_final, grid, omega)
    phibar_final = phi_from_ybar(ybar_final, grid, omega)
    charge = solver.compute_charge(y_final, ybar_final)
    charge_iter = solver.last_iteration_charge
    if charge_iter is None:
        charge_iter = charge

    # Compute energy using Eq.(2.4) at slice i=0
    energy = compute_energy(y_final, ybar_final, grid, omega, solver.eta0, params, index_tau=0)
    
    solution = QBall2DSolution(
        settings=settings,
        grid=grid,
        profile=profile,
        unstable_mode=mode,
        ansatz=ansatz,
        newton=newton_res,
        y=y_final,
        ybar=ybar_final,
        phi=phi_final,
        phibar=phibar_final,
        charge=charge,
        charge_iter=charge_iter,
        energy=energy,
        energy_ratio=energy / settings.energy_reference if settings.energy_reference is not None else None,
    )
    
    return charge, solution, newton_res.x


def scan_eta_to_match_charge(
    params: LogisticPotentialParams,
    omega: float,
    profile: QBallProfile,
    mode: UnstableMode,
    *,
    settings: QBall2DSettings,
    eta_start: float,
    target_charge: float,
    d_eta: float = 0.02,
    max_scan_steps: int = 15,
    tol: float = 1e-4,
    verbose: bool = False,
    x0_initial: Optional[np.ndarray] = None,
) -> EtaScanResult:
    grid = build_grid(settings.Nr, settings.Ntau, settings.Lr, settings.beta)
    history: List[EtaScanHistoryEntry] = []

    quiet_settings = replace(settings, newton_verbose=False)

    x_last = x0_initial
    charge_start, solution_start, x_last = solve_bounce_for_eta(
        params,
        omega,
        profile,
        mode,
        grid,
        settings,
        eta_start,
        x0_prev=x_last,
        verbose=verbose,
    )
    history.append(EtaScanHistoryEntry(eta_start, charge_start))

    if verbose:
        print(
            f"[eta-scan] eta = {eta_start:.6f}, "
            f"charge = {charge_start:.6e}, target = {target_charge:.6e}, "
            f"ratio = {charge_start/target_charge:.6f}"
        )

    direction = 1.0 if charge_start < target_charge else -1.0
    eta_low = eta_start
    charge_low = charge_start
    eta_high: Optional[float] = None
    charge_high: Optional[float] = None

    for step in range(1, max_scan_steps + 1):
        eta_try = eta_start + direction * d_eta * step
        charge_try, solution_try, x_last = solve_bounce_for_eta(
            params,
            omega,
            profile,
            mode,
            grid,
            quiet_settings,
            eta_try,
            x0_prev=x_last,
            verbose=False,
        )
        history.append(EtaScanHistoryEntry(eta_try, charge_try))

        if verbose:
            print(
                f"[eta-scan] try eta = {eta_try:.6f}, "
                f"charge = {charge_try:.6e}, ratio = {charge_try/target_charge:.6f}"
            )

        f_low = charge_low - target_charge
        f_try = charge_try - target_charge
        if f_low * f_try < 0.0:
            eta_high = eta_try
            charge_high = charge_try
            bracket = (eta_low, eta_high)
            break
        eta_low, charge_low = eta_try, charge_try

    if eta_high is None or charge_high is None:
        raise RuntimeError("Failed to bracket target charge during eta scan.")

    # Store already computed values to avoid recomputation
    cached_values = {eta_low: charge_low, eta_high: charge_high}
    
    def f_eta(val: float) -> float:
        nonlocal x_last
        # Check if this value was already computed during the scan
        if val in cached_values:
            charge_val = cached_values[val]
            if verbose:
                print(
                    f"[eta-root] eta = {val:.6f} (cached), "
                    f"charge = {charge_val:.6e}, ratio = {charge_val/target_charge:.6f}"
                )
            return charge_val - target_charge
        
        # Otherwise, compute it
        charge_val, _, x_last = solve_bounce_for_eta(
            params,
            omega,
            profile,
            mode,
            grid,
            quiet_settings,
            val,
            x0_prev=x_last,
            verbose=False,
        )
        history.append(EtaScanHistoryEntry(val, charge_val))
        cached_values[val] = charge_val
        if verbose:
            print(
                f"[eta-root] eta = {val:.6f}, "
                f"charge = {charge_val:.6e}, ratio = {charge_val/target_charge:.6f}"
            )
        return charge_val - target_charge

    eta_star = brentq(f_eta, eta_low, eta_high, xtol=tol, rtol=tol)
    charge_star, solution_star, x_last = solve_bounce_for_eta(
        params,
        omega,
        profile,
        mode,
        grid,
        quiet_settings,
        eta_star,
        x0_prev=x_last,
        verbose=False,
    )
    # DO NOT overwrite solution_star.settings as it already has the correct eta0 from solve_bounce_for_eta
    history.append(EtaScanHistoryEntry(eta_star, charge_star))

    if verbose:
        print(
            f"[eta-scan] eta* = {eta_star:.6f}, charge = {charge_star:.6e}, "
            f"ratio = {charge_star/target_charge:.6f}"
        )
        if solution_star.energy is not None:
            if solution_star.energy_ratio is not None:
                print(
                    f"[eta-scan] energy(τ≈0) = {solution_star.energy:.6e}, "
                    f"energy/E_Q = {solution_star.energy_ratio:.6f}"
                )
            else:
                print(f"[eta-scan] energy(τ≈0) = {solution_star.energy:.6e}")

    return EtaScanResult(
        eta=eta_star,
        charge=charge_star,
        solution=solution_star,
        history=history,
        bracket=(eta_low, eta_high),
    )


__all__ = [
    "EtaScanHistoryEntry",
    "EtaScanResult",
    "QBall2DSettings",
    "QBall2DSolution",
    "QBall2DSolver",
    "scan_eta_to_match_charge",
    "solve_bounce_for_eta",
    "solve_qball_bounce_2d",
]

