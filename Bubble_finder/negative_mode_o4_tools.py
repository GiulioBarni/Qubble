from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# Flexible imports: package usage first, local fallback second.
try:
    from Bubble_finder.ansatz_bubble import PotentialModel, compute_Dp_shooting_1d
except Exception:
    from ansatz_bubble import PotentialModel, compute_Dp_shooting_1d


@dataclass
class O4NegativeModeResult:
    gamma: float
    gamma_refined: float
    lam_neg: float
    beta_natural: float

    r: np.ndarray
    phi_slice: np.ndarray
    phi_false: float
    y1d: np.ndarray
    ybar1d: np.ndarray

    xi_y: np.ndarray
    xi_ybar: np.ndarray
    yR: np.ndarray
    yI: np.ndarray

    gamma_grid: np.ndarray
    Dp_vals: np.ndarray
    bracket: Optional[Tuple[float, float]]
    Dp_at_gamma: float

    V1_bg: np.ndarray
    V2_bg: np.ndarray
    rho_bg: np.ndarray

    grid_is_uniform: bool
    dr_min: float
    dr_max: float
    rel_spread: float
    n_points_used: int
    drop_points_near_origin: int
    message: str

    @property
    def xi(self) -> np.ndarray:
        return self.xi_y


def _check_r_grid(r: np.ndarray, tol_rel: float = 1e-6) -> Tuple[bool, float, float, float]:
    r = np.asarray(r, dtype=float)
    if len(r) < 3:
        return True, np.nan, np.nan, 0.0
    drs = np.diff(r)
    dr_min = float(np.min(drs))
    dr_max = float(np.max(drs))
    dr_med = float(np.median(drs))
    rel_spread = 0.0 if dr_med == 0.0 else float(np.max(np.abs(drs - dr_med)) / abs(dr_med))
    return rel_spread < tol_rel, dr_min, dr_max, rel_spread


def _build_region(
    r_bounce: np.ndarray,
    phi_bounce: np.ndarray,
    phi_false_bounce: float,
    *,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    drop_points_near_origin: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r_full = np.asarray(r_bounce, dtype=float)
    phi_full = np.asarray(phi_bounce, dtype=float)

    mask = (r_full >= r_min)
    if r_max is not None:
        mask &= (r_full <= r_max)

    r = r_full[mask].copy()
    phi_slice = phi_full[mask].copy()

    if drop_points_near_origin > 0:
        if len(r) <= drop_points_near_origin + 3:
            raise RuntimeError("Too few points left after drop_points_near_origin.")
        r = r[drop_points_near_origin:].copy()
        phi_slice = phi_slice[drop_points_near_origin:].copy()

    if len(r) < 4:
        raise RuntimeError("Need at least 4 points in the selected radial region.")

    return r, phi_slice, float(phi_false_bounce)


def _build_y_arrays(r: np.ndarray, phi_slice: np.ndarray, phi_false: float) -> Tuple[np.ndarray, np.ndarray]:
    y1d = np.zeros_like(r, dtype=float)
    ybar1d = np.zeros_like(r, dtype=float)
    if abs(r[0]) < 1e-14:
        y1d[0] = 0.0
        ybar1d[0] = 0.0
        y1d[1:] = r[1:] * (phi_slice[1:] - phi_false)
        ybar1d[1:] = r[1:] * (phi_slice[1:] - phi_false)
    else:
        y1d[:] = r * (phi_slice - phi_false)
        ybar1d[:] = r * (phi_slice - phi_false)
    return y1d, ybar1d


def _reconstruct_phi_from_y(r: np.ndarray, y1d: np.ndarray, ybar1d: np.ndarray, phi_false: float) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.empty_like(y1d, dtype=float)
    phibar = np.empty_like(ybar1d, dtype=float)
    if abs(r[0]) < 1e-14:
        phi[0] = phi_false
        phibar[0] = phi_false
        phi[1:] = phi_false + y1d[1:] / r[1:]
        phibar[1:] = phi_false + ybar1d[1:] / r[1:]
    else:
        phi[:] = phi_false + y1d / r
        phibar[:] = phi_false + ybar1d / r
    return phi, phibar


def _find_best_bracket(gamma_grid: np.ndarray, Dp_vals: np.ndarray) -> Optional[Tuple[float, float]]:
    gamma_grid = np.asarray(gamma_grid, dtype=float)
    Dp_vals = np.asarray(Dp_vals, dtype=float)
    candidates = []
    for i in range(len(gamma_grid) - 1):
        DL = Dp_vals[i]
        DR = Dp_vals[i + 1]
        if not (np.isfinite(DL) and np.isfinite(DR)):
            continue
        if DL == 0.0:
            candidates.append((0.0, float(gamma_grid[i]), float(gamma_grid[i])))
        elif np.sign(DL) != np.sign(DR):
            score = abs(DL) + abs(DR)
            candidates.append((float(score), float(gamma_grid[i]), float(gamma_grid[i + 1])))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return (candidates[0][1], candidates[0][2])


def _build_shooting_problem(
    *,
    r: np.ndarray,
    phi: np.ndarray,
    phibar: np.ndarray,
    potential: Any,
    omega: float,
    interp_kind: str = "cubic",
    ode_method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: Optional[float] = None,
    sign_convention: int = +1,
) -> Dict[str, Any]:
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    phibar = np.asarray(phibar, dtype=float)

    s_bg = np.maximum((phi * phibar).real, 0.0)
    rho_bg = np.sqrt(2.0 * s_bg)
    V_s = potential.dV_ds(s_bg)
    V_ss = potential.d2V_ds2(s_bg)

    # Keep the same mapping as the current ansatz_bubble implementation,
    # so diagnostics are consistent with your present code path.
    V1_bg = 2.0 * V_s
    V2_bg = 4.0 * rho_bg * V_ss

    V1 = interp1d(r, V1_bg, kind=interp_kind, fill_value="extrapolate")
    V2 = interp1d(r, V2_bg, kind=interp_kind, fill_value="extrapolate")
    rho = interp1d(r, rho_bg, kind=interp_kind, fill_value="extrapolate")

    sgn = +1.0 if sign_convention >= 0 else -1.0
    r_min = float(r[0])
    r_max = float(r[-1])
    t_eval = r[::-1]

    def ode_rhs(rv: float, Y: np.ndarray, gamma: float) -> np.ndarray:
        yR, dyR, yI, dyI = Y
        V1v = float(V1(rv))
        V2v = float(V2(rv))
        rhv = float(rho(rv))

        coeff = (gamma * gamma - omega * omega) + V1v
        extra = 2.0 * V2v * rhv

        yR2 = (coeff + extra) * yR - 2.0 * sgn * omega * gamma * yI
        yI2 = coeff * yI + 2.0 * sgn * omega * gamma * yR
        return np.array([dyR, yR2, dyI, yI2], dtype=float)

    def solve_basis(gamma: float):
        y0_A = np.array([1.0, 1.0 / r_max, 0.0, 0.0], dtype=float)
        y0_B = np.array([0.0, 0.0, 1.0, 1.0 / r_max], dtype=float)
        kwargs: Dict[str, Any] = dict(method=ode_method, t_eval=t_eval, rtol=rtol, atol=atol)
        if max_step is not None:
            kwargs["max_step"] = max_step
        sol_A = solve_ivp(lambda rr, yy: ode_rhs(rr, yy, gamma), (r_max, r_min), y0_A, **kwargs)
        sol_B = solve_ivp(lambda rr, yy: ode_rhs(rr, yy, gamma), (r_max, r_min), y0_B, **kwargs)
        if not (sol_A.success and sol_B.success):
            raise RuntimeError(f"ODE solver failed for gamma={gamma}")
        yR_A = sol_A.y[0, ::-1]
        yI_A = sol_A.y[2, ::-1]
        yR_B = sol_B.y[0, ::-1]
        yI_B = sol_B.y[2, ::-1]
        return yR_A, yI_A, yR_B, yI_B

    def Dp_of_gamma(gamma: float) -> float:
        yR_A, yI_A, yR_B, yI_B = solve_basis(gamma)
        return float(yR_A[0] * yI_B[0] - yR_B[0] * yI_A[0])

    def mode_at_gamma(gamma: float) -> Dict[str, np.ndarray]:
        yR_A, yI_A, yR_B, yI_B = solve_basis(gamma)
        c_A = float(yR_B[0])
        c_B = -float(yR_A[0])
        yR = c_A * yR_A + c_B * yR_B
        yI = c_A * yI_A + c_B * yI_B
        norm = np.sqrt(np.trapz(yR * yR + yI * yI, r))
        if norm > 0.0:
            yR = yR / norm
            yI = yI / norm
        xi_y = yR + yI
        xi_ybar = yR - yI
        if abs(r[0]) < 1e-14:
            xi_y[0] = 0.0
            xi_ybar[0] = 0.0
        return {"yR": yR, "yI": yI, "xi_y": xi_y, "xi_ybar": xi_ybar}

    return {
        "Dp_of_gamma": Dp_of_gamma,
        "mode_at_gamma": mode_at_gamma,
        "V1_bg": V1_bg,
        "V2_bg": V2_bg,
        "rho_bg": rho_bg,
    }


def compute_o4_negative_mode_from_bounce(
    r_bounce: np.ndarray,
    phi_bounce: np.ndarray,
    phi_false_bounce: float,
    U: Callable,
    dU: Callable,
    d2U: Callable,
    omega: float,
    *,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    drop_points_near_origin: int = 0,
    gamma_scan: Tuple[float, float] = (0.3, 1.0),
    n_scan: int = 240,
    sign_convention: int = +1,
    interp_kind: str = "cubic",
    ode_method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: Optional[float] = None,
    grid_uniform_tol_rel: float = 1e-6,
    use_existing_scan: bool = True,
) -> O4NegativeModeResult:
    """
    Robust wrapper for your current 1D negative-mode shooting logic.

    Important:
    this keeps the same Q-ball-like 2x2 shooting structure used in ansatz_bubble,
    so the output is directly comparable with your current notebook diagnostics.
    It improves only the robustness of the scan, bracket selection, root refinement,
    and plotting payload.
    """
    r, phi_slice, phi_false = _build_region(
        r_bounce,
        phi_bounce,
        phi_false_bounce,
        r_min=r_min,
        r_max=r_max,
        drop_points_near_origin=drop_points_near_origin,
    )
    y1d, ybar1d = _build_y_arrays(r, phi_slice, phi_false)
    phi, phibar = _reconstruct_phi_from_y(r, y1d, ybar1d, phi_false)

    grid_is_uniform, dr_min, dr_max, rel_spread = _check_r_grid(r, tol_rel=grid_uniform_tol_rel)
    potential = PotentialModel(U, dU, d2U)

    problem = _build_shooting_problem(
        r=r,
        phi=phi,
        phibar=phibar,
        potential=potential,
        omega=float(omega),
        interp_kind=interp_kind,
        ode_method=ode_method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        sign_convention=sign_convention,
    )
    Dp_of_gamma = problem["Dp_of_gamma"]
    mode_at_gamma = problem["mode_at_gamma"]

    # Optional consistency with the exact scan routine currently in ansatz_bubble.
    if use_existing_scan:
        out_scan = compute_Dp_shooting_1d(
            r=r,
            phi=phi,
            phibar=phibar,
            potential=potential,
            omega=float(omega),
            gamma_scan=gamma_scan,
            n_scan=n_scan,
            interp_kind=interp_kind,
            ode_method=ode_method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            sign_convention=sign_convention,
            require_bracket=False,
        )
        gamma_grid = np.asarray(out_scan["gamma_grid"], dtype=float)
        Dp_vals = np.asarray(out_scan["Dp_vals"], dtype=float)
        message = "scan from ansatz_bubble.compute_Dp_shooting_1d"
    else:
        gamma_grid = np.linspace(gamma_scan[0], gamma_scan[1], int(n_scan), dtype=float)
        Dp_vals = np.array([Dp_of_gamma(g) for g in gamma_grid], dtype=float)
        message = "scan from local Dp_of_gamma"

    bracket = _find_best_bracket(gamma_grid, Dp_vals)
    gamma_refined = np.nan
    Dp_at_gamma = np.nan
    yR = np.zeros_like(r)
    yI = np.zeros_like(r)
    xi_y = np.zeros_like(r)
    xi_ybar = np.zeros_like(r)

    if bracket is not None and bracket[0] == bracket[1]:
        gamma_refined = float(bracket[0])
    elif bracket is not None:
        root = root_scalar(Dp_of_gamma, bracket=bracket, method="brentq", xtol=1e-10, rtol=1e-10, maxiter=200)
        if not root.converged:
            raise RuntimeError("Root finding for gamma did not converge.")
        gamma_refined = float(root.root)

    if np.isfinite(gamma_refined):
        Dp_at_gamma = float(Dp_of_gamma(gamma_refined))
        mode = mode_at_gamma(gamma_refined)
        yR = np.asarray(mode["yR"], dtype=float)
        yI = np.asarray(mode["yI"], dtype=float)
        xi_y = np.asarray(mode["xi_y"], dtype=float)
        xi_ybar = np.asarray(mode["xi_ybar"], dtype=float)
        lam_neg = -gamma_refined * gamma_refined
        beta_natural = 2.0 * np.pi / gamma_refined if gamma_refined > 0 else np.nan
        gamma_final = gamma_refined
        message += "; bracketed and refined with brentq"
    else:
        lam_neg = np.nan
        beta_natural = np.nan
        gamma_final = np.nan
        message += "; no sign-change bracket found"

    if not grid_is_uniform:
        message += f"; nonuniform r-grid (rel_spread={rel_spread:.3e})"

    return O4NegativeModeResult(
        gamma=float(gamma_final),
        gamma_refined=float(gamma_refined),
        lam_neg=float(lam_neg),
        beta_natural=float(beta_natural),
        r=np.asarray(r, dtype=float),
        phi_slice=np.asarray(phi_slice, dtype=float),
        phi_false=float(phi_false),
        y1d=np.asarray(y1d, dtype=float),
        ybar1d=np.asarray(ybar1d, dtype=float),
        xi_y=np.asarray(xi_y, dtype=float),
        xi_ybar=np.asarray(xi_ybar, dtype=float),
        yR=np.asarray(yR, dtype=float),
        yI=np.asarray(yI, dtype=float),
        gamma_grid=np.asarray(gamma_grid, dtype=float),
        Dp_vals=np.asarray(Dp_vals, dtype=float),
        bracket=bracket,
        Dp_at_gamma=float(Dp_at_gamma),
        V1_bg=np.asarray(problem["V1_bg"], dtype=float),
        V2_bg=np.asarray(problem["V2_bg"], dtype=float),
        rho_bg=np.asarray(problem["rho_bg"], dtype=float),
        grid_is_uniform=bool(grid_is_uniform),
        dr_min=float(dr_min),
        dr_max=float(dr_max),
        rel_spread=float(rel_spread),
        n_points_used=int(len(r)),
        drop_points_near_origin=int(drop_points_near_origin),
        message=str(message),
    )


def print_o4_negative_mode_report(result: O4NegativeModeResult) -> None:
    print("\nO(4) negative-mode report")
    print("-------------------------")
    print(f"gamma                 = {result.gamma}")
    print(f"gamma_refined         = {result.gamma_refined}")
    print(f"lambda_neg            = {result.lam_neg}")
    print(f"beta_natural          = {result.beta_natural}")
    print(f"Dp(gamma_refined)     = {result.Dp_at_gamma}")
    print(f"best bracket          = {result.bracket}")
    print(f"grid_is_uniform       = {result.grid_is_uniform}")
    print(f"dr_min, dr_max        = {result.dr_min}, {result.dr_max}")
    print(f"rel_spread            = {result.rel_spread}")
    print(f"n_points_used         = {result.n_points_used}")
    print(f"drop_points_near_org  = {result.drop_points_near_origin}")
    print(f"message               = {result.message}")


def plot_o4_negative_mode_summary(
    result: O4NegativeModeResult,
    *,
    gamma_xlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (16, 5),
    fontsize: int = 15,
    show_components: bool = False,
) -> Tuple[Any, Any]:
    """
    Three panels:
      1) xi_y and xi_ybar
      2) signed D_p(gamma)
      3) log10|D_p(gamma)|
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax0, ax1, ax2 = axes

    r = result.r
    ax0.plot(r, result.xi_y, lw=2.0, label=r"$\xi_y(r)$")
    ax0.plot(r, result.xi_ybar, "--", lw=1.6, label=r"$\xi_{\bar y}(r)$")
    if show_components:
        ax0.plot(r, result.yR, ":", lw=1.4, label=r"$y_R(r)$")
        ax0.plot(r, result.yI, "-.", lw=1.4, label=r"$y_I(r)$")
    ax0.axhline(0.0, color="k", lw=0.9, alpha=0.7)
    ax0.set_xlabel(r"$r$", fontsize=fontsize)
    ax0.set_ylabel(r"mode profile", fontsize=fontsize)
    ax0.set_title(rf"Negative mode: $\lambda$ = {result.lam_neg:.4e}, $\gamma$ = {result.gamma:.4e}", fontsize=fontsize)
    ax0.grid(True, alpha=0.4)
    ax0.legend(fontsize=max(fontsize - 2, 10))
    ax0.tick_params(axis="both", labelsize=fontsize)

    gg = result.gamma_grid
    Dp = result.Dp_vals

    ax1.plot(gg, Dp, "-o", lw=1.5, ms=4, label=r"$D_p(\gamma)$")
    ax1.axhline(0.0, color="k", lw=0.9)
    if result.bracket is not None:
        ax1.axvspan(result.bracket[0], result.bracket[1], color="orange", alpha=0.2, label="best bracket")
    if np.isfinite(result.gamma_refined):
        ax1.axvline(result.gamma_refined, color="r", ls="--", lw=1.5, alpha=0.8, label=rf"$\gamma_*$ = {result.gamma_refined:.6f}")
    ax1.set_xlabel(r"$\gamma$", fontsize=fontsize)
    ax1.set_ylabel(r"$D_p(\gamma)$", fontsize=fontsize)
    ax1.set_title("Signed shooting function", fontsize=fontsize)
    ax1.grid(True, alpha=0.4)
    if gamma_xlim is not None:
        ax1.set_xlim(*gamma_xlim)
    ax1.legend(fontsize=max(fontsize - 2, 10))
    ax1.tick_params(axis="both", labelsize=fontsize)

    ax2.plot(gg, np.log10(np.abs(Dp) + 1e-300), "-o", lw=1.5, ms=4)
    ax2.axhline(0.0, color="k", lw=0.9)
    if result.bracket is not None:
        ax2.axvspan(result.bracket[0], result.bracket[1], color="orange", alpha=0.2, label="best bracket")
    if np.isfinite(result.gamma_refined):
        ax2.axvline(result.gamma_refined, color="r", ls="--", lw=1.5, alpha=0.8, label=rf"$\gamma_*$ = {result.gamma_refined:.6f}")
    ax2.set_xlabel(r"$\gamma$", fontsize=fontsize)
    ax2.set_ylabel(r"$\log_{10}|D_p(\gamma)|$", fontsize=fontsize)
    ax2.set_title("Log shooting diagnostic", fontsize=fontsize)
    ax2.grid(True, alpha=0.4)
    if gamma_xlim is not None:
        ax2.set_xlim(*gamma_xlim)
    ax2.legend(fontsize=max(fontsize - 2, 10))
    ax2.tick_params(axis="both", labelsize=fontsize)

    fig.tight_layout()
    return fig, axes
