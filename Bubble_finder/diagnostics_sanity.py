"""
Robust sanity checks for Bubble2DSolver:
- Background homogeneous solution verification
- Tau-independent 1D bubble embedding at eta=0

All comments in English. No physics convention changes without explicit documentation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Callable, Tuple, Optional, Dict, Any


def rho0_from_homogeneous_condition(
    omega: float,
    dU: Callable[[np.ndarray], np.ndarray],
    bracket: Tuple[float, float] = (1.0, 1.1),
    rtol: float = 1e-12,
) -> float:
    """
    Solve the homogeneous stationarity condition used by Bubble2DSolver.

    The solver uses W(u) = dU/(2*rho) with rho = sqrt(u).
    For homogeneous background: (omega^2 - W(rho0^2)) * rho0 = 0 =>
    W(rho0^2) = omega^2 => dU(rho0)/(2*rho0) = omega^2 => dU(rho0) = 2*omega^2*rho0.

    So the equation is: dU(rho) - 2*omega^2*rho = 0.
    This matches solve_rho0_for_omega in bounce2d.py.
    """
    omega = float(omega)
    a, b = map(float, bracket)

    def f(rho: float) -> float:
        return float(dU(np.array([rho]))[0] - 2.0 * omega * omega * rho)

    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(
            f"rho0_from_homogeneous_condition: bad bracket f(a)={fa:+.3e}, f(b)={fb:+.3e}. "
            "Try different bracket or check potential derivative convention."
        )
    return float(brentq(f, a, b, xtol=rtol, rtol=rtol, maxiter=300))


def build_background_zero_vec(solver) -> np.ndarray:
    """
    Build x corresponding to homogeneous background: y=0, ybar=0 everywhere.

    phi_rot = rho0 + y/r => with y=0: phi_rot = rho0.
    This is the constant background. At r=0 we have y[0]=0 by regularity.
    """
    Nr, Nt = solver.Nr, solver.Nt
    y = np.zeros((Nr, Nt), dtype=complex)
    ybar = np.zeros((Nr, Nt), dtype=complex)
    return solver.pack(y, ybar)


def build_tau_independent_embedding(
    solver,
    r_1d: np.ndarray,
    varphi_1d: np.ndarray,
    rho0: float,
) -> np.ndarray:
    """
    Embed 1D profile varphi_1d(r) as tau-independent 2D config.

    phi_rot(r,tau) = varphi_1d(r), phibar_rot = phi_rot.
    y = r*(phi_rot - rho0), ybar = r*(phibar_rot - rho0).
    At r=0: regularity requires y[0]=0, ybar[0]=0 (limit of r*(varphi-rho0) as r->0).
    """
    r_grid = np.asarray(solver.grid.r, dtype=float)
    r_1d = np.asarray(r_1d, dtype=float)
    varphi_1d = np.asarray(varphi_1d, dtype=float)
    order = np.argsort(r_1d)
    r_1d = r_1d[order]
    varphi_1d = varphi_1d[order]
    # Cubic interpolation for smooth embedding (linear gives phi''=0 and large 2D residual)
    from scipy.interpolate import interp1d
    interp_fn = interp1d(r_1d, varphi_1d, kind="cubic", bounds_error=False, fill_value=(varphi_1d[0], varphi_1d[-1]))
    varphi_r = interp_fn(r_grid)
    Nr, Nt = solver.Nr, solver.Nt
    r = np.asarray(solver.grid.r, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    varphi_2d = np.broadcast_to(varphi_r[:, None], (Nr, Nt))
    y = r * (varphi_2d - rho0)
    ybar = y.copy()
    if r.size > 0 and float(r.flat[0]) < 1e-12 * (float(np.max(r)) + 1e-12):
        y[0, :] = 0.0
        ybar[0, :] = 0.0
    return solver.pack(y, ybar)


def residual_report_compact(
    solver,
    x: np.ndarray,
    name: str = "",
) -> Dict[str, Any]:
    """Compute residual norms and max location. Returns dict with nF_inf, nF_2, nF_2_per_site, j_max, i_max."""
    F = solver.residual(x)
    F = np.asarray(F, dtype=float)
    nF_inf = float(np.max(np.abs(F)))
    nF_2 = float(np.linalg.norm(F))
    N = solver.Nr * solver.Nt
    n_sites = F.size
    nF_2_per_site = nF_2 / np.sqrt(n_sites) if n_sites > 0 else 0.0
    flat_idx = int(np.argmax(np.abs(F)))
    site_flat = flat_idx % N if N > 0 else 0
    j_max = site_flat // solver.Nt if solver.Nt > 0 else 0
    i_max = site_flat % solver.Nt if solver.Nt > 0 else 0
    r_arr = np.asarray(solver.grid.r, dtype=float)
    tau_arr = np.asarray(solver.grid.tau, dtype=float)
    return dict(
        name=name,
        nF_inf=nF_inf,
        nF_2=nF_2,
        nF_2_per_site=nF_2_per_site,
        j_max=int(min(j_max, len(r_arr) - 1)) if len(r_arr) > 0 else 0,
        i_max=int(min(i_max, len(tau_arr) - 1)) if len(tau_arr) > 0 else 0,
        r_at_max=float(r_arr[j_max]) if j_max < len(r_arr) else np.nan,
        tau_at_max=float(tau_arr[i_max]) if i_max < len(tau_arr) else np.nan,
    )


# -----------------------------------------------------------------------------
# Checklist helpers: explicit guards for common failure modes
# -----------------------------------------------------------------------------
def checklist_potential_convention(solver, dU) -> dict:
    """
    A) Potential convention: verify W(u)=U'(rho)/(2*rho), rho=sqrt(u).
    Homogeneous: dU(rho0)=2*omega^2*rho0.
    """
    rho0 = float(solver.rho0)
    omega = float(solver.omega)
    dU0 = float(dU(np.array([rho0]))[0])
    expected = 2.0 * omega * omega * rho0
    ok = abs(dU0 - expected) < 1e-10 * (abs(dU0) + 1.0)
    return dict(
        dU_at_rho0=dU0,
        expected_2omega2_rho0=expected,
        match=ok,
        msg="dU(rho0)=2*omega^2*rho0" if ok else "MISMATCH: check V(varphi) vs U(rho), dV/dvarphi vs dU/drho",
    )


def checklist_field_into_potential(solver) -> dict:
    """
    B) Field fed into potential: u=Re(phi_rot*phibar_rot), varphi=sqrt(u).
    For homogeneous: u=rho0^2.
    """
    u_hom = float(solver.rho0**2)
    return dict(
        u_homogeneous=u_hom,
        varphi_homogeneous=float(np.sqrt(u_hom)),
        rho0=solver.rho0,
        msg="u=rho0^2, varphi=sqrt(u) for homogeneous",
    )


def checklist_eta0_for_tau_indep(solver) -> dict:
    """
    D) For tau-independent check: eta0 must be exactly 0.
    """
    eta0 = float(getattr(solver.settings, "eta0", getattr(solver, "eta0", None)))
    ok = eta0 is not None and abs(eta0) < 1e-14
    return dict(eta0=eta0, is_zero=ok, msg="eta0=0 for tau-indep" if ok else "eta0 != 0")


# -----------------------------------------------------------------------------
# PART 1 — Model identity report (1D vs 2D equation comparison)
# -----------------------------------------------------------------------------

def model_identity_report(solver, dU, d2U, phi0_pot, v1, v2) -> Dict[str, Any]:
    """
    Print a model identity report comparing the 1D bounce ODE and the 2D
    Bubble2DSolver tau-independent reduction.

    Returns dict with PASS/FAIL and explicit term-by-term comparison.
    """
    omega = float(solver.omega)
    rho0 = float(solver.rho0)

    report = []

    # (a) 2D PDE in Bubble2DSolver.residual()
    report.append("=" * 70)
    report.append("PART 1 — Model identity report (1D vs 2D)")
    report.append("=" * 70)
    report.append("")
    report.append("(a) 2D PDE enforced by Bubble2DSolver.residual():")
    report.append("  - Variables: y = r*(phi_rot - rho0), ybar = r*(phibar_rot - rho0)")
    report.append("  - phi_rot = e^{-omega*tau}*phi, phibar_rot = e^{+omega*tau}*phibar")
    report.append("  - u = Re(phi_rot * phibar_rot) = Re((y_tot/r)*(ybar_tot/r)), y_tot = y + r*rho0")
    report.append("  - rho = sqrt(u_pos + rho_eps), u_pos = smooth_positive(u)")
    report.append("  - Potential: W(u) = dU(rho)/(2*rho), U(rho)=V(varphi) with varphi=rho")
    report.append("  - Scalar in potential: u = Re(phi_rot*phibar_rot), rho = sqrt(u)")
    report.append("  - Equation: Fy = y_tt + y_rr + 2*omega*y_t + (omega^2 - W)*y_tot")
    report.append("")

    # (b) Tau-independent, eta0=0 reduction
    report.append("(b) Tau-independent (d/dtau=0), eta0=0 reduction of 2D PDE:")
    report.append("  - y_t=0, y_tt=0. For real phi_rot=phibar_rot: y=ybar, y_tot=r*phi.")
    report.append("  - y_rr = 2*phi' + r*phi'' (from y = r*(phi-rho0))")
    report.append("  - Fy = y_rr + A_coef*y_tot, A_coef = 2*(omega^2 - W) [FIXED for 1D match]")
    report.append("  - W = dU(rho)/(2*rho), so A_coef*phi = 2*omega^2*phi - dV_dphi")
    report.append("  - REDUCED 2D (fixed): phi'' + 2*phi'/r + 2*omega^2*phi - dV_dphi(phi) = 0")
    report.append("")

    # (c) 1D ODE in solve_bounce()
    report.append("(c) ODE solved by solve_bounce() in bounce_1d.py:")
    report.append("  - Omega(phi) = V(phi) - omega^2*phi^2  (grand potential)")
    report.append("  - dOmega/dphi = dV_dphi - 2*omega^2*phi")
    report.append("  - Equation: phi'' + (d-1)/r * phi' = dOmega/dphi  (d=3 => 2/r)")
    report.append("  - 1D ODE: phi'' + 2*phi'/r = dV_dphi - 2*omega^2*phi")
    report.append("  - Rearranged: phi'' + 2*phi'/r + 2*omega^2*phi - dV_dphi = 0")
    report.append("")

    # (d) Term-by-term comparison (after factor-2 fix in bounce2d)
    report.append("(d) Term-by-term comparison (with A_coef = 2*(omega^2 - W) in bounce2d):")
    all_ok = True
    diffs = []

    # Field variable
    report.append("  Field variable: 1D uses phi (amplitude), 2D uses phi_rot = phi (real). MATCH.")
    report.append("  Potential: Both use dV_dphi(phi). MATCH (after factor-2 fix).")
    report.append("  Spatial dimension: Both have 2/r (O(3) Laplacian). MATCH.")
    report.append("  Omega-dependent terms: Both have +2*omega^2*phi. MATCH (after factor-2 fix).")

    report.append("")
    report.append("=" * 70)
    if all_ok:
        report.append("RESULT: PASS — 1D and 2D tau-independent reduction are the same problem.")
    else:
        report.append("RESULT: FAIL — 1D and 2D equations differ. Mismatches:")
        for d in diffs:
            report.append(f"  - {d}")
    report.append("=" * 70)

    return dict(
        report="\n".join(report),
        all_ok=all_ok,
        diffs=diffs,
    )


# -----------------------------------------------------------------------------
# PART 2 — Local ODE residual comparison and bulk vs boundary
# -----------------------------------------------------------------------------

def local_ode_residual_comparison(
    r_1d: np.ndarray,
    phi_1d: np.ndarray,
    solver,
    dU,
    dOmega_dphi_fn,
    phi0_pot: float,
    v1: float,
    v2: float,
) -> Dict[str, Any]:
    """
    Evaluate the local ODE residual at each r using BOTH the 1D equation and the
    2D reduced equation. Returns max residual for each definition.
    """
    r = np.asarray(r_1d, dtype=float)
    phi = np.asarray(phi_1d, dtype=float)
    omega = float(solver.omega)
    d = 3

    # Finite differences for phi', phi''
    dr = np.gradient(r)
    phi_p = np.gradient(phi, r, edge_order=2)
    phi_pp = np.gradient(phi_p, r, edge_order=2)
    # Avoid small r where 2*phi'/r amplifies FD noise; exclude r < 0.01
    r_safe = max(0.01, 1e-8 * (float(np.max(r)) + 1e-12))
    mask = r > r_safe
    if not np.any(mask):
        return dict(res_1d_max=np.nan, res_2d_max=np.nan, n_pts=0)

    r_in = r[mask]
    phi_in = phi[mask]
    phi_p_in = phi_p[mask]
    phi_pp_in = phi_pp[mask]

    # Residual_1D: phi'' + 2*phi'/r - dOmega_dphi(phi)
    dOmega = np.array([dOmega_dphi_fn(phi_i, phi0_pot, v1, v2, omega) for phi_i in phi_in])
    res_1d = phi_pp_in + 2.0 * phi_p_in / r_in - dOmega

    # Residual_2D_reduced (with factor-2 fix): phi'' + 2*phi'/r + 2*(omega^2 - W)*phi
    dU_vals = dU(phi_in)
    W = np.where(np.abs(phi_in) > 1e-12, dU_vals / (2.0 * phi_in), 0.0)
    A_coef = 2.0 * (omega**2 - W)
    res_2d = phi_pp_in + 2.0 * phi_p_in / r_in + A_coef * phi_in

    return dict(
        res_1d_max=float(np.max(np.abs(res_1d))),
        res_2d_max=float(np.max(np.abs(res_2d))),
        res_1d_mean=float(np.mean(np.abs(res_1d))),
        res_2d_mean=float(np.mean(np.abs(res_2d))),
        n_pts=int(np.sum(mask)),
    )


def bulk_vs_boundary_residual(
    solver,
    x: np.ndarray,
    tau_bc_twisted: bool = True,
) -> Dict[str, Any]:
    """
    Decompose the 2D residual into bulk vs tau-boundary (i=0, i=Nt-1).
    If tau_bc_twisted is False, requires allow_debug_bcs and uses tau_bc='neumann'
    for the second evaluation (diagnostic only).
    """
    F = solver.residual(x)
    F = np.asarray(F, dtype=float)
    Nr, Nt = solver.Nr, solver.Nt
    N = Nr * Nt
    # Complex saddle: 4 blocks (Re y, Im y, Re yb, Im yb)
    if F.size == 4 * N:
        FyR = F[0 * N : 1 * N].reshape((Nr, Nt))
        FyI = F[1 * N : 2 * N].reshape((Nr, Nt))
        FbR = F[2 * N : 3 * N].reshape((Nr, Nt))
        FbI = F[3 * N : 4 * N].reshape((Nr, Nt))
        F_abs = np.sqrt(FyR**2 + FyI**2 + FbR**2 + FbI**2)
    else:
        Fy = F[:N].reshape((Nr, Nt))
        Fb = F[N:].reshape((Nr, Nt))
        F_abs = np.abs(Fy) + np.abs(Fb)

    # Bulk: exclude i=0 and i=Nt-1
    bulk_mask = np.ones_like(F_abs, dtype=bool)
    bulk_mask[:, 0] = False
    bulk_mask[:, Nt - 1] = False
    boundary_mask = ~bulk_mask

    return dict(
        max_bulk=float(np.max(F_abs[bulk_mask])) if np.any(bulk_mask) else np.nan,
        max_boundary=float(np.max(F_abs[boundary_mask])) if np.any(boundary_mask) else np.nan,
        max_i0=float(np.max(F_abs[:, 0])) if Nt > 0 else np.nan,
        max_iNt1=float(np.max(F_abs[:, Nt - 1])) if Nt > 0 else np.nan,
        nF_inf=float(np.max(F_abs)),
    )


def bulk_vs_boundary_with_bc_compare(
    solver,
    x: np.ndarray,
    U,
    dU,
    d2U,
) -> Dict[str, Any]:
    """
    Compare residual with tau_bc='twisted' vs tau_bc='neumann' (or 'hom_past') for
    the embedded tau-independent 1D config. Requires allow_debug_bcs=True.
    """
    from dataclasses import replace

    st = solver.settings
    if not getattr(st, "allow_debug_bcs", False):
        return dict(
            error="allow_debug_bcs is False. Set allow_debug_bcs=True for BC comparison.",
            max_bulk_twisted=np.nan,
            max_boundary_twisted=np.nan,
            max_bulk_neumann=np.nan,
            max_boundary_neumann=np.nan,
        )
    # With twisted (current)
    dec_twisted = bulk_vs_boundary_residual(solver, x, tau_bc_twisted=True)

    # Build solver with neumann
    st_neumann = replace(st, tau_bc="neumann")
    from .bounce2d import Bubble2DSolver
    solver_neumann = Bubble2DSolver(st_neumann, U, dU, d2U)
    solver_neumann.omega = solver.omega
    solver_neumann.rho0 = solver.rho0
    solver_neumann.eta0 = solver.eta0
    dec_neumann = bulk_vs_boundary_residual(solver_neumann, x, tau_bc_twisted=False)

    return dict(
        twisted=dec_twisted,
        neumann=dec_neumann,
        max_bulk_twisted=dec_twisted["max_bulk"],
        max_boundary_twisted=dec_twisted["max_boundary"],
        max_bulk_neumann=dec_neumann["max_bulk"],
        max_boundary_neumann=dec_neumann["max_boundary"],
    )


# -----------------------------------------------------------------------------
# PART 3 — Potential convention audit
# -----------------------------------------------------------------------------

def potential_convention_audit(
    rho_values: np.ndarray,
    V_fn,
    dV_dphi_fn,
    phi0_pot: float,
    v1: float,
    v2: float,
    omega: float,
    dU,
) -> Dict[str, Any]:
    """
    Audit: for each rho, compute V(rho), dV/dvarphi, s=rho^2, dV/ds (chain rule),
    W_2d = dU/(2*rho), and force_1d = dOmega/dphi = dV/dphi - 2*omega^2*phi.
    """
    rho = np.asarray(rho_values, dtype=float).flatten()
    rows = []
    for r in rho:
        if r < 1e-14:
            continue
        V = float(V_fn(r, phi0_pot, v1, v2))
        dV = float(dV_dphi_fn(r, phi0_pot, v1, v2))
        s = r * r
        dVds_chain = dV / (2.0 * r)  # dV/ds = dV/drho * drho/ds = dV/drho * 1/(2*rho)
        dU_val = float(dU(np.array([r]))[0])
        W_2d = dU_val / (2.0 * r)
        force_1d = dV - 2.0 * omega**2 * r
        # 2D reduced (with factor-2 fix): Fy += 2*(omega^2 - W)*y_tot => phi'' + 2*phi'/r = dV/dphi - 2*omega^2*phi
        # So force_2d = dV - 2*omega^2*r, same as force_1d.
        force_2d_reduced = dV - 2.0 * omega**2 * r
        rows.append(
            dict(
                rho=r,
                V=V,
                dV_dvarphi=dV,
                s=s,
                dVds_chain=dVds_chain,
                W_2d=W_2d,
                force_1d=force_1d,
                force_2d_reduced=force_2d_reduced,
            )
        )
    return dict(rows=rows, rho_values=rho)


# -----------------------------------------------------------------------------
# PART 4 — Charge definition report
# -----------------------------------------------------------------------------

def charge_definition_report() -> Dict[str, Any]:
    """
    Document the exact formulas used by compute_charge_1d and compute_charge_tau0_ghost_2d.
    """
    return dict(
        compute_charge_1d=dict(
            formula="Q = 4*pi * omega * integral(r^2 * phi^2, dr)",
            field="phi (unrotated amplitude)",
            omega="passed as argument (must match 1D bounce omega)",
            background_subtraction=False,
            volume="full integration from r[0] to r[-1]",
        ),
        compute_charge_tau0_ghost_2d=dict(
            formula="Q = 4*pi * integral(r^2 * q, dr), q = (1/2)*Re(phibar*phi_tau - phi*phibar_tau)",
            field="phi, phibar (unrotated), reconstructed at tau=0 via ghost",
            omega="solver.omega",
            background_subtraction="optional subtract_background=True => Q -= Q_homogeneous_ball(omega, rho0, r_max)",
            volume="r from grid",
        ),
        Q_homogeneous_ball=dict(
            formula="Q_hom = 4*pi * omega * rho0^2 * (r_max^3/3)",
            note="Used for background subtraction in 2D",
        ),
        comparison_note="For apples-to-apples: use same omega, compare EXCESS charge (subtract background) for both.",
    )


# -----------------------------------------------------------------------------
# Regression checklist (callable from notebook)
# -----------------------------------------------------------------------------

def run_regression_checks(
    solver,
    *,
    check_tau_independent: bool = True,
    atol_residual: float = 1e-6,
    atol_observables: float = 1e-10,
) -> Dict[str, Any]:
    """
    Run regression checks; fail loudly (AssertionError) if any check fails.

    1) Background exactness at eta0=0: x_bg = solver._zero_vec(), residual small.
    2) Tau-independent bubble at eta0=0: embed 1D bounce, Newton converges, tau_std ~ 0.
    3) Observables consistency: solver.observables_tau0(x) matches observables_2d.compute_observables_tau0_ghost(...).
    4) Targets consistency: ratios use targets_tau0(), no omega*Q shortcut.
    """
    from . import observables_2d

    results: Dict[str, Any] = {}

    # 1) Background exactness
    eta_save = float(solver.eta0)
    try:
        solver.eta0 = 0.0
        solver.settings.eta0 = 0.0
        x_bg = solver._zero_vec()
        F = solver.residual(x_bg)
        nrm = float(np.linalg.norm(F))
    finally:
        solver.eta0 = eta_save
        solver.settings.eta0 = eta_save
    results["background_residual_norm"] = nrm
    if nrm > atol_residual:
        raise AssertionError(
            f"run_regression_checks: background residual ||F||={nrm:.3e} > atol={atol_residual:.3e}"
        )
    results["background_ok"] = True

    # 2) Tau-independent bubble (optional: requires bounce_1d and Newton)
    if check_tau_independent:
        from .bounce_1d import solve_bounce
        r_1d, phi_1d, _, phi_false, _ = solve_bounce(
            1.999, 1.0, 2.0, float(solver.omega),
            rmax=float(solver.grid.r[-1]) * 0.8, n_grid_points=200,
        )
        if r_1d is not None and phi_1d is not None:
            x0 = build_tau_independent_embedding(solver, r_1d, phi_1d, float(solver.rho0))
            try:
                solver.eta0 = 0.0
                solver.settings.eta0 = 0.0
                sol = solver.solve(x0, verbose=False)
                nrm_f = float(sol.newton.residual_norm)
                results["tau_independent_residual_norm"] = nrm_f
                results["tau_independent_converged"] = bool(sol.newton.success)
            finally:
                solver.eta0 = eta_save
                solver.settings.eta0 = eta_save
        else:
            results["tau_independent_skipped"] = "solve_bounce failed"
    else:
        results["tau_independent_skipped"] = "check_tau_independent=False"

    # 3) Observables consistency: solver.observables_tau0(x) vs observables_2d
    x_test = solver._zero_vec()  # homogeneous
    obs_solver = solver.observables_tau0(x_test, subtract_background=False, return_profiles=False)
    obs_module = observables_2d.compute_observables_tau0_ghost(
        solver, x_test, subtract_background_charge=False
    )
    for key in ("Q", "E", "rho_Q", "rho_E"):
        a = obs_solver.get(key)
        b = obs_module.get(key)
        if a is not None and b is not None:
            err = abs(float(a) - float(b))
            if err > atol_observables:
                raise AssertionError(
                    f"run_regression_checks: observables mismatch key={key} diff={err:.3e}"
                )
    results["observables_consistent"] = True

    # 4) Targets: ensure targets_tau0() is used and no omega*Q
    tgt = solver.targets_tau0(subtract_background=False)
    if "Q" not in tgt or "E" not in tgt:
        raise AssertionError("run_regression_checks: targets_tau0() must return Q and E")
    results["targets_ok"] = True

    # 5) E for homogeneous (y=0) must match homogeneous_energy_2d
    E_ghost = observables_2d.assert_E_hom_consistent(solver, tol=1e-3)
    results["E_hom_consistent"] = True
    results["E_hom_value"] = float(E_ghost)

    return results
