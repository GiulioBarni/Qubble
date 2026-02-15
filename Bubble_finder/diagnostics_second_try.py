# Bubble_finder/diagnostics_second_try.py
"""
Helper per la sezione "Second try: diagnostics" del notebook bubble_2D.
Seed τ-dipendenti omogenei in r, seed τ-indipendenti da bolla 1D, e report Newton.
"""
from __future__ import annotations
import numpy as np


def build_x_from_phi(solver, phi: np.ndarray, phibar: np.ndarray, enforce_y0_zero: bool = True) -> np.ndarray:
    """
    Costruisce il vettore di stato x a partire da phi, phibar (campi "rotati" / densità).

    Input:
        phi, phibar: array shape (Nr, Ntau) sul grid del solver.
    Convenzione: phi_rot = rho0 + y/r, quindi y = r*(phi - rho0).
    enforce_y0_zero: se True (default), impone y[0]=ybar[0]=0 (regolarità).
        Se False, mantiene y[0]=r[0]*(phi[0]-rho0) così rho(r=0) = phi[0] (es. centro bolla).
    """
    phi = np.asarray(phi, dtype=complex)
    phibar = np.asarray(phibar, dtype=complex)
    r = np.asarray(solver.grid.r, dtype=float)
    if r.ndim == 1:
        r = r[:, None]  # (Nr, 1)
    rho0 = float(getattr(solver, "rho0", getattr(solver.settings, "rho0", 0.0)))

    y = r * (phi - rho0)
    ybar = r * (phibar - rho0)
    if enforce_y0_zero:
        y[0, :] = 0.0
        ybar[0, :] = 0.0

    return solver.pack(y, ybar)


def _tau_ramp_halfbox(tau: np.ndarray) -> np.ndarray:
    """Rampa h(τ): 0 a τ_min, 1 a τ_max (come ansatz_bubble._tau_ramp_halfbox)."""
    tmin, tmax = float(np.min(tau)), float(np.max(tau))
    if abs(tmax - tmin) < 1e-14:
        return np.zeros_like(tau, float)
    u = np.clip((tau - tmin) / (tmax - tmin), 0.0, 1.0)
    return 3.0 * u * u - 2.0 * u * u * u  # smoothstep


def build_tau_ramped_hom_seed(solver, rho_ref: float, rho_tilde: float) -> np.ndarray:
    """
    Seed omogeneo in r con rampa τ come build_seed_bubble:
      rho(τ) = rho_ref + h(τ)*(rho_tilde - rho_ref),
    con h=0 a τ_min, h=1 a τ_max (smoothstep).
    rho_ref = rho_hom(omega_ref), rho_tilde = φ_bubble(r_max).
    """
    tau = np.asarray(solver.grid.tau, dtype=float)
    h_bg = _tau_ramp_halfbox(tau)
    rho_tau = rho_ref + h_bg * (rho_tilde - rho_ref)
    return build_tau_dependent_hom_seed(solver, rho_tau)


def build_tau_dependent_hom_seed(solver, rho_tau) -> np.ndarray:
    """
    Seed omogeneo in r ma τ-dipendente: phi(r,τ)=rho_tau(τ), phibar=phi.

    rho_tau: array shape (Ntau,) oppure callable(tau) -> array/float.
    Restituisce x0 (vettore di stato).
    """
    tau = np.asarray(solver.grid.tau, dtype=float)
    Nr, Nt = solver.Nr, solver.Nt
    if callable(rho_tau):
        rho_vals = np.asarray([float(rho_tau(t)) for t in tau], dtype=float)
    else:
        rho_vals = np.asarray(rho_tau, dtype=float).ravel()
        if rho_vals.size != Nt:
            rho_vals = np.interp(np.linspace(0, 1, Nt), np.linspace(0, 1, rho_vals.size), rho_vals)
    # phi(r,τ) = rho_vals(τ) per ogni r
    phi = np.broadcast_to(rho_vals[None, :], (Nr, Nt)).copy()
    phibar = phi.copy()
    return build_x_from_phi(solver, phi, phibar)


def build_tau_independent_bubble_seed(
    solver,
    r_1d: np.ndarray,
    rho_1d: np.ndarray,
    rho_hom=None,
) -> np.ndarray:
    """
    Seed bolla 1D continuata in τ: phi(r,τ)=profilo(r) per ogni τ, phibar=phi.

    Interpola rho_1d(r) su solver.grid.r. Per r > max(r_1d) usa rho_hom
    (come nella cella esempio: interpolazione tra rho_bubble e rho_hom al bordo).
    rho_hom: valore omogeneo al bordo (es. phi_false_omega); se None usa solver.rho0
    o rho_1d[-1].
    """
    r_grid = np.asarray(solver.grid.r, dtype=float)
    r_1d = np.asarray(r_1d, dtype=float)
    rho_1d = np.asarray(rho_1d, dtype=float)
    if r_1d.size != rho_1d.size:
        raise ValueError("r_1d e rho_1d devono avere la stessa lunghezza")
    order = np.argsort(r_1d)
    r_1d = r_1d[order]
    rho_1d = rho_1d[order]
    if rho_hom is None:
        rho_hom = float(getattr(solver, "rho0", getattr(solver.settings, "rho0", rho_1d[-1])))
    else:
        rho_hom = float(rho_hom)
    # Al bordo (r > max(r_1d)) usa rho_hom come nella cella ρ(r_max,τ) vs ρ_hom
    rho_r = np.interp(r_grid, r_1d, rho_1d, left=rho_1d[0], right=rho_hom)
    Nr, Nt = solver.Nr, solver.Nt
    phi = np.broadcast_to(rho_r[:, None], (Nr, Nt)).copy()
    phibar = phi.copy()
    # enforce_y0_zero=False: rho(r≈0) = phi[0] = valore centro bolla, non rho0
    return build_x_from_phi(solver, phi, phibar, enforce_y0_zero=False)


def run_newton_and_report(
    solver,
    x0: np.ndarray,
    name: str = "",
    store_history: bool = True,
):
    """
    Esegue Newton a partire da x0, stampa residuo iniziale, min/max rho_cross,
    check τ-dipendenza (es. norma di rho(rmax,τ)-mean), e ritorna result + rho_init, rho_final.

    Usa solver.residual (o residual_fields) e solver.solve(..., store_iteration_history=store_history).
    """
    x0 = np.asarray(x0)
    F0 = getattr(solver, "residual_fields", solver.residual)(x0)
    nF0 = float(np.linalg.norm(F0))
    print(f"[{name}] Residuo iniziale ||F|| = {nF0:.6e}")

    # rho_cross da maps_from_x se disponibile, altrimenti da solver.phi
    try:
        from Bubble_finder.diagnostics_qfixed import maps_from_x
        M0 = maps_from_x(solver, x0)
        rho_cross = M0["rho_cross"]
    except Exception:
        y0, ybar0 = solver.unpack(x0)
        phi0, phibar0 = solver.phi(y0, ybar0)
        rho_cross = np.sqrt(np.maximum((phi0 * phibar0).real, 0.0))
    print(f"[{name}] rho_cross: min = {float(np.min(rho_cross)):.6f}, max = {float(np.max(rho_cross)):.6f}")

    # τ-dipendenza: rho(rmax, τ) - mean
    rho_rmax = rho_cross[-1, :]
    tau_var = float(np.std(rho_rmax))
    print(f"[{name}] τ-var @ r_max (std): {tau_var:.6e}")

    try:
        NewtonConvergenceError = __import__("Bubble_finder.bounce2d", fromlist=["NewtonConvergenceError"]).NewtonConvergenceError
    except Exception:
        try:
            NewtonConvergenceError = __import__("Q_ball_finder.nr_solver", fromlist=["NewtonConvergenceError"]).NewtonConvergenceError
        except Exception:
            NewtonConvergenceError = RuntimeError

    try:
        result = solver.solve(x0, verbose=True, store_iteration_history=store_history)
    except Exception as e:
        if isinstance(e, NewtonConvergenceError) or "does not converge" in str(e).lower() or "exceeds threshold" in str(e):
            print(f"[{name}] Newton NON convergente: {e}")
            print(f"[{name}] Si restituisce result=None, rho_final=rho_init (per plot diagnostici).")
            return None, rho_cross, rho_cross.copy()
        raise

    nF_final = float(np.linalg.norm(getattr(solver, "residual_fields", solver.residual)(result.x)))
    print(f"[{name}] Newton: success={result.success}, iters={result.newton.iterations}, final ||F|| = {nF_final:.6e}")

    try:
        from Bubble_finder.diagnostics_qfixed import maps_from_x
        Mf = maps_from_x(solver, result.x)
        rho_final = Mf["rho_cross"]
    except Exception:
        yf, ybarf = solver.unpack(result.x)
        phif, phibarf = solver.phi(yf, ybarf)
        rho_final = np.sqrt(np.maximum((phif * phibarf).real, 0.0))

    return result, rho_cross, rho_final
