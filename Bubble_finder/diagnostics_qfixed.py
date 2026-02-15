# Bubble_finder/diagnostics_qfixed.py
from __future__ import annotations
import numpy as np

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
except Exception:
    sp = None
    eigsh = None


# -----------------------------------------------------------------------------
# Low-level: pack (y,ybar) -> x_fields
# -----------------------------------------------------------------------------
def pack_from_unpacked_views(solver, y, ybar):
    """
    Costruisce x_fields da (y,ybar). Usa solver.pack se disponibile.
    """
    if hasattr(solver, "pack"):
        return solver.pack(y, ybar)
    raise RuntimeError(
        "solver.pack non disponibile. Serve un metodo pack(y, ybar) nel solver."
    )


# -----------------------------------------------------------------------------
# Ansatz builder: tau "indipendente" per Q fissata
# -----------------------------------------------------------------------------
def _get_beta_eta0(solver):
    beta = None
    eta0 = 0.0

    # prova settings
    if hasattr(solver, "settings"):
        beta = getattr(solver.settings, "beta", None)
        eta0 = float(getattr(solver.settings, "eta0", 0.0))

    # fallback: ricava beta dalla griglia (half-box tipico: tau in [-beta/2, 0])
    tau = np.asarray(solver.grid.tau, float)
    if beta is None or not np.isfinite(beta) or beta <= 0:
        beta = 2.0 * abs(float(tau.min()))  # assume tau_min ~ -beta/2

    return float(beta), float(eta0)


def build_background_fields(solver, rho=None, mode="twist_density"):
    """
    Background omogeneo:
      - mode="flat": y=ybar=rho costante (può violare twist se eta0!=0)
      - mode="twist_density": densità costante ma y~exp(+s tau), ybar~exp(-s tau)
    """
    if rho is None:
        rho = float(getattr(solver, "rho0", 0.0))

    r = np.asarray(solver.grid.r, float)
    tau = np.asarray(solver.grid.tau, float)

    beta, eta0 = _get_beta_eta0(solver)

    # riferimento tau ~ 0 (nella tua half-box tipicamente tau include 0)
    tau_ref = float(tau[np.argmin(np.abs(tau))])
    t = (tau - tau_ref)

    if mode == "flat":
        fac = np.ones_like(t)
        fac_inv = np.ones_like(t)
    elif mode == "twist_density":
        s = eta0 / beta
        fac = np.exp(s * t)
        fac_inv = np.exp(-s * t)
    else:
        raise ValueError("mode must be 'flat' or 'twist_density'")

    rho_r = np.full_like(r, float(rho))
    y = (rho_r[:, None] * fac[None, :]).astype(np.complex128)
    ybar = (rho_r[:, None] * fac_inv[None, :]).astype(np.complex128)

    x = pack_from_unpacked_views(solver, y, ybar)
    return x


def build_tauind_density_from_radial_profile(
    solver,
    r_1d,
    rho_1d,
    mode="twist_density",
    fill="edge",
):
    """
    Embedding 1D -> 2D:
      - densità in τ costante (a parte il twist esponenziale su y/ybar)
      - profilo radiale rho(r) interpolato sulla griglia del solver
    """
    r_grid = np.asarray(solver.grid.r, float)
    tau = np.asarray(solver.grid.tau, float)

    r_1d = np.asarray(r_1d, float)
    rho_1d = np.asarray(rho_1d, float)

    # pulizia monotonia per interp
    order = np.argsort(r_1d)
    r_1d = r_1d[order]
    rho_1d = rho_1d[order]

    if fill == "edge":
        left = float(rho_1d[0])
        right = float(rho_1d[-1])
    else:
        left = 0.0
        right = 0.0

    rho_r = np.interp(r_grid, r_1d, rho_1d, left=left, right=right)

    beta, eta0 = _get_beta_eta0(solver)
    tau_ref = float(tau[np.argmin(np.abs(tau))])
    t = (tau - tau_ref)

    if mode == "flat":
        fac = np.ones_like(t)
        fac_inv = np.ones_like(t)
    elif mode == "twist_density":
        s = eta0 / beta
        fac = np.exp(s * t)
        fac_inv = np.exp(-s * t)
    else:
        raise ValueError("mode must be 'flat' or 'twist_density'")

    y = (rho_r[:, None] * fac[None, :]).astype(np.complex128)
    ybar = (rho_r[:, None] * fac_inv[None, :]).astype(np.complex128)

    x = pack_from_unpacked_views(solver, y, ybar)
    return x


# -----------------------------------------------------------------------------
# Residual + map extraction
# -----------------------------------------------------------------------------
def residual_report(solver, x_fields, name=""):
    F = solver.residual_fields(x_fields)
    n2 = float(np.linalg.norm(F))
    ninf = float(np.max(np.abs(F)))
    return {
        "name": name,
        "norm2": n2,
        "norminf": ninf,
        "size": int(np.size(F)),
    }


def maps_from_x(solver, x_fields):
    """
    Estrae mappe utili per plottare.
    Per Bubble2DSolver: usa solver.phi(y,ybar) per ricavare phi,phibar fisici da y,ybar.
    """
    y, ybar = solver.unpack(x_fields)
    # Bubble2DSolver: phi = exp(omega*tau)*(rho0 + y/r), phibar = exp(-omega*tau)*(rho0 + ybar/r)
    if hasattr(solver, "phi"):
        phi, phibar = solver.phi(y, ybar)
    else:
        phi = np.asarray(y)
        phibar = np.asarray(ybar)

    rho_cross = np.sqrt(np.maximum((phi * phibar).real, 0.0))
    abs_phi = np.abs(phi)
    abs_phibar = np.abs(phibar)
    conj_err = np.abs(phibar - np.conj(phi))

    return {
        "phi": phi,
        "phibar": phibar,
        "rho_cross": rho_cross,
        "abs_phi": abs_phi,
        "abs_phibar": abs_phibar,
        "conj_err": conj_err,
    }


# -----------------------------------------------------------------------------
# Linear stability diagnostic (modo negativo): symmetrized Jacobian
# -----------------------------------------------------------------------------
def sym_jacobian_smallest_eigs(solver, x_fields, k=6):
    """
    Calcola i k autovalori più piccoli del Jacobiano simmetrizzato H=(J+J^T)/2.
    Richiede scipy sparse.
    """
    if sp is None or eigsh is None:
        raise RuntimeError("scipy.sparse / eigsh non disponibili nel tuo env.")

    # prendi Jacobiano
    if hasattr(solver, "jacobian_fields"):
        J = solver.jacobian_fields(x_fields)
    else:
        J = solver.jacobian(x_fields)

    if not sp.issparse(J):
        # fallback: prova a convertirlo (attenzione alla memoria)
        J = sp.csr_matrix(J)

    H = 0.5 * (J + J.T)

    # eigsh richiede k < N
    N = H.shape[0]
    kk = min(int(k), max(1, N - 2))

    evals, evecs = eigsh(H, k=kk, which="SA")  # smallest algebraic
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


def rayleigh_curvature_symJ(solver, x0, direction):
    """
    Curvatura lungo una direzione d: (d^T H d)/(d^T d), con H=(J+J^T)/2
    Economico e utile anche se non fai eigensolve grande.
    """
    if sp is None:
        raise RuntimeError("scipy.sparse non disponibile nel tuo env.")

    if hasattr(solver, "jacobian_fields"):
        J = solver.jacobian_fields(x0)
    else:
        J = solver.jacobian(x0)

    if not sp.issparse(J):
        J = sp.csr_matrix(J)

    H = 0.5 * (J + J.T)

    d = np.asarray(direction, float).copy()
    d_norm2 = float(d @ d)
    if d_norm2 == 0.0:
        return np.nan

    Hd = H @ d
    return float(d @ Hd) / d_norm2
