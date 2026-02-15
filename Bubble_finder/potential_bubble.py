"""
Potential functions for bubble nucleation.

This module provides the potential V(φ) and its derivatives for the bubble nucleation
model. The potential is:
    V(φ) = -1/2 + (φ - 1)² [2φ - 5 + (2 - φ)² log((2-φ)²(φ₀-1)²/((φ-1)²(φ₀-2)²))]

This potential has a false minimum at φ = v1 = 1 and true minimum at φ = v2 = 2.
"""

from __future__ import annotations

import numpy as np


def V_phi(phi, phi0, v1, v2):
    """
    Underlying potential: V(φ) = -1/2 + (φ - 1)² [2φ - 5 + (2 - φ)² log((2-φ)²(φ₀-1)²/((φ-1)²(φ₀-2)²))]

    This potential has false minimum at φ = v1 = 1 and true minimum at φ = v2 = 2.
    Valid for φ in [v1, v2] = [1, 2].

    Parameters
    ----------
    phi : float or array
        Field value(s)
    phi0 : float
        Reference field value (typically 1.999)
    v1 : float
        False vacuum value (typically 1.0)
    v2 : float
        True vacuum value (typically 2.0)

    Returns
    -------
    float or array
        Potential value(s)
    """
    # Check if input is scalar or array
    is_scalar = np.isscalar(phi) or (isinstance(phi, np.ndarray) and phi.ndim == 0)

    # Convert to array for calculations
    phi_arr = np.asarray(phi)
    phi_flat = phi_arr.flatten() if phi_arr.ndim > 0 else np.array([phi_arr])

    # Evaluate the potential as-is, without clipping the domain.
    # For the log argument only: avoid singularities at phi=v1 (denominator 0)
    # and phi=v2 (numerator 0). Do NOT clip to [v1,v2]: leave phi > v2 free.
    eps = 1e-12
    phi_for_log = np.where(
        phi_flat <= v1 + eps,
        v1 + eps,
        np.where(phi_flat >= v2, np.maximum(phi_flat, v2 + eps), phi_flat),
    )

    # Compute log argument: (2-φ)²(φ₀-1)²/((φ-1)²(φ₀-2)²)
    log_arg = ((v2 - phi_for_log) ** 2 * (phi0 - v1) ** 2) / (
        (phi_for_log - v1) ** 2 * (phi0 - v2) ** 2
    )
    log_term = np.log(np.maximum(log_arg, 1e-300))  # Avoid log(0)

    # Compute potential: V = -1/2 + (φ - 1)² [2φ - 5 + (2 - φ)² log(...)]
    V = -0.5 + (phi_flat - v1) ** 2 * (2.0 * phi_flat - 5.0 + (v2 - phi_flat) ** 2 * log_term)

    # Return scalar or array as appropriate
    if is_scalar:
        return float(V[0])
    else:
        return V.reshape(phi_arr.shape)


def dV_dphi(phi, phi0, v1, v2):
    """
    First derivative of V(φ) with respect to φ.

    Must be consistent with V_phi: use clipping only for log calculation, not for other terms.

    Parameters
    ----------
    phi : float or array
        Field value(s)
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value

    Returns
    -------
    float or array
        First derivative value(s)
    """
    # Check if input is scalar or array (same as V_phi)
    is_scalar = np.isscalar(phi) or (isinstance(phi, np.ndarray) and phi.ndim == 0)
    phi_arr = np.asarray(phi)
    phi_flat = phi_arr.flatten() if phi_arr.ndim > 0 else np.array([phi_arr])

    # Same as V_phi: avoid log singularities at v1,v2 but do not clip to [v1,v2]; leave phi > v2 free
    eps = 1e-12
    phi_for_log = np.where(
        phi_flat <= v1 + eps,
        v1 + eps,
        np.where(phi_flat >= v2, np.maximum(phi_flat, v2 + eps), phi_flat),
    )

    # Terms using original phi (not clipped) - consistent with V_phi
    term1 = (phi_flat - v1) ** 2  # (φ - 1)²
    term2 = 2.0 * phi_flat - 5.0  # 2φ - 5
    term3 = (v2 - phi_flat) ** 2  # (2 - φ)²

    # Log argument and log term
    log_arg = ((v2 - phi_for_log) ** 2 * (phi0 - v1) ** 2) / (
        (phi_for_log - v1) ** 2 * (phi0 - v2) ** 2
    )
    log_term = np.log(np.maximum(log_arg, 1e-300))

    # Derivative of log term: d/dφ log((2-φ)²(φ₀-1)²/((φ-1)²(φ₀-2)²))
    # = d/dφ [2 log(2-φ) + 2 log(φ₀-1) - 2 log(φ-1) - 2 log(φ₀-2)]
    # = -2/(2-φ) - 2/(φ-1) = -2(2-φ + φ-1)/((2-φ)(φ-1)) = -2/((2-φ)(φ-1))
    dlog_dphi = -2.0 / ((v2 - phi_for_log) * (phi_for_log - v1))

    # dV/dφ = d/dφ [-1/2 + (φ-1)² [2φ-5 + (2-φ)² log(...)]]
    # = 2(φ-1)[2φ-5 + (2-φ)² log(...)] + (φ-1)²[2 + d/dφ((2-φ)² log(...))]
    # = 2(φ-1)[2φ-5 + (2-φ)² log(...)] + (φ-1)²[2 + 2(2-φ)(-1)log(...) + (2-φ)² dlog/dφ]
    dV = (
        2.0 * (phi_flat - v1) * (term2 + term3 * log_term)
        + term1 * (2.0 - 2.0 * (v2 - phi_flat) * log_term + term3 * dlog_dphi)
    )

    # Return scalar or array as appropriate
    if is_scalar:
        return float(dV[0])
    else:
        return dV.reshape(phi_arr.shape)


def d2V_dphi2(phi, phi0, v1, v2):
    """
    Second derivative of V(φ) with respect to φ.

    Computed numerically for robustness.

    Parameters
    ----------
    phi : float or array
        Field value(s)
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value

    Returns
    -------
    float or array
        Second derivative value(s)
    """
    h = 1e-6
    dV_plus = dV_dphi(phi + h, phi0, v1, v2)
    dV_minus = dV_dphi(phi - h, phi0, v1, v2)
    return (dV_plus - dV_minus) / (2 * h)


def Omega_phi(phi, phi0, v1, v2, omega):
    """
    Grand potential: Ω(φ) = V(φ) - ω² φ² (consistent with 2D PDE: W(ρ²)=ω² ⇒ V'(ρ)=2ω²ρ).

    Parameters
    ----------
    phi : float or array
        Field value(s)
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value
    omega : float
        Chemical potential

    Returns
    -------
    float or array
        Grand potential value(s)
    """
    V = V_phi(phi, phi0, v1, v2)
    return V - omega**2 * phi**2


def dOmega_dphi(phi, phi0, v1, v2, omega):
    """
    First derivative of grand potential: dΩ/dφ = dV/dφ - 2 ω² φ (Ω = V - ω² φ²).

    Parameters
    ----------
    phi : float or array
        Field value(s)
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value
    omega : float
        Chemical potential

    Returns
    -------
    float or array
        First derivative value(s)
    """
    dV = dV_dphi(phi, phi0, v1, v2)
    return dV - 2.0 * omega**2 * phi


def d2Omega_dphi2(phi, phi0, v1, v2, omega):
    """
    Second derivative of grand potential: d²Ω/dφ² = d²V/dφ² - 2 ω² (Ω = V - ω² φ²).

    Parameters
    ----------
    phi : float or array
        Field value(s)
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value
    omega : float
        Chemical potential

    Returns
    -------
    float or array
        Second derivative value(s)
    """
    d2V = d2V_dphi2(phi, phi0, v1, v2)
    return d2V - 2.0 * omega**2


def W_of_s(s, phi0, v1, v2, omega):
    """
    Potential derivative for use in 2D solver: W(s) = dΩ/dφ / (2*ρ) where s = ρ² = (φ φ̄).real

    This is used in the 2D solver where we work with s = real(phi*phibar) instead of phi directly.
    Uses the grand potential Ω(φ) = V(φ) - ω²φ².

    Parameters
    ----------
    s : float or array
        s = real(phi*phibar), must be non-negative
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value
    omega : float
        Chemical potential

    Returns
    -------
    float or array
        W(s) = dOmega_dphi(ρ) / (2*ρ) where ρ = sqrt(s)
    """
    # Ensure s is non-negative
    s = np.maximum(s, 0.0)
    rho = np.sqrt(s)
    
    # Avoid division by zero (rho ~ O(1) typically, but guard anyway)
    mask = rho > 1e-10
    W = np.zeros_like(s)
    
    if np.any(mask):
        dOmega = dOmega_dphi(rho[mask], phi0, v1, v2, omega)
        W[mask] = dOmega / (2.0 * rho[mask])
    
    # For rho ≈ 0, use L'Hôpital: lim_{ρ→0} dΩ/dφ / (2ρ) = d²Ω/dφ²(0) / 2
    if np.any(~mask):
        d2Omega_at_zero = d2Omega_dphi2(0.0, phi0, v1, v2, omega)
        W[~mask] = d2Omega_at_zero / 2.0
    
    return W


def Wp_of_s(s, phi0, v1, v2, omega):
    """
    Derivative of W(s) with respect to s: Wp(s) = d/ds W(s)

    Using chain rule: d/ds W(s) = d/dρ W(ρ²) * dρ/ds = [d²Ω/dφ²(ρ)*ρ - dΩ/dφ(ρ)] / (4*ρ³)

    Parameters
    ----------
    s : float or array
        s = real(phi*phibar), must be non-negative
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value
    omega : float
        Chemical potential

    Returns
    -------
    float or array
        Wp(s) = d/ds W(s)
    """
    # Ensure s is non-negative
    s = np.maximum(s, 0.0)
    rho = np.sqrt(s)
    
    # Avoid division by zero
    mask = rho > 1e-10
    Wp = np.zeros_like(s)
    
    if np.any(mask):
        dOmega = dOmega_dphi(rho[mask], phi0, v1, v2, omega)
        d2Omega = d2Omega_dphi2(rho[mask], phi0, v1, v2, omega)
        Wp[mask] = (d2Omega * rho[mask] - dOmega) / (4.0 * rho[mask] ** 3)
    
    # For rho ≈ 0, use expansion
    if np.any(~mask):
        d2Omega_at_zero = d2Omega_dphi2(0.0, phi0, v1, v2, omega)
        d2Omega_eps = d2Omega_dphi2(1e-6, phi0, v1, v2, omega)
        d3Omega_approx = (d2Omega_eps - d2Omega_at_zero) / 1e-6
        Wp[~mask] = d3Omega_approx / 8.0  # Approximate from expansion
    
    return Wp


def V_W_of_s(s, phi0, v1, v2):
    """
    Potential derivative for use in 2D solver: W(s) = dV/dφ / (2*ρ) where s = ρ² = (φ φ̄).real
    
    This version uses V(φ) directly, NOT the grand potential Ω(φ).
    Use this when the operator already includes explicit ω terms (Q-ball style).
    
    This solver uses the Q-ball-style operator with explicit ω terms; therefore W(s) 
    must be derived from V(φ), not Ω(φ). Ω is used only for diagnostics (energy), not in the EOM.

    Parameters
    ----------
    s : float or array
        s = real(phi*phibar), must be non-negative
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value

    Returns
    -------
    float or array
        W(s) = dV_dphi(ρ) / (2*ρ) where ρ = sqrt(s)
    """
    # Ensure s is non-negative
    s = np.maximum(s, 0.0)
    rho = np.sqrt(s)
    
    # Avoid division by zero (rho ~ O(1) typically, but guard anyway)
    mask = rho > 1e-10
    W = np.zeros_like(s)
    
    if np.any(mask):
        dV = dV_dphi(rho[mask], phi0, v1, v2)
        W[mask] = dV / (2.0 * rho[mask])
    
    # For rho ≈ 0, use L'Hôpital: lim_{ρ→0} dV/dφ / (2ρ) = d²V/dφ²(0) / 2
    if np.any(~mask):
        d2V_at_zero = d2V_dphi2(0.0, phi0, v1, v2)
        W[~mask] = d2V_at_zero / 2.0
    
    return W


def V_Wp_of_s(s, phi0, v1, v2):
    """
    Derivative of V_W_of_s(s) with respect to s: Wp(s) = d/ds W(s)
    
    IMPORTANT: This returns dW/ds (derivative with respect to s = phi*phibar),
    NOT dW/drho (derivative with respect to rho = sqrt(s)).

    Using chain rule: d/ds W(s) = d/dρ W(ρ²) * dρ/ds = [d²V/dφ²(ρ)*ρ - dV/dφ(ρ)] / (4*ρ³)
    where dρ/ds = 1/(2*ρ) since s = ρ².
    
    This version uses V(φ) directly, NOT the grand potential Ω(φ).
    Use this when the operator already includes explicit ω terms (Q-ball style).

    Parameters
    ----------
    s : float or array
        s = real(phi*phibar), must be non-negative
    phi0 : float
        Reference field value
    v1 : float
        False vacuum value
    v2 : float
        True vacuum value

    Returns
    -------
    float or array
        Wp(s) = d/ds W(s) where the derivative is with respect to s, not rho
    """
    # Ensure s is non-negative
    s = np.maximum(s, 0.0)
    rho = np.sqrt(s)
    
    # Avoid division by zero
    mask = rho > 1e-10
    Wp = np.zeros_like(s)
    
    if np.any(mask):
        dV = dV_dphi(rho[mask], phi0, v1, v2)
        d2V = d2V_dphi2(rho[mask], phi0, v1, v2)
        Wp[mask] = (d2V * rho[mask] - dV) / (4.0 * rho[mask] ** 3)
    
    # For rho ≈ 0, use expansion
    if np.any(~mask):
        d2V_at_zero = d2V_dphi2(0.0, phi0, v1, v2)
        d2V_eps = d2V_dphi2(1e-6, phi0, v1, v2)
        d3V_approx = (d2V_eps - d2V_at_zero) / 1e-6
        Wp[~mask] = d3V_approx / 8.0  # Approximate from expansion
    
    return Wp


def vacua_of_Omega(phi0, v1, v2, omega, verbose=True,
                   *, ngrid=4000, bracket_frac=0.10, xtol=1e-12, distinct_tol=1e-6):
    """
    Find the two local minima (vacua) of the grand potential Ω(φ) = V(φ) - ω² φ².
    Search range: [v1, v2 + delta] so the true vacuum can sit slightly above v2
    when the -ω²φ² term shifts the minimum (no artificial clipping at v2).

    Strategy:
      1) Grid scan to locate local minima (including boundaries if lower than neighbours).
      2) Take the leftmost and rightmost minima.
      3) Refine each with minimize_scalar(method="bounded") in a neighbourhood.

    Returns
    -------
    (phi_false, phi_true) with phi_false metastable (higher Ω) and phi_true stable (lower Ω).

    Raises
    ------
    RuntimeError
        If two distinct minima are not found (typically ω too large → single minimum).
    """
    from scipy.optimize import minimize_scalar

    if not np.isfinite(phi0) or not np.isfinite(v1) or not np.isfinite(v2) or not np.isfinite(omega):
        raise ValueError("phi0, v1, v2, omega must be finite.")
    if v2 <= v1:
        raise ValueError(f"Require v2 > v1, got v1={v1}, v2={v2}.")
    if omega < 0:
        raise ValueError("omega must be >= 0 (use omega^2 if you only need the square).")

    # Allow true vacuum slightly above v2 (Ω minimum can shift right due to -ω²φ²)
    v2_max = v2 + 0.2

    # --- 1) Grid scan in [v1, v2_max]
    xs = np.linspace(v1, v2_max, int(ngrid))
    Om = Omega_phi(xs, phi0, v1, v2, omega)

    if not np.all(np.isfinite(Om)):
        bad = np.where(~np.isfinite(Om))[0][:10]
        raise RuntimeError(f"Omega_phi produced NaN/inf on grid. Example indices: {bad}.")

    # Discrete local minima (including boundaries if lower than neighbour)
    mins = []

    # Left boundary
    if Om[0] <= Om[1]:
        mins.append(0)

    # Interior
    inner = np.where((Om[1:-1] < Om[:-2]) & (Om[1:-1] < Om[2:]))[0] + 1
    mins.extend(inner.tolist())

    # Right boundary
    if Om[-1] <= Om[-2]:
        mins.append(len(xs) - 1)

    if len(mins) < 2:
        raise RuntimeError(
            f"Could not find two distinct local minima in [v1, v2_max]=[{v1}, {v2_max}] for ω={omega}. "
            "Probably ω is too large and one of the two wells disappears."
        )

    # Take leftmost and rightmost minima
    iL, iR = mins[0], mins[-1]
    xL0, xR0 = float(xs[iL]), float(xs[iR])

    # --- 2) Bounded refinement
    width = max(bracket_frac * (v2 - v1), 5.0 * (v2 - v1) / max(ngrid, 10))

    def refine(x0):
        a = max(v1, x0 - width)
        b = min(v2_max, x0 + width)
        if b <= a:
            return float(x0)
        res = minimize_scalar(
            lambda p: Omega_phi(p, phi0, v1, v2, omega),
            bounds=(a, b),
            method="bounded",
            options={"xatol": xtol},
        )
        if not res.success or not np.isfinite(res.x):
            raise RuntimeError(f"Refine failed around x0={x0}: success={res.success}, x={res.x}")
        return float(res.x)

    phi1 = refine(xL0)
    phi2 = refine(xR0)

    # --- 3) Check distinct
    if abs(phi1 - phi2) < max(distinct_tol, distinct_tol * (v2 - v1)):
        raise RuntimeError(
            f"The two minima collapse: φ1={phi1:.12g}, φ2={phi2:.12g} (ω={omega}). "
            "This usually indicates only one minimum remains."
        )

    # --- 4) Order as false/true by Ω
    Om1 = float(Omega_phi(phi1, phi0, v1, v2, omega))
    Om2 = float(Omega_phi(phi2, phi0, v1, v2, omega))

    if Om1 > Om2:
        phi_false, phi_true = phi1, phi2
        Om_false, Om_true = Om1, Om2
    else:
        phi_false, phi_true = phi2, phi1
        Om_false, Om_true = Om2, Om1

    # --- 5) Diagnostics
    if verbose:
        def W_at_rho(rho):
            rho = float(rho)
            if abs(rho) < 1e-12:
                return float(d2V_dphi2(0.0, phi0, v1, v2) / 2.0)
            return float(dV_dphi(rho, phi0, v1, v2) / (2.0 * rho))

        # dΩ/dφ = dV/dφ - 2 ω² φ (should be ~0 at interior minima)
        rF = float(dOmega_dphi(phi_false, phi0, v1, v2, omega))
        rT = float(dOmega_dphi(phi_true,  phi0, v1, v2, omega))

        mF = float(omega**2 - W_at_rho(phi_false))
        mT = float(omega**2 - W_at_rho(phi_true))

        # Curvature: Ω = V - ω² φ²  =>  Ω'' = V'' - 2 ω²
        cF = float(d2V_dphi2(phi_false, phi0, v1, v2) - 2.0 * omega**2)
        cT = float(d2V_dphi2(phi_true,  phi0, v1, v2) - 2.0 * omega**2)

        print(f"[vacua_of_Omega] ω={omega:.6g}")
        print(f"  phi_false={phi_false:.12g}  Ω_false={Om_false:.12g}  dΩ={rF:.2e}  (ω²-W)={mF:.2e}  Ω''={cF:.3e}")
        print(f"  phi_true ={phi_true :.12g}  Ω_true ={Om_true :.12g}  dΩ={rT:.2e}  (ω²-W)={mT:.2e}  Ω''={cT:.3e}")

    return phi_false, phi_true



__all__ = [
    "V_phi",
    "dV_dphi",
    "d2V_dphi2",
    "Omega_phi",
    "dOmega_dphi",
    "d2Omega_dphi2",
    "W_of_s",
    "Wp_of_s",
    "V_W_of_s",
    "V_Wp_of_s",
    "vacua_of_Omega",
]
