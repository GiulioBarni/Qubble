"""
Utility functions and data structures for the scalar potentials that appear
in the Q-ball analysis.  The primary target is the logarithmic potential used
in the reference notebook, but a couple of auxiliary toy potentials are also
exposed for testing the one-dimensional bounce solver.

All functions are vectorised through NumPy so they can be used seamlessly with
arrays.  The derivatives are coded analytically for good numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.special import expit

ArrayLike = np.ndarray | float


@dataclass(frozen=True)
class LogisticPotentialParams:
    """
    Parameters for the potential

        V(ρ) = - (m^2 v^2 / b) · log( e^{-b ρ / v^2} + c ),
        c = e^{-b} / (1 + e^{-b}),

    with ρ = |φ|^2.  In the notebook χ denotes |φ| and is frequently used
    instead of ρ; conversion helpers are provided below.
    """

    m: float = 1.0
    v: float = 1.0
    b: float = 8.0

    def _log_c(self) -> float:
        # store log(c) for numerical stability
        return np.log(np.exp(-self.b) / (1.0 + np.exp(-self.b)))


def logistic_potential_rho(
    params: LogisticPotentialParams,
) -> Tuple[
    Callable[[ArrayLike], ArrayLike],
    Callable[[ArrayLike], ArrayLike],
    Callable[[ArrayLike], ArrayLike],
]:
    """
    Return V(ρ), ∂V/∂ρ, and ∂²V/∂ρ² for the logistic potential defined by
    `params`.
    """
    m, v, b = params.m, params.v, params.b
    logc = params._log_c()

    def V_rho(rho: ArrayLike) -> ArrayLike:
        rho = np.asarray(rho, dtype=float)
        z = -b * rho / (v * v)
        # logaddexp improves stability when z is large negative
        return -(m * m * v * v / b) * np.logaddexp(z, logc)

    def dV_drho(rho: ArrayLike) -> ArrayLike:
        rho = np.asarray(rho, dtype=float)
        t = (b * rho / (v * v)) + logc
        return m * m * expit(-t)

    def d2V_drho2(rho: ArrayLike) -> ArrayLike:
        rho = np.asarray(rho, dtype=float)
        t = (b * rho / (v * v)) + logc
        sigma = expit(-t)
        return -m * m * (b / (v * v)) * sigma * (1.0 - sigma)

    return V_rho, dV_drho, d2V_drho2


def logistic_potential_chi(
    params: LogisticPotentialParams,
) -> Tuple[
    Callable[[ArrayLike], ArrayLike],
    Callable[[ArrayLike], ArrayLike],
    Callable[[ArrayLike], ArrayLike],
]:
    """
    Same potential, now expressed in terms of χ = |φ|.  We use ρ = χ² / 2,
    consistently with the notebook.
    """
    V_rho, dV_drho, d2V_drho2 = logistic_potential_rho(params)

    def V_chi(chi: ArrayLike) -> ArrayLike:
        chi = np.asarray(chi, dtype=float)
        rho = 0.5 * chi * chi
        return V_rho(rho)

    def dV_dchi(chi: ArrayLike) -> ArrayLike:
        chi = np.asarray(chi, dtype=float)
        rho = 0.5 * chi * chi
        return chi * dV_drho(rho)

    def d2V_dchi2(chi: ArrayLike) -> ArrayLike:
        chi = np.asarray(chi, dtype=float)
        rho = 0.5 * chi * chi
        return dV_drho(rho) + chi * chi * d2V_drho2(rho)

    return V_chi, dV_dchi, d2V_dchi2


def effective_qball_potential(
    params: LogisticPotentialParams, omega: float
) -> Tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]:
    """
    Construct the effective potential V_hat(χ) = V(χ) - ½ ω² χ² and its first
    derivative with respect to χ.  This is the potential fed to the 1D bounce
    solver when computing Q-ball profiles.
    """
    V_chi, dV_dchi, _ = logistic_potential_chi(params)
    omega2 = omega * omega

    def V_hat(chi: ArrayLike) -> ArrayLike:
        chi = np.asarray(chi, dtype=float)
        return V_chi(chi) - 0.5 * omega2 * chi * chi

    def dV_hat_dchi(chi: ArrayLike) -> ArrayLike:
        chi = np.asarray(chi, dtype=float)
        return dV_dchi(chi) - omega2 * chi

    return V_hat, dV_hat_dchi


# --------------------------------------------------------------------------- #
# Auxiliary toy potentials
# --------------------------------------------------------------------------- #


def quartic_potential(
    eta: float = 0.2,
) -> Tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]:
    """
    Bounded quartic potential often used for testing:
        V(χ) = ½ χ² - ⅓ χ³ + (η/4) χ⁴.
    """

    def V(chi: ArrayLike) -> ArrayLike:
        chi = np.asarray(chi, dtype=float)
        return 0.5 * chi * chi - (1.0 / 3.0) * chi * chi * chi + (eta / 4.0) * chi**4

    def dV(chi: ArrayLike) -> ArrayLike:
        chi = np.asarray(chi, dtype=float)
        return chi - chi * chi + eta * chi * chi * chi

    return V, dV


def unbounded_cubic_quartic(
    kappa: float = 0.02,
) -> Tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]:
    """
    Unbounded potential:
        V(φ) = ½ φ² - (1/3) φ³ - (κ / 4) φ⁴,
    useful to exercise the unbounded branch of the solver.
    """

    def V(phi: ArrayLike) -> ArrayLike:
        phi = np.asarray(phi, dtype=float)
        return 0.5 * phi * phi - (1.0 / 3.0) * phi**3 - (kappa / 4.0) * phi**4

    def dV(phi: ArrayLike) -> ArrayLike:
        phi = np.asarray(phi, dtype=float)
        return phi - phi * phi - kappa * phi**3

    return V, dV


__all__ = [
    "ArrayLike",
    "LogisticPotentialParams",
    "effective_qball_potential",
    "logistic_potential_chi",
    "logistic_potential_rho",
    "quartic_potential",
    "unbounded_cubic_quartic",
]

