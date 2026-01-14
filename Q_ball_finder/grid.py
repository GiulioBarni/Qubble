"""
Utilities to construct the (r, τ) lattice and convert between field
representations used in the 2D bounce solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

ArrayLike = np.ndarray | float


@dataclass(frozen=True)
class RadialTimeGrid:
    r: np.ndarray
    tau: np.ndarray
    dr: float
    dtau: float
    Nr: int
    Ntau: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.Nr, self.Ntau


def build_grid(Nr: int, Ntau: int, Lr: float, beta: float) -> RadialTimeGrid:
    """
    Create the radial/time lattice following paper conventions (arXiv:1711.05279v2):
        r_j   = dr (j + 1),             j = 0..Nr-1  (no half-step offset)
        tau_i = -dtau (i + 1/2),        i = 0..Ntau-1  (domain -beta/2 < tau < 0)
    
    where dr = Lr / Nr and dtau = beta / (2 * Ntau).
    """
    dr = Lr / Nr
    dtau = beta / (2.0 * Ntau)

    r = dr * (np.arange(Nr) + 1.0)
    tau = -dtau * (np.arange(Ntau) + 0.5)
    return RadialTimeGrid(r=r, tau=tau, dr=dr, dtau=dtau, Nr=Nr, Ntau=Ntau)


def pack_fields(y: np.ndarray, ybar: np.ndarray) -> np.ndarray:
    """Flatten y and ybar arrays of shape (Nr, Ntau) into a single vector."""
    if y.shape != ybar.shape:
        raise ValueError("y and ybar must have the same shape.")
    return np.concatenate([y.ravel(), ybar.ravel()])


def unpack_fields(vec: np.ndarray, Nr: int, Ntau: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_fields."""
    total = Nr * Ntau
    if vec.size != 2 * total:
        raise ValueError(f"Vector size {vec.size} incompatible with Nr={Nr}, Ntau={Ntau}.")
    y = vec[:total].reshape(Nr, Ntau)
    ybar = vec[total:].reshape(Nr, Ntau)
    return y, ybar


def phi_from_y(y: np.ndarray, grid: RadialTimeGrid, omega: float) -> np.ndarray:
    """Recover φ(r, τ) from y(r, τ) via y = r e^{-ωτ} φ."""
    r = grid.r[:, None]
    tau = grid.tau[None, :]
    return np.exp(+omega * tau) * y / r


def phi_from_ybar(ybar: np.ndarray, grid: RadialTimeGrid, omega: float) -> np.ndarray:
    """Recover \bar φ(r, τ) from ybar(r, τ) via ybar = r e^{+ωτ} \bar φ."""
    r = grid.r[:, None]
    tau = grid.tau[None, :]
    return np.exp(-omega * tau) * ybar / r


def y_from_phi(phi: np.ndarray, grid: RadialTimeGrid, omega: float) -> np.ndarray:
    """Compute y given φ."""
    r = grid.r[:, None]
    tau = grid.tau[None, :]
    return r * np.exp(-omega * tau) * phi


def ybar_from_phi(phibar: np.ndarray, grid: RadialTimeGrid, omega: float) -> np.ndarray:
    """Compute ybar given \bar φ."""
    r = grid.r[:, None]
    tau = grid.tau[None, :]
    return r * np.exp(+omega * tau) * phibar


__all__ = [
    "RadialTimeGrid",
    "build_grid",
    "pack_fields",
    "unpack_fields",
    "phi_from_y",
    "phi_from_ybar",
    "y_from_phi",
    "ybar_from_phi",
]

