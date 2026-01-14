"""
Utilities to pre-compute and cache the scan in ω̃ used to match a target
charge between a metastable Q-ball and the corresponding Q-cloud.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from .potentials import LogisticPotentialParams
from .profiles import QBallProfile, compute_unstable_mode, solve_qball_profile
from .qball_observables import compute_dimensionless_charge

DEFAULT_SCAN_POINTS = 60
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCAN_FILENAME = DATA_DIR / "cloud_scan.txt"


@dataclass
class CloudScanData:
    params: LogisticPotentialParams
    omega_min: float
    omega_max: float
    omegas: np.ndarray
    wtilde: np.ndarray
    charges: np.ndarray
    metadata: Dict[str, float]
    cache_file: Path


def _metadata_matches(meta: Dict[str, float], expected: Dict[str, float]) -> bool:
    keys = set(meta.keys()) | set(expected.keys())
    for key in keys:
        a = meta.get(key)
        b = expected.get(key)
        if isinstance(a, float) or isinstance(b, float):
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                return False
        else:
            if a != b:
                return False
    return True


def load_or_compute_cloud_scan(
    params: LogisticPotentialParams,
    *,
    omega_min: float,
    omega_max: float,
    n_points: int = DEFAULT_SCAN_POINTS,
    phi0_cap: float = 10.0,
    x_min: float = -1.0,
    x_max: float = 6.0,
    prefer_side: str = "+",
) -> CloudScanData:
    """
    Compute (or load from cache) the scan in ω̃ used to match Q-ball and
    Q-cloud charges. Returns arrays of successful ω and corresponding charges.
    """
    expected_meta = {
        "m": params.m,
        "v": params.v,
        "b": params.b,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "n_points": float(n_points),
        "phi0_cap": phi0_cap,
        "x_min": x_min,
        "x_max": x_max,
        "prefer_side": {"-": -1.0, "+": 1.0}.get(prefer_side, 0.0),
    }

    cache_file = SCAN_FILENAME

    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
            if first_line.startswith("# metadata "):
                try:
                    meta = json.loads(first_line[len("# metadata ") :])
                except json.JSONDecodeError:
                    meta = {}
            else:
                meta = {}

            if _metadata_matches(meta, expected_meta):
                data_lines = [line for line in fh if not line.startswith("#")]
                if data_lines:
                    data = np.loadtxt(data_lines)
                    if data.ndim == 1:
                        data = data[None, :]
                    omegas = data[:, 0]
                    wtilde = data[:, 1]
                    charges = data[:, 2]
                    return CloudScanData(
                        params=params,
                        omega_min=omega_min,
                        omega_max=omega_max,
                        omegas=omegas,
                        wtilde=wtilde,
                        charges=charges,
                        metadata=meta,
                        cache_file=cache_file,
                    )

    wtilde_grid = np.linspace(omega_min / params.m, omega_max / params.m, n_points)
    omega_grid = wtilde_grid * params.m

    omegas: list[float] = []
    charges: list[float] = []

    for omega in omega_grid:
        try:
            profile = solve_qball_profile(
                params,
                omega,
                phi0_cap=phi0_cap,
                x_min=x_min,
                x_max=x_max,
                prefer_side=prefer_side,
            )
            charge = compute_dimensionless_charge(
                profile.solution, m=params.m, v=params.v, omega=omega
            )
            omegas.append(omega)
            charges.append(charge)
        except Exception:
            continue

    if len(omegas) < 2:
        raise RuntimeError("Not enough Q-cloud points computed to build scan.")

    omegas_arr = np.array(omegas, dtype=float)
    charges_arr = np.array(charges, dtype=float)
    wtilde_arr = omegas_arr / params.m

    with cache_file.open("w", encoding="utf-8") as fh:
        fh.write("# metadata " + json.dumps(expected_meta) + "\n")
        fh.write("# columns: omega  wtilde  charge\n")
        for om, wt, ch in zip(omegas_arr, wtilde_arr, charges_arr):
            fh.write(f"{om:.12e} {wt:.12e} {ch:.12e}\n")

    return CloudScanData(
        params=params,
        omega_min=omega_min,
        omega_max=omega_max,
        omegas=omegas_arr,
        wtilde=wtilde_arr,
        charges=charges_arr,
        metadata=expected_meta,
        cache_file=cache_file,
    )


def find_cloud_frequency_for_charge(
    scan: CloudScanData,
    target_charge: float,
) -> float:
    """
    Use the cached scan to find ω̃ whose charge matches the target. Returns ω.
    """
    omegas = scan.omegas
    charges = scan.charges
    diff = charges - target_charge

    bracket: Optional[Tuple[int, int]] = None
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            bracket = (i, i + 1)
            break
    if bracket is None:
        raise RuntimeError("Could not bracket target charge within cached scan.")

    i_lo, i_hi = bracket
    f_interp = interp1d(
        omegas,
        diff,
        kind="cubic",
        fill_value="extrapolate",
        bounds_error=False,
    )
    omega_star = float(brentq(f_interp, omegas[i_lo], omegas[i_hi], xtol=1e-8, rtol=1e-8))
    return omega_star


def build_qcloud_profile_for_charge(
    params: LogisticPotentialParams,
    target_charge: float,
    *,
    omega_min: float,
    omega_max: float,
    n_points: int = DEFAULT_SCAN_POINTS,
    phi0_cap: float = 10.0,
    x_min: float = -1.0,
    x_max: float = 6.0,
    prefer_side: str = "+",
) -> Tuple[float, QBallProfile]:
    """
    Retrieve (or compute) the cached scan and solve once more for the Q-cloud
    profile whose charge matches `target_charge`.
    """
    scan = load_or_compute_cloud_scan(
        params,
        omega_min=omega_min,
        omega_max=omega_max,
        n_points=n_points,
        phi0_cap=phi0_cap,
        x_min=x_min,
        x_max=x_max,
        prefer_side=prefer_side,
    )
    omega_cloud = find_cloud_frequency_for_charge(scan, target_charge)
    profile_cloud = solve_qball_profile(
        params,
        omega_cloud,
        phi0_cap=phi0_cap,
        x_min=x_min,
        x_max=x_max,
        prefer_side=prefer_side,
    )
    return omega_cloud, profile_cloud


__all__ = [
    "CloudScanData",
    "build_qcloud_profile_for_charge",
    "find_cloud_frequency_for_charge",
    "load_or_compute_cloud_scan",
]

