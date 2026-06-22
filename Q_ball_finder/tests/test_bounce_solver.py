import numpy as np

from .. import bounce_solver, potentials, qball_observables


def test_quartic_bounce_matches_notebook_values():
    """Quartic bounce regression values."""
    V, dV = potentials.quartic_potential(eta=0.2)

    solution = bounce_solver.solve_bounce(
        V,
        dV=dV,
        d=3,
        x_min=-1.0,
        x_max=6.0,
        allow_unbounded=False,
        verbose=False,
    )

    # Reference values from notebook cell 7 output
    assert np.isclose(solution.SE, 541.8031962, rtol=5e-3)
    assert np.isclose(solution.phi0, 3.6111976, rtol=5e-3)
    assert np.isclose(solution.false_vac["x"], 3.98848e-07, atol=1e-5)
    assert np.isclose(solution.r_end, 17.7155, rtol=5e-3)

    # --- Q-ball configuration (logistic potential, ω=0.8765 m) ---
    params = potentials.LogisticPotentialParams(m=1.0, v=1.0, b=8.0)
    omega = 0.8765 * params.m
    V_hat, dV_hat = potentials.effective_qball_potential(params, omega)

    qball_solution = bounce_solver.solve_bounce(
        V_hat,
        dV=dV_hat,
        d=3,
        x_min=-1.0,
        x_max=6.0,
        allow_unbounded=True,
        phi0_cap=10.0,
        prefer_side="+",
        verbose=False,
    )

    g2Q = qball_observables.compute_dimensionless_charge(
        qball_solution, m=params.m, v=params.v, omega=omega
    )
    energy = qball_observables.compute_energy(
        qball_solution, omega=omega, potential_chi=potentials.logistic_potential_chi(params)[0]
    )

    print(f"[Q-ball] omega = {omega:.6f}, g^2 Q = {g2Q:.6f}, E = {energy:.6f}")

