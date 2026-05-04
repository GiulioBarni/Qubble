"""
Pytest for BubbleSolver2D (microcanonical 2D bubble, Q-ball-like).
Run from the Qubble project root:  pytest Bubble_finder/tests/test_bubble_solver_2d.py -v
Or from Bubble_finder:              pytest tests/test_bubble_solver_2d.py -v
(Ensure the project root is on PYTHONPATH so Q_ball_finder is found.)
"""
from __future__ import annotations

import os
import sys

# Ensure the project root and Bubble_finder are on path
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_bubble_finder = os.path.dirname(_tests_dir)
_project_root = os.path.dirname(_bubble_finder)
for _d in (_project_root, _bubble_finder):
    if _d not in sys.path:
        sys.path.insert(0, _d)

try:
    from Bubble_finder.bounce2d import (
        Bubble2DSettings,
        Bubble2DSolver,
        make_potential_from_V,
    )
except ImportError:
    from bounce2d import (
        Bubble2DSettings,
        Bubble2DSolver,
        make_potential_from_V,
    )
try:
    from Bubble_finder.potential_bubble import V_phi, dV_dphi, d2V_dphi2
except ImportError:
    from potential_bubble import V_phi, dV_dphi, d2V_dphi2


def _get_potential():
    """External potential as in notebook (potential_bubble)."""
    return make_potential_from_V(V_phi, dV_dphi, d2V_dphi2, phi0=1.999, v1=1.0, v2=2.0)


RHO0_BRACKET = (1.0, 1.3)  # potential_bubble: root V'(ρ)=2ω²ρ (false vacuum)


def test_sanity_background():
    """Background y=ybar=0 must give residual ~0 (twist+shift correct)."""
    U, dU, d2U = _get_potential()
    settings = Bubble2DSettings(
        Nr=80, Ntau=160, Lr=15.0, beta=30.0,
        omega_ref=0.4, eta0=0.0, tau_bc="twisted", verbose=False,
    )
    settings.rho0_bracket = RHO0_BRACKET
    solver = Bubble2DSolver(settings, U, dU, d2U)
    out = solver.sanity_background()
    assert out["residual_norm"] < 1e-6, f"residual_norm={out['residual_norm']}"


def test_check_jacobian():
    """J@v vs finite-difference directional derivative."""
    U, dU, d2U = _get_potential()
    settings = Bubble2DSettings(
        Nr=40, Ntau=80, Lr=12.0, beta=20.0,
        omega_ref=0.35, eta0=0.0, tau_bc="twisted", verbose=False,
        rho0_bracket=RHO0_BRACKET,
    )
    solver = Bubble2DSolver(settings, U, dU, d2U)
    y0 = solver.grid.r[:, None] * 0.01 * (solver.grid.tau[None, :] + 1.0)
    yb0 = y0.copy()
    x0 = solver.pack(y0, yb0)
    out = solver.check_jacobian(x0, n_tests=4, eps=1e-6)
    assert out["max_rel"] < 0.02, f"max_rel={out['max_rel']}"


def test_demo_newton_few_steps():
    """Ansatz + a few Newton steps (no convergence required)."""
    U, dU, d2U = _get_potential()
    settings = Bubble2DSettings(
        Nr=60, Ntau=120, Lr=15.0, beta=25.0,
        omega_ref=0.4, eta0=0.0, tau_bc="twisted", verbose=False,
        newton_max_iter=3,
        rho0_bracket=RHO0_BRACKET,
    )
    solver = Bubble2DSolver(settings, U, dU, d2U)
    x0 = solver.build_anchored_initial_guess(rho_center=2.0, Rb=3.0, Lw=0.5)
    sol = solver.solve(x0)
    assert sol.newton.iterations == 3
    assert sol.newton.residual_norm >= 0
