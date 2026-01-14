import numpy as np

from ..nr_solver import newton_solve


def test_newton_solver_simple_quadratic():
    def residual(x):
        return np.array([x[0] ** 2 - 2.0], dtype=float)

    def jacobian(x):
        return np.array([[2.0 * x[0]]], dtype=float)

    res = newton_solve(residual, jacobian, np.array([1.0]))
    assert res.success
    assert np.isclose(res.x[0], np.sqrt(2.0), atol=1e-8)

