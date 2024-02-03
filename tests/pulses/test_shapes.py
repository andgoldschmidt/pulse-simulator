import numpy as np
from scipy.integrate import simpson
import sympy

import pytest

import pulse_simulator


def test_resample():
    times = np.linspace(0, 2 * np.pi, 201)
    cos2 = np.cos(times) ** 2
    dt = times[1] - times[0]
    a = simpson(cos2, dx=dt)

    cos2_1, dt1 = pulse_simulator.resample(
        len(times) // 2, cos2, keep_step=False, return_step=True
    )
    a1 = simpson(cos2_1, dx=dt1 * dt)
    assert np.isclose(a1, a, atol=1e-1)

    cos2_2, dt2 = pulse_simulator.resample(
        len(times) * 2, cos2, keep_step=False, return_step=True
    )
    a2 = simpson(cos2_2, dx=dt2 * dt)
    assert np.isclose(a2, a, atol=1e-1)


@pytest.fixture
def envelope():
    # Scalable symbolic envelope
    pass


def test_fit_spline_envelope(envelope):
    pass
    # _t, _amp, _angle = sympy.symbols("t, amp, angle")
    # λ_fn = sympy.utilities.lambdify([_t, _amp, _angle], envelope)
    # x = np.arange(0, T)
    # y = [λ_fn(i, 1, 0).real for i in x]
