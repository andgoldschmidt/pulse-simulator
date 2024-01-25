import functools
import jax
import numpy as np
from qiskit.providers.fake_provider import FakeManila
import qiskit_dynamics
from qiskit import pulse
import pytest

import pulse_simulator

# Configuration for jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
qiskit_dynamics.array.Array.set_default_backend("jax")


@pytest.fixture
def registers():
    return [0, 1]


@pytest.fixture
def backend():
    return FakeManila()


@pytest.fixture
def units():
    return 1e9  # GHz & ns


@pytest.fixture
def timing(backend, units):
    ns = units
    dt = backend.configuration().dt * ns
    duration = 225 * dt  # ns
    return {"dt": dt, "duration": duration}


def test_timing(timing):
    # Check for reasonable bounds on timesteps
    assert 10**-2 < timing["dt"] < 10**2


@pytest.fixture
def signals(registers, timing):
    dt = timing["dt"]
    duration = timing["duration"]

    # Play signals on all registers
    # Implement an x-gate, no rotating frame 1/2 on the control H.
    x_gate = pulse_simulator.gaussian_envelope(dt, duration, angle=np.pi / 2)
    with pulse.build(name="Moment") as signals:
        for i in registers:
            pulse.play(x_gate, pulse.DriveChannel(i))
    return signals


@pytest.fixture
def config(backend, units):
    return pulse_simulator.backend_simulation_vars(backend, units=units)


@pytest.fixture
def one_qubit_model(registers, backend, config):
    return functools.partial(
        pulse_simulator.rx_model,
        registers=registers,
        backend=backend,
        variables=config,
        rotating_frame=True,
    )


@pytest.fixture
def one_qubit_decay(registers, config):
    return functools.partial(
        pulse_simulator.qubit_decay_model,
        registers=registers,
        variables=config,
    )


@pytest.fixture
def crosstalk(registers, backend, config):
    return pulse_simulator.crosstalk_model(
        registers, pulse_simulator.backend_edges(backend), config
    )


def test_nonzero_zz_crosstalk(crosstalk):
    # Value could change if backend changes
    assert np.max(np.abs(crosstalk.data)) > 0.0001
    signature = np.sign(np.diag(np.real(crosstalk.data / crosstalk.data[0, 0])))
    assert np.all(signature == [1, -1, -1, 1])


@pytest.fixture
def one_qubit_solver(registers, one_qubit_model, one_qubit_decay, crosstalk):
    # Solver parameters
    Hs_drift = []
    Hs_control = []
    Hs_channels = []
    As_static = []
    for qubit in registers:
        Hj_drift, Hjs_control, Hjs_channel = one_qubit_model(qubit)
        Ajs_static = one_qubit_decay(qubit)

        Hs_drift += [Hj_drift]
        Hs_control += Hjs_control
        Hs_channels += Hjs_channel
        As_static += Ajs_static

    print(Hs_drift)
    return {
        "static_hamiltonian_drift": np.sum(Hs_drift, axis=0),
        "static_hamiltonian_crosstalk": crosstalk,
        "hamiltonian_operators": Hs_control,
        "static_dissipators": As_static,
        "hamiltonian_channels": Hs_channels,
        "channel_carrier_freqs": {ch: 0.0 for ch in Hs_channels},
    }


def test_statevector_sim_closed(registers, timing, signals, one_qubit_solver):
    y0 = pulse_simulator.qiskit_ground_state(len(registers))
    probs_expected = np.array([0.0, 0.0, 0.0, 1.0])

    # Unpack timing parameters
    dt = timing["dt"]
    duration = timing["duration"]

    # Unpack solver parameters
    static_hamiltonian_drift = one_qubit_solver["static_hamiltonian_drift"]
    static_hamiltonian_crosstalk = one_qubit_solver["static_hamiltonian_crosstalk"]
    hamiltonian_operators = one_qubit_solver["hamiltonian_operators"]
    hamiltonian_channels = one_qubit_solver["hamiltonian_channels"]
    channel_carrier_freqs = one_qubit_solver["channel_carrier_freqs"]

    # Closed, no crosstalk
    # =====
    solver_closed_drift = qiskit_dynamics.Solver(
        static_hamiltonian=static_hamiltonian_drift,
        hamiltonian_operators=hamiltonian_operators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_carrier_freqs,
        dt=dt,
    )

    # Dense, jax_expm
    # -----
    solver_closed_drift.model.evaluation_mode = "dense"
    sol_dense_expm = solver_closed_drift.solve(
        t_span=[0, duration],
        y0=y0,
        signals=signals,
        max_dt=dt,
        t_eval=[0, duration],
        method="jax_expm",
        magnus_order=1,
    )
    yf = sol_dense_expm.y[-1].data
    probs_f = np.abs(yf) ** 2
    assert np.isclose(np.abs(yf) ** 2, probs_expected, atol=1e-6).all()

    # Sparse, jax_odeint
    # -----
    solver_closed_drift.model.evaluation_mode = "sparse"
    sol_sparse_odeint = solver_closed_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=y0,
        signals=signals,
        atol=1e-8,
        rtol=1e-8,
        method="jax_odeint",
    )
    yf = sol_sparse_odeint.y[-1].data
    assert np.isclose(np.abs(yf) ** 2, probs_f, atol=1e-6).all()
    assert np.isclose(np.abs(yf) ** 2, probs_expected, atol=1e-6).all()

    # Closed, with crosstalk
    # =====
    solver_closed_crosstalk = qiskit_dynamics.Solver(
        static_hamiltonian=static_hamiltonian_crosstalk,
        hamiltonian_operators=hamiltonian_operators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_carrier_freqs,
        dt=dt,
    )

    # Dense, jax_expm
    # -----
    solver_closed_crosstalk.model.evaluation_mode = "dense"
    sol_dense_expm_zz = solver_closed_crosstalk.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=y0,
        signals=signals,
        max_dt=dt,
        method="jax_expm",
        magnus_order=1,
    )
    # Loss of fidelity due to crosstalk
    y_zz_f = sol_dense_expm_zz.y[-1].data
    probs_zz_f = np.abs(y_zz_f) ** 2
    assert not np.isclose(probs_zz_f, probs_f, atol=1e-6).all()

    # But still close to expected
    assert np.isclose(probs_f, probs_expected, atol=1e-3).all()


# def test_statevector_sim_open(self):
#     r0 = pulse_simulator.qiskit_ground_state(self.registers, open=True)

# def test_unitary_sim_closed(self):
#     U0 = pulse_simulator.qiskit_identity_operator(self.registers)

# def test_unitary_sim_open(self):
#     C0 = pulse_simulator.qiskit_identity_operator(self.registers, open=True)
