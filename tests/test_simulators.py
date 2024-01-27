import functools
import jax
import numpy as np
from qiskit.providers.fake_provider import FakeManila
import qiskit_dynamics
from qiskit import pulse, quantum_info
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


@pytest.fixture
def initial_statevec(registers):
    return pulse_simulator.qiskit_ground_state(len(registers))


def test_initial_statevec(initial_statevec):
    # Check that the initial state is normalized
    assert np.isclose(np.linalg.norm(initial_statevec), 1.0)

    # Check that the initial state is the ground state
    assert np.isclose(np.abs(initial_statevec[0]), 1.0)


@pytest.fixture
def initial_statemat(registers):
    psi = pulse_simulator.qiskit_ground_state(len(registers))
    return quantum_info.DensityMatrix(psi)


def test_initial_statemat(initial_statemat):
    # Check that the initial state is normalized
    assert np.isclose(initial_statemat.trace(), 1.0)

    # Check that the initial state is the ground state
    assert np.isclose(initial_statemat.data[0, 0], 1.0)


@pytest.fixture
def initial_unitary(registers):
    return pulse_simulator.qiskit_identity_operator(len(registers))


def test_initial_unitary(registers, initial_unitary):
    # Check that the initial state is normalized
    assert np.isclose(
        initial_unitary.adjoint() @ initial_unitary, np.identity(2 ** len(registers))
    ).all()


@pytest.fixture
def closed_drift(timing, one_qubit_solver):
    # Unpack timing parameters
    dt = timing["dt"]

    # Unpack solver parameters
    static_hamiltonian_drift = one_qubit_solver["static_hamiltonian_drift"]
    hamiltonian_operators = one_qubit_solver["hamiltonian_operators"]
    hamiltonian_channels = one_qubit_solver["hamiltonian_channels"]
    channel_carrier_freqs = one_qubit_solver["channel_carrier_freqs"]

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=static_hamiltonian_drift,
        hamiltonian_operators=hamiltonian_operators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_carrier_freqs,
        dt=dt,
    )
    return solver


@pytest.fixture
def closed_crosstalk(timing, one_qubit_solver):
    # Unpack timing parameters
    dt = timing["dt"]

    # Unpack solver parameters
    static_hamiltonian_drift = one_qubit_solver["static_hamiltonian_drift"]
    static_hamiltonian_crosstalk = one_qubit_solver["static_hamiltonian_crosstalk"]
    hamiltonian_operators = one_qubit_solver["hamiltonian_operators"]
    hamiltonian_channels = one_qubit_solver["hamiltonian_channels"]
    channel_carrier_freqs = one_qubit_solver["channel_carrier_freqs"]

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=static_hamiltonian_drift.data
        + static_hamiltonian_crosstalk.data,
        hamiltonian_operators=hamiltonian_operators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_carrier_freqs,
        dt=dt,
    )
    return solver


@pytest.fixture
def open_drift(timing, one_qubit_solver):
    # Unpack timing parameters
    dt = timing["dt"]

    # Unpack solver parameters
    static_hamiltonian_drift = one_qubit_solver["static_hamiltonian_drift"]
    hamiltonian_operators = one_qubit_solver["hamiltonian_operators"]
    static_dissipators = one_qubit_solver["static_dissipators"]
    hamiltonian_channels = one_qubit_solver["hamiltonian_channels"]
    channel_carrier_freqs = one_qubit_solver["channel_carrier_freqs"]

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=static_hamiltonian_drift,
        hamiltonian_operators=hamiltonian_operators,
        static_dissipators=static_dissipators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_carrier_freqs,
        dt=dt,
    )
    return solver


@pytest.fixture
def state_dense_closed_drift(initial_statevec, timing, signals, closed_drift):
    # Unpack timing parameters
    dt = timing["dt"]
    duration = timing["duration"]

    # Dense, jax_expm
    closed_drift.model.evaluation_mode = "dense"
    return closed_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_statevec,
        signals=signals,
        max_dt=dt,
        method="jax_expm",
        magnus_order=1,
    )


@pytest.fixture
def state_sparse_closed_drift(initial_statevec, timing, signals, closed_drift):
    duration = timing["duration"]

    # Sparse, jax_odeint
    closed_drift.model.evaluation_mode = "sparse"
    return closed_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_statevec,
        signals=signals,
        atol=1e-16,
        rtol=1e-16,
        method="jax_odeint",
    )


@pytest.fixture
def state_dense_closed_crosstalk(initial_statevec, timing, signals, closed_crosstalk):
    dt = timing["dt"]
    duration = timing["duration"]

    # With crosstalk
    closed_crosstalk.model.evaluation_mode = "dense"
    return closed_crosstalk.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_statevec,
        signals=signals,
        max_dt=dt,
        method="jax_expm",
        magnus_order=1,
    )


@pytest.fixture
def state_dense_open_drift(initial_statemat, timing, signals, open_drift):
    dt = timing["dt"]
    duration = timing["duration"]

    # With crosstalk
    open_drift.model.evaluation_mode = "dense_vectorized"
    return open_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_statemat,
        signals=signals,
        max_dt=dt,
        method="jax_expm",
        magnus_order=1,
    )


@pytest.fixture
def state_sparse_open_drift(initial_statemat, timing, signals, open_drift):
    duration = timing["duration"]

    # With crosstalk
    open_drift.model.evaluation_mode = "sparse_vectorized"
    return open_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_statemat,
        signals=signals,
        atol=1e-16,
        rtol=1e-16,
        method="jax_odeint",
    )


def test_state_sim_closed(state_dense_closed_drift, state_sparse_closed_drift):
    # Assume two X gates
    probs_expected = np.array([0.0, 0.0, 0.0, 1.0])

    # Dense, jax_expm
    yf = state_dense_closed_drift.y[-1].data
    probs_f = np.abs(yf) ** 2
    assert np.isclose(np.abs(yf) ** 2, probs_expected, atol=1e-6).all()

    # Sparse, jax_odeint
    yf = state_sparse_closed_drift.y[-1].data
    assert np.isclose(np.abs(yf) ** 2, probs_f, atol=1e-6).all()
    assert np.isclose(np.abs(yf) ** 2, probs_expected, atol=1e-6).all()


def test_state_sim_closed_crosstalk(
    state_dense_closed_drift, state_dense_closed_crosstalk
):
    # Assume two X gates
    probs_expected = np.array([0.0, 0.0, 0.0, 1.0])

    # Without crosstalk
    yf = state_dense_closed_drift.y[-1].data
    probs_f = np.abs(yf) ** 2

    # With crosstalk
    y_zz_f = state_dense_closed_crosstalk.y[-1].data
    probs_zz_f = np.abs(y_zz_f) ** 2

    # Loss of fidelity due to crosstalk
    assert not np.isclose(probs_zz_f, probs_f, atol=1e-6).all()

    # ...but still close to expected
    assert np.isclose(probs_f, probs_expected, atol=1e-3).all()


def test_state_sim_open(
    state_dense_closed_drift, state_dense_open_drift, state_sparse_open_drift
):
    # Closed
    rho_f_closed = quantum_info.DensityMatrix(state_dense_closed_drift.y[-1])

    # Open, dense
    rho_f_open_dense = state_dense_open_drift.y[-1].data

    # Open, sparse
    rho_f_open_sparse = state_sparse_open_drift.y[-1].data

    # Check that the solutions are close
    assert np.isclose(rho_f_open_sparse, rho_f_open_dense, atol=1e-6).all()

    # Check that the density matrices are close
    assert np.isclose(rho_f_closed, rho_f_open_dense, atol=1e-2).all()

    # ... but not identical
    assert not np.isclose(rho_f_closed, rho_f_open_dense, atol=1e-6).all()


@pytest.fixture
def unitary_dense_closed(initial_unitary, timing, signals, closed_drift):
    dt = timing["dt"]
    duration = timing["duration"]

    closed_drift.model.evaluation_mode = "dense"
    return closed_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_unitary,
        signals=signals,
        max_dt=dt,
        method="jax_expm",
        magnus_order=1,
    )


@pytest.fixture
def unitary_sparse_closed(initial_unitary, timing, signals, closed_drift):
    duration = timing["duration"]

    closed_drift.model.evaluation_mode = "sparse"
    return closed_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_unitary,
        signals=signals,
        atol=1e-16,
        rtol=1e-16,
        method="jax_odeint",
    )


@pytest.fixture
def unitary_dense_open(initial_unitary, timing, signals, open_drift):
    dt = timing["dt"]
    duration = timing["duration"]

    initial_channel = quantum_info.SuperOp(initial_unitary)

    open_drift.model.evaluation_mode = "dense_vectorized"
    return open_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_channel,
        signals=signals,
        max_dt=dt,
        method="jax_expm",
        magnus_order=1,
    )


@pytest.fixture
def unitary_sparse_open(initial_unitary, timing, signals, open_drift):
    duration = timing["duration"]

    initial_channel = quantum_info.SuperOp(initial_unitary)

    open_drift.model.evaluation_mode = "sparse_vectorized"
    return open_drift.solve(
        t_span=[0, duration],
        t_eval=[0, duration],
        y0=initial_channel,
        signals=signals,
        atol=1e-16,
        rtol=1e-16,
        method="jax_odeint",
    )


def test_unitary_sim_closed(registers, unitary_dense_closed, unitary_sparse_closed):
    # Dense, jax_expm
    yf1 = unitary_dense_closed.y[-1]
    assert np.isclose(yf1.adjoint() @ yf1, np.identity(2 ** len(registers))).all()

    # Sparse, jax_odeint
    yf2 = unitary_sparse_closed.y[-1]
    assert np.isclose(yf2.adjoint() @ yf2, np.identity(2 ** len(registers))).all()

    # Check that the solutions are close
    assert np.isclose(yf1, yf2, atol=1e-6).all()

    # Check the fidelity with the expected unitary
    y_expected = pulse_simulator.from_label("XX")
    assert np.isclose(quantum_info.process_fidelity(yf1, y_expected), 1.0, atol=1e-6)


def test_unitary_sim_open(registers, unitary_dense_open, unitary_sparse_open):
    # Dense, jax_expm
    y1 = unitary_dense_open.y
    assert y1[0].is_unitary()
    assert not y1[-1].is_unitary()

    # Sparse, jax_odeint
    yf2 = unitary_sparse_open.y[-1]

    # Check that the solutions are close
    assert np.isclose(y1[-1], yf2, atol=1e-6).all()

    # Check the fidelity with the expected unitary
    y_expected = pulse_simulator.from_label("XX")
    fidelity = quantum_info.process_fidelity(y1[-1], y_expected)
    assert not np.isclose(fidelity, 1.0, atol=1e-6)
    assert np.isclose(fidelity, 1.0, atol=1e-2)
