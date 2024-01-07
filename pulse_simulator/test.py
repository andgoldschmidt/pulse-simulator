import simulator
import pulse_simulator as ps
import functools
import qiskit_dynamics as qk_d
import qiskit.providers.fake_provider as qk_fp
import numpy as np
import csv
import qiskit

backend = qk_fp.FakeManila()
units = 1e9
ns = units
dt = backend.configuration().dt * ns

N = 5  # number of spins
hz = 1.0 * 2 * np.pi  # magnetic field along z
Jx = 1.0 * 2 * np.pi  # Coupling along x
Δt = 0.05  # time step for integration
tlist = np.arange(50) * Δt  # time values

registers = [i for i in range(N)]
config_vars = ps.backend_simulation_vars(backend, rabi=False, units=units)

H_rx = functools.partial(
    ps.rx_model,
    registers=registers,
    backend=backend,
    variables=config_vars,
    rotating_frame=False,
)

Hs_control = []
Hs_channels = []
for qubit in registers:
    Hj_drift, Hjs_control, Hjs_channel = H_rx(qubit)
    Hs_control += Hjs_control
    Hs_channels += Hjs_channel

H_xtalk = ps.crosstalk_model(registers, ps.backend_edges(backend), config_vars)

solver = qk_d.Solver(
    static_hamiltonian=H_xtalk,
    hamiltonian_operators=Hs_control,
    static_dissipators=None,
    rotating_frame=None,
    rwa_cutoff_freq=None,
    hamiltonian_channels=Hs_channels,
    channel_carrier_freqs={ch: 0.0 for ch in Hs_channels},
    dt=dt,
)

sim = ps.simulator.Simulator(
    basis_gates=["rz", "sx", "x", "cx"], solver=solver, backend=backend
)

# load and set pulses
file_name = "./pico-pulses/saved-pulses-2023-12-13/a_single_qubit_gateset_R1e-6.csv"
gates = []
with open(file_name) as file:
    reader = csv.reader(file)
    for row in reader:
        gates.append(np.array([float(x) for x in row]))

expected_angles = [np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4]
for i in range(len(gates)):
    normalization = np.trapz(gates[i], dx=dt) / expected_angles[i]
    gates[i] = gates[i] / normalization

for pulse, name in zip(gates, ["x_blue", "x_red", "sx_blue", "sx_red"]):
    pulse = qiskit.pulse.Waveform(pulse, limit_amplitude=False)
    sim.set_pulse(name, pulse)

def create_initial_state(qr, cr):
    """
    Creates the initial state for the experiment.
    Here we would like to prepare the |+>|+>|+>|+> state.
    """
    circ = qiskit.QuantumCircuit(qr, cr)
    [circ.h(qr[i]) for i in range(qr.size)]
    return circ


def rxx(circ, θx, q1, q2):
    """
    Implements the exp(-i theta/2 sx_1 sx_2) gate.
    """
    circ.cx(q1, q2)
    circ.rx(θx, q1)
    circ.cx(q1, q2)
    return circ


def first_order_Trotter_unitary(circ, Δt, barrier=False):
    """
    Applies a Unitary for a single time step U(Δt) using first-order trotter expansion to the input quantum circuit.
    input : circ is a quantum circuit
    output : circ-U(Δt)-
    """
    # Layer A: coupling XX between n and n+1
    # Induce parallel CR gates
    if barrier:
        circ.barrier()
    for p in range(0, circ.num_qubits - 1, 2):
        circ = rxx(circ, Jx * Δt, circ.qubits[p], circ.qubits[p + 1])
    if barrier:
        circ.barrier()
    for p in range(1, circ.num_qubits - 1, 2):
        circ = rxx(circ, Jx * Δt, circ.qubits[p], circ.qubits[p + 1])

    #  Alt., a naive loop will apply XX rotations sequentially, failing to stack.
    # for p in range(0, circ.num_qubits - 1):
    #     circ = rxx(circ, Jx * Δt, circ.qubits[p], circ.qubits[p + 1])

    # Layer B: on-site Z
    for p in range(circ.num_qubits):
        circ.rz(hz * Δt, circ.qubits[p])

    return circ


def second_order_Trotter_unitary(circ, Δt):
    """
    Applies a Unitary for a single time step U(Δt) using second-order trotter expansion to the input quantum circuit.
    input : circ is a quantum circuit
    output : circ-U(Δt)-
    """
    # Layer 2: on-site Z (half)
    for p in range(circ.num_qubits):
        circ.rz(hz * Δt / 2, circ.qubits[p])

    # Layer 1: coupling XX between n and n+1
    # Induce parallel structure (and symmetry)
    for p in range(1, circ.num_qubits - 1, 2):
        circ = rxx(circ, Jx * Δt / 2, circ.qubits[p], circ.qubits[p + 1])
    # NOTE: Add barriers to separate the parallel section
    circ.barrier()
    for p in range(0, circ.num_qubits - 1, 2):
        circ = rxx(circ, Jx * Δt, circ.qubits[p], circ.qubits[p + 1])
    circ.barrier()
    for p in range(1, circ.num_qubits - 1, 2):
        circ = rxx(circ, Jx * Δt / 2, circ.qubits[p], circ.qubits[p + 1])

    # Layer 2: on-site Z (half)
    for p in range(circ.num_qubits):
        circ.rz(hz * Δt / 2, circ.qubits[p])
    return circ


# qc = qiskit.QuantumCircuit(2)
# qc.h(0)
# qc.cx(0, 1)

qr = qiskit.QuantumRegister(N)
cr = qiskit.ClassicalRegister(N)
qc = create_initial_state(qr, cr)
qc.barrier()
qc = first_order_Trotter_unitary(qc, Δt)

print("Circuit to simulate: ")
print(sim.get_compiled_circuit(qc).draw())

out = sim.simulate_circuit(qc)

# print("Output operator:")
# print(np.round(out.data, 3))
# print("Expected operator:")
# print(np.round(qiskit.quantum_info.Operator(qc).data, 3))
# print("Equivalent:", out.equiv(qiskit.quantum_info.Operator(qc)))

expected = qiskit.quantum_info.Operator(qc)
print("Fidelity: ", qiskit.quantum_info.process_fidelity(out, expected))
