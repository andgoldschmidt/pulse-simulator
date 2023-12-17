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

registers = [0, 1]
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


solver = qk_d.Solver(
    static_hamiltonian=None,
    hamiltonian_operators=Hs_control,
    static_dissipators=None,
    rotating_frame=None,
    rwa_cutoff_freq=None,
    hamiltonian_channels=Hs_channels,
    channel_carrier_freqs={ch: 0.0 for ch in Hs_channels},
    dt=dt,
)

sim = ps.simulator.Simulator(basis_gates=["rz", "sx", "x", "cx"], solver=solver, backend=backend)

# load and set pulses
file_name = "./pico-pulses/saved-pulses-2023-12-13/a_single_qubit_gateset_R1e-3.csv"
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

qc = qiskit.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
out = sim.simulate_circuit(qc)

print("Output density matrix:")
print(np.round(out.data, 3))
print("Expected density matrix:")
print(np.round(qiskit.quantum_info.DensityMatrix(qc).data, 3))