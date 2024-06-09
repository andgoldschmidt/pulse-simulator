import pulse_simulator as ps
import functools
import qiskit_dynamics as qk_d
import qiskit.providers.fake_provider as qk_fp
import numpy as np
import csv
import qiskit

if __name__ == "__main__":
    backend = qk_fp.FakeManila()
    units = 1e9
    ns = units
    dt = backend.configuration().dt * ns
    print("dt: ", dt)

    N = 4 # number of spins

    registers = [i for i in range(N)]
    config_vars = ps.backend_simulation_vars(backend, rabi=False, units=units)

    cr_model = functools.partial(
        ps.cross_resonance_model,
        registers=registers,
        backend=backend,
        variables=config_vars,
        model_name="Toy",
        qiskit_order=True,
    )

    # Control model
    H_drift = 0.0
    Hs_control = []
    Hs_channels = []
    for i in range(N - 1):
        # control-target
        Hj_drift, Hjs_control, Hjs_channel, params = cr_model(
            (i, i + 1), return_params=True
        )
        H_drift += Hj_drift
        for c_i in range(len(Hjs_channel)):
            if Hjs_channel[c_i] in Hs_channels:
                continue
            else:
                # print(Hjs_channel[c_i])
                Hs_control += [Hjs_control[c_i]]
                Hs_channels += [Hjs_channel[c_i]]
        # print(params)

        # target-control
        Hj_drift, Hjs_control, Hjs_channel, params = cr_model(
            (i + 1, i), return_params=True
        )
        H_drift += Hj_drift
        for c_i in range(len(Hjs_channel)):
            if Hjs_channel[c_i] in Hs_channels:
                continue
            else:
                # print(Hjs_channel[c_i])
                Hs_control += [Hjs_control[c_i]]
                Hs_channels += [Hjs_channel[c_i]]

    # Cross talk term: ζ ZZ
    # H_xtalk = ps.crosstalk_model(registers, ps.backend_edges(backend), config_vars)
    ζ = 0.001
    Z = np.array([[1, 0], [0, -1]])
    H_xtalk = Z
    for _ in range(N - 1):
        H_xtalk = np.kron(H_xtalk, Z)
    H_xtalk = ζ * qiskit.quantum_info.Operator(H_xtalk)

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
    solver.model.evaluation_mode = "dense"

    sim = ps.simulator.Simulator(
        basis_gates=["rz", "sx", "x", "ecr"], solver=solver, backend=backend
    )

    # load and set one-qubit pulses
    file_name = "./pico-pulses/saved-pulses-2023-12-13/a_single_qubit_gateset_R1e-5.csv"
    gates = []
    with open(file_name) as file:
        reader = csv.reader(file)
        for row in reader:
            gates.append(np.array([float(x) for x in row]))

    for pulse, name in zip(gates, ["x_blue", "x_red", "sx_blue", "sx_red"]):
        qiskit_pulse = qiskit.pulse.Waveform(pulse, limit_amplitude=False)
        sim.set_pulse(name, qiskit_pulse)

    # load and set two-qubit pulses
    file_name = "./pico-pulses/saved-pulses-2024-01-16/a_two_qubit_robust.csv"
    gates = []
    with open(file_name) as file:
        reader = csv.reader(file)
        for row in reader:
            gates.append(np.array([float(x) for x in row]))

    rzx_p = qiskit.pulse.Waveform(gates[0], limit_amplitude=False)
    rzx_n = qiskit.pulse.Waveform(-gates[0], limit_amplitude=False)
    sim.set_pulse("zx+_blue", rzx_p)
    sim.set_pulse("zx+_red", rzx_p)
    sim.set_pulse("zx-_blue", rzx_n)
    sim.set_pulse("zx-_red", rzx_n)

    qc = qiskit.QuantumCircuit(N)   
    for _ in range(5):
        qc.h(0)
        qc.cx(0, 1)

    expected = qiskit.quantum_info.Operator(qc)

    print("\nCircuit to simulate: ")
    print(sim.get_compiled_circuit(qc).draw())

    out = sim.simulate_circuit(qc)

    print("Output operator:")
    print(np.round(out.data, 3), "\n")
    print("Expected operator:")
    print(np.round(expected.data, 3), "\n")

    print("\nFidelity: ", qiskit.quantum_info.process_fidelity(out, expected))
    # print("Equivalent: ", expected.equiv(out))
