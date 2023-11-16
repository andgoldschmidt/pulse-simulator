from .plot_utils import from_label, to_label

import numpy as np


def get_edges(backend, directed=False):
    """Get the unique edges of the provided backend based on allowed two-qubit
    gates.

    Arguments:
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend

    Returns:
        List of unique edges (as tuples)
    """
    dir_graph = backend.configuration().coupling_map
    if not directed:
        dir_graph = np.unique([set(edge) for edge in dir_graph])
    return [tuple(edge) for edge in dir_graph]


def crosstalk_model(circuit, crosstalk_graph):
    """Get the crosstalk operator for the circuit based on the provided
    crosstalk graph.

    Arguments:
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) -- Circuit
        crosstalk_graph (Dict{Tuple(Int, Int): float}) -- Edge keys, crosstalk
            values. The keys are the qubit indices for the circuit.

    Returns:
        The ZZ crosstalk operator for the circuit.
    """
    crosstalk_op = 0.
    for edge, zz_val in crosstalk_graph.items():
        label = ""
        for i in range(circuit.num_qubits):
            label += "Z" if i in edge else "I"
        crosstalk_op += 2 * np.pi * zz_val * from_label(label) / 4
    return crosstalk_op


def cross_resonance_model(qubits, circuit, swpt=False, **kwargs):
    """Construct a two qubit model for pulse CR gates.
    TODO: Rabi rates?

    Arguments:
        qubits (List[qiskit.circuit.quantumregister.Qubit]) -- The two qubits,
            ordered as (control, target).
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) -- Circuit in
            which the qubits resides.
        swpt (bool) -- Whether to use the effective SWPT CR model. Default
            false.

    Keyword arguments:
        freq_c (float) -- Energy of control qubit.
        freq_t (float) -- Energy of target qubit.
        alpha (float) -- Anharmonicty of all qubits
        coupling (float) -- Energy of lab frame control-target coupling
        rabi_freq_c (float) -- Energy of control qubit drive coupling
        rabi_freq_t (float) -- Energy of target qubit drive coupling

    Returns:
        Drift operator, List[Control operators], List[Drive channels]
    """
    # Unpack parameters
    wc = kwargs.pop("freq_c")
    wt = kwargs.pop("freq_t")
    delta_ct = wc - wt
    J = kwargs.pop("coupling")
    alpha = kwargs.pop("alpha")  # Global
    r_c = kwargs.pop("rabi_freq_c")
    r_t = kwargs.pop("rabi_freq_t")

    # Unpack qubits
    control, target = qubits
    index_c = circuit.find_bit(control).index
    index_t = circuit.find_bit(target).index

    # Construct operator labels
    ZZ_label = to_label({index_c: "Z", index_t: "Z"}, circuit.num_qubits)
    ZI_label = to_label({index_c: "Z", index_t: "I"}, circuit.num_qubits)
    XI_label = to_label({index_c: "X", index_t: "I"}, circuit.num_qubits)
    IX_label = to_label({index_c: "I", index_t: "X"}, circuit.num_qubits)
    ZX_label = to_label({index_c: "Z", index_t: "X"}, circuit.num_qubits)
    
    # Construct Hamiltonian operators
    ZZ = from_label(ZZ_label)
    ZI = from_label(ZZ_label)
    XI = from_label(XI_label)
    IX = from_label(IX_label)
    ZX = from_label(ZX_label)
    J_p = J / (delta_ct + alpha)
    J_m = J / (delta_ct - alpha)
    if swpt:
        drift_op = 0.  # -J * (J_p - J_m) * ZZ
        control_drive_op = 2 * np.pi * r_c * J_p * (IX + (alpha / delta_ct) * ZX) / 2
        target_drive_op = 2 * np.pi * r_t * IX / 2
        control_ops = [control_drive_op, target_drive_op]
        # Control drive uses channel u#, target drive uses normal channel d#.
        control_channels = [f"u{index_c}", f"d{index_t}"]
    else:
        drift_op = 2 * np.pi * delta_ct / 2 * ZI
        control_drive_op = 2 * np.pi * r_c * (XI + J / delta_ct * ZX) / 2
        target_drive_op = 2 * np.pi * r_t * IX / 2
        control_ops = [control_drive_op, target_drive_op]
        # Control drive uses channel u#, target drive uses normal channel d#.
        control_channels = [f"u{index_c}", f"d{index_t}"]

    return drift_op, control_ops, control_channels
