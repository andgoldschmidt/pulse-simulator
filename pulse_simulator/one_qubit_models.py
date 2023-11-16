from .plot_utils import from_label, to_label

import numpy as np


def qubit_decay_model(qubit, circuit, **kwargs):
    """Construct a single qubit error model for pulse gates from t1 and t2
    times. The functionality is similar to the behavior of QuTip pulse.

    Arguments:
        qubit (qiskit.circuit.quantumregister.Qubit) -- Qubit to model
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) -- Circuit in
            which qubit resides.

    Keyword arguments:
        t1 [optional] (float) -- Damping decay constant.
        t2 [optional] (float) -- Dephasing decay constant.

    Raises:
        ValueError: Inconsistent t1 and t2.

    Returns:
        List[Static jump operators]
    """
    # Unpack parameters
    t1 = kwargs.pop("t1", None)
    t2 = kwargs.pop("t2", None)

    qubit_index = circuit.find_bit(qubit).index

    # Construct operator labels
    # Destroy D is an augmented label
    damp_label = to_label({qubit_index: "D"}, circuit.num_qubits)
    dephase_label = to_label({qubit_index: "1"}, circuit.num_qubits)

    # Construct dissipation operators (QuTip pulse)
    static_op = []
    if t1 is not None:
        static_op += [1 / np.sqrt(t1) * from_label(damp_label)]
    if t2 is not None:
        # Keep the total dephasing ~ exp(-t/t2)
        if t1 is not None:
            if 2 * t1 < t2:
                raise ValueError(f"t1={t1}, t2={t2} does not fulfill 2*t1>t2")
            T2_eff = 1 / (1 / t2 - 1 / 2 / t1)
        else:
            T2_eff = t2
        static_op += [1 / np.sqrt(2 * T2_eff) * 2 * from_label(dephase_label)]

    return static_op


def rx_model(qubit, circuit, **kwargs):
    """Construct a single qubit model for pulse gates.

    Arguments:
        qubit (qiskit.circuit.quantumregister.Qubit) -- Qubit to model
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) -- Circuit in
            which qubit resides.

    Keyword arguments:
        freq (float) -- Energy of qubit.
        rabi_freq (float) -- Energy of drive coupling.

    Raises:
        ValueError: Inconsistent t1 and t2.

    Returns:
        Drift operator, List[Control operators], List[Drive channels]
    """
    # Unpack parameters
    w = kwargs.pop("freq")
    r = kwargs.pop("rabi_freq")

    qubit_index = circuit.find_bit(qubit).index

    # Construct operator labels
    drift_label = to_label({qubit_index: "Z"}, circuit.num_qubits)
    control_label = to_label({qubit_index: "X"}, circuit.num_qubits)

    # Construct Hamiltonian operators
    drift_op = 2 * np.pi * w * from_label(drift_label) / 2
    control_op = 2 * np.pi * r * from_label(control_label) / 2

    return drift_op, [control_op], [f"d{qubit_index}"]
