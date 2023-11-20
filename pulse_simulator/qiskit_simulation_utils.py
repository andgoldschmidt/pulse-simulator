import functools
import numpy as np
from qiskit import QuantumCircuit, quantum_info


def rz_moment(virtual_zs, registers):
    """Find the unitary matrix of a moment of Z rotations.

    Each entry of `registers` should have an entry in `virtual_zs`.

    Arguments:
        virtual_zs (Dict{Int: Float}) -- R_z qubit and angle dictionary.
        registers (List[Int]) -- Active registers.

    Returns:
        _description_
    """
    circuit = QuantumCircuit(len(registers))
    # Registers order the Hilbert space
    for i, r in enumerate(registers):
        circuit.rz(virtual_zs[r], i)
    return quantum_info.Operator(circuit)


def qiskit_ground_state(n_qubits):
    return quantum_info.states.Statevector(
        functools.reduce(np.kron, np.repeat([[1, 0]], n_qubits, axis=0))
    )


def qiskit_identity_operator(n_qubits):
    id_label = "".join(["I"] * n_qubits)
    return quantum_info.Operator.from_label(id_label)
