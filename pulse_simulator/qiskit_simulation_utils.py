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


def qiskit_ground_state(n_qubits, open=False):
    psi = functools.reduce(np.kron, np.repeat([[1, 0]], n_qubits, axis=0))
    if open:
        return quantum_info.states.DensityMatrix(psi)
    else:
        return quantum_info.states.Statevector(psi)


def qiskit_identity_operator(n_qubits, open=False):
    id_label = "".join(["I"] * n_qubits)
    op = quantum_info.Operator.from_label(id_label)
    if open:
        return quantum_info.SuperOp(op)
    else:
        return op


def operator_fidelity(
    a: quantum_info.Operator,
    b: quantum_info.Operator
) -> float:
    a2 = np.trace(a.data.conj().T @ a.data)
    b2 = np.trace(b.data.conj().T @ b.data)
    a_b = np.trace(a.data.conj().T @ b.data)
    return np.abs(a_b / np.sqrt(a2 * b2))**2
