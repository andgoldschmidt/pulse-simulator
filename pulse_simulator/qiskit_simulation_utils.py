import functools
import numpy as np
from qiskit import QuantumCircuit, quantum_info
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict
from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.transpiler import PassManager
from typing import Union


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
    a: Union[np.ndarray, quantum_info.Operator],
    b: Union[np.ndarray, quantum_info.Operator]
) -> float:
    if isinstance(a, quantum_info.Operator):
        a = a.data
    if isinstance(b, quantum_info.Operator):
        b = b.data
    a2 = np.trace(a.conj().T @ a)
    b2 = np.trace(b.conj().T @ b)
    a_b = np.trace(a.conj().T @ b)
    return np.abs(a_b / np.sqrt(a2 * b2))**2


def remove_idle_wires(circ):
    """
    Remove inactive qubits from a circuit.
    https://quantumcomputing.stackexchange.com/questions/25672/
    """
    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    dag.qregs = OrderedDict()

    return dag_to_circuit(dag)

def remove_final_measurement(circ):
    # Create a pass manager and add the RemoveFinalMeasurements pass
    pass_manager = PassManager()
    pass_manager.append(RemoveFinalMeasurements())

    # Apply the pass manager to the circuit
    return pass_manager.run(circ)
