import re
import numpy as np

from qiskit import QiskitError
from qiskit.circuit.library import standard_gates
from qiskit.quantum_info.operators import Operator


def zero_operator(num_qubits):
    return np.zeros((2**num_qubits, 2**num_qubits))


def from_label(label):
    """Return a tensor product of single-qubit operators.

    Args:
        label (Str) -- single-qubit operator string.

    Returns:
        Operator -- The N-qubit operator.

    Raises:
        QiskitError -- if the label contains invalid characters, or the
                        length of the label is larger than an explicitly
                        specified num_qubits.

    Additional Information:
        The labels correspond to the single-qubit matrices:
        'I': [[1, 0], [0, 1]]
        'X': [[0, 1], [1, 0]]
        'Y': [[0, -1j], [1j, 0]]
        'Z': [[1, 0], [0, -1]]
        'H': [[1, 1], [1, -1]] / sqrt(2)
        'S': [[1, 0], [0 , 1j]]
        'T': [[1, 0], [0, (1+1j) / sqrt(2)]]
        '0': [[1, 0], [0, 0]]
        '1': [[0, 0], [0, 1]]
        '+': [[0.5, 0.5], [0.5 , 0.5]]
        '-': [[0.5, -0.5], [-0.5 , 0.5]]
        'r': [[0.5, -0.5j], [0.5j , 0.5]]
        'l': [[0.5, 0.5j], [-0.5j , 0.5]]
        'D': [[0, 1], [0 , 0]]
        'C': [[0, 0], [1 , 0]]
    """
    # Check label is valid
    label_mats = {
        "I": standard_gates.IGate().to_matrix(),
        "X": standard_gates.XGate().to_matrix(),
        "Y": standard_gates.YGate().to_matrix(),
        "Z": standard_gates.ZGate().to_matrix(),
        "H": standard_gates.HGate().to_matrix(),
        "S": standard_gates.SGate().to_matrix(),
        "T": standard_gates.TGate().to_matrix(),
        "0": np.array([[1, 0], [0, 0]], dtype=complex),
        "1": np.array([[0, 0], [0, 1]], dtype=complex),
        "+": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
        "-": np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex),
        "r": np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex),
        "l": np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex),
        "D": np.array([[0, 1], [0, 0]], dtype=complex),
        "C": np.array([[0, 0], [1, 0]], dtype=complex),
    }
    if label == "":
        raise QiskitError("Label is empty.")
    if re.match(r"^[IXYZHST01rlDC\-+]+$", label) is None:
        raise QiskitError("Label contains invalid characters.")
    # Initialize an identity matrix and apply each gate
    num_qubits = len(label)
    op = Operator(np.eye(2**num_qubits, dtype=complex))
    for qubit, char in enumerate(reversed(label)):
        if char != "I":
            op = op.compose(label_mats[char], qargs=[qubit])
    return op


def to_label(index_char_dict, registers):
    """Make an operator string where the character is placed at the index for
    each entry of the dictionary, or an "I" is added.

    Arguments:
        index_char_dict (Dict{Int, Char}) -- a dictionary of
            {Integer: Character} entries.
        registers (List[Int]) -- the entries of Hilbert space in the string

    Returns:
        The operator label of the specified length.
    """
    label = ""
    for r in registers:
        if r in index_char_dict.keys():
            label += index_char_dict[r]
        else:
            label += "I"
    return label
