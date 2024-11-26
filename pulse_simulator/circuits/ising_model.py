import numpy as np
import qiskit as qk
from qiskit.circuit import QuantumCircuit
from typing import Union


class TransverseIsingModel:
    def __init__(
        self,
        quantum_registers: int = 4,
        classical_registers: list[int] = [],
        h: float = 2 * np.pi,
        J: float = 2 * np.pi,
        Δt: float = 0.05,
    ):
        """
        Initialize the transverse Ising model.

        Args:
            num_qubit_registers (int): number of qubits in the register
            num_classical_registers (int): number of classical registers
            h (float): magnetic field along z or x
            J (float): Coupling along xx or zz
            Δt (float): time step for integration
        """
        self._N = len(quantum_registers)
        self._C = len(classical_registers)
        self._h = h
        self._J = J
        self._Δt = Δt

    def create_initial_state(self, qr: int, cr: int):
        """
        Creates the initial state for the experiment.
        Here we would like to prepare the |+>|+>|+>|+> state.
        """
        circ = qk.QuantumCircuit(qr, cr)
        [circ.h(i) for i in range(qr)]
        return circ

    def rxx(self, circ: QuantumCircuit, θx, q1: int, q2: int):
        """
        Implements the exp(-i theta/2 sx_1 sx_2) gate.
        """
        circ.cx(q1, q2)
        circ.rx(θx, q1)
        circ.cx(q1, q2)
        return circ

    def rzz(self, circ: QuantumCircuit, θz, q1: int, q2: int):
        """
        Implements the exp(-i theta/2 sz_1 sz_2) gate.
        """
        circ.cx(q1, q2)
        circ.rz(θz, q1)
        circ.cx(q1, q2)
        return circ

    def first_order_Trotter_unitary(
        self, circ: Union[QuantumCircuit, None] = None, barrier=False, coupling="xx"
    ):
        """
        Applies a Unitary for a single time step U(Δt) using first-order trotter
            expansion to the input quantum circuit.

        input : circ is a quantum circuit
        output : circ-U(Δt)-
        """
        # Initial circuit
        if circ is None:
            circ = self.create_initial_state(self._N, self._C)

        # Layer A: coupling XX between n and n+1
        # Induce parallel CR gates
        if barrier:
            circ.barrier()
        for p in range(0, circ.num_qubits - 1, 2):
            if coupling == "xx":
                circ = self.rxx(
                    circ, self._J * self._Δt, circ.qubits[p], circ.qubits[p + 1]
                )
            elif coupling == "zz":
                circ = self.rzz(
                    circ, self._J * self._Δt, circ.qubits[p], circ.qubits[p + 1]
                )
            else:
                raise ValueError("Invalid coupling")
        if barrier:
            circ.barrier()
        for p in range(1, circ.num_qubits - 1, 2):
            if coupling == "xx":
                circ = self.rxx(
                    circ, self._J * self._Δt, circ.qubits[p], circ.qubits[p + 1]
                )
            elif coupling == "zz":
                circ = self.rzz(
                    circ, self._J * self._Δt, circ.qubits[p], circ.qubits[p + 1]
                )
            else:
                raise ValueError("Invalid coupling")

        # Layer B: on-site Z
        for p in range(circ.num_qubits):
            if coupling == "xx":
                circ.rz(self._h * self._Δt, circ.qubits[p])
            elif coupling == "zz":
                circ.rx(self._h * self._Δt, circ.qubits[p])
            else:
                raise ValueError("Invalid coupling")

        return circ

    def second_order_Trotter_unitary(
        self, circ: Union[QuantumCircuit, None] = None, barrier=False
    ):
        """
        TODO: Add coupling toggle

        Applies a Unitary for a single time step U(Δt) using the second-order trotter
            expansion to the input quantum circuit.

        input : circ is a quantum circuit
        output : circ-U(Δt)-
        """
        # Initial circuit
        if circ is None:
            circ = self.create_initial_state(self._N, self._C)

        # Layer 2: on-site Z (half)
        for p in range(circ.num_qubits):
            circ.rz(self._h * self._Δt / 2, circ.qubits[p])

        # Layer 1: coupling XX between n and n+1
        # Induce parallel structure (and symmetry)
        for p in range(1, circ.num_qubits - 1, 2):
            circ = self.rxx(
                circ, self._J * self._Δt / 2, circ.qubits[p], circ.qubits[p + 1]
            )
        # Add barriers to separate the parallel section
        if barrier:
            circ.barrier()
        for p in range(0, circ.num_qubits - 1, 2):
            circ = self.rxx(
                circ, self._J * self._Δt, circ.qubits[p], circ.qubits[p + 1]
            )
        if barrier:
            circ.barrier()
        for p in range(1, circ.num_qubits - 1, 2):
            circ = self.rxx(
                circ, self._J * self._Δt / 2, circ.qubits[p], circ.qubits[p + 1]
            )

        # Layer 2: on-site Z (half)
        for p in range(circ.num_qubits):
            circ.rz(self._h * self._Δt / 2, circ.qubits[p])
        return circ
