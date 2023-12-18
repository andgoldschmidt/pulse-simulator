from qiskit import QuantumCircuit, transpile
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit

from .passes import (
    SlideOneQubitOps,
    SeparateMoments,
    DeleteConsecutiveBarriers,
    SlideTwoQubitOps,
    AttachVirtualGates,
    ExpandVirtualGates,
)


class RobustScheduler:
    def __init__(
        self,
        basis_gates: list[str],
        coupling_map: CouplingMap | None = None,
        reattach: bool = True,
        attach_final_virtual: bool = True,
    ):
        pm = PassManager(
            [
                DeleteConsecutiveBarriers(),
                SlideOneQubitOps(),
                DeleteConsecutiveBarriers(),
                SlideTwoQubitOps(),
                DeleteConsecutiveBarriers(),
            ]
        )

        self._basis_gates = basis_gates
        self._coupling_map = coupling_map
        self._reattach = reattach
        self._attach_final_virtual = attach_final_virtual
        self._pm = pm

    def _transpile(self, qc: QuantumCircuit) -> QuantumCircuit:
        basis_gates = self._basis_gates
        coupling_map = self._coupling_map

        # if not given, coupling map is linear
        if coupling_map is None:
            coupling_map = CouplingMap([[i, i + 1] for i in range(qc.num_qubits - 1)])
            if qc.num_qubits == 1:
                coupling_map = CouplingMap([[0, 0]])
        return transpile(
            qc,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            seed_transpiler=12345,  # set seed for testing
        )

    def _schedule(self, qc: QuantumCircuit) -> QuantumCircuit:
        return self._pm.run(qc)

    def run(
        self, qc: QuantumCircuit, return_dag: bool = False
    ) -> QuantumCircuit | DAGCircuit:
        transpiled_qc = self._transpile(qc)

        # Attach virtual gates and keep track of any virtual gates at the end of the circuit
        virtual_pass = AttachVirtualGates()
        attached_dag = virtual_pass.run(circuit_to_dag(transpiled_qc))
        self._final_virtuals = virtual_pass.get_final_virtuals()

        # Separate circuit into one- and two-qubit moments
        separated_dag = SeparateMoments().run(attached_dag)
        separated_qc = dag_to_circuit(separated_dag)

        # Slide gates to be executed as soon as possible until no more changes are being done
        flag = False
        last_qc = separated_qc
        while not flag:
            next_qc = self._schedule(last_qc)
            if str(next_qc.draw()) == str(last_qc.draw()):
                flag = True
            last_qc = next_qc

        expand_pass = ExpandVirtualGates()
        # Reattach virtual gates only if indicated
        if not self._reattach:
            final_dag = circuit_to_dag(last_qc)
        else:
            # Make virtual gates "real" again and attach any pending at the end of the circuit
            final_dag = expand_pass.run(circuit_to_dag(last_qc))

        if self._final_virtuals and self._attach_final_virtual:
            final_dag = expand_pass.handle_final_virtuals(
                final_dag, final_virtuals=self._final_virtuals
            )
            self._final_virtuals = None

        if return_dag:
            return final_dag
        return dag_to_circuit(final_dag)
