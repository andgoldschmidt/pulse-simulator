from typing import Union

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
    MergeAdjacentRzs,
    ForbidEndToEnd,
)


class RobustScheduler:
    """
    A scheduler that takes in a circuit and schedules it for simulation with robust pulses,
    so that we have moments of one- and two-qubit gates.
    """

    def __init__(
        self,
        basis_gates: list[str],
        coupling_map: CouplingMap | None = None,
        reattach: bool = True,
        attach_final_virtual: bool = True,
    ) -> None:
        """
        Instantiate a `RobustScheduler` object.

        Arguments:
            basis_gates: The basis gates for the backend.
            coupling_map: The coupling map for the backend.
            reattach: Whether to reattach virtual gates after scheduling. If the gates
              are not reattached, their information goes to the label of adjacent
              non-virtual gates.
            attach_final_virtual: Whether to attach any pending virtual gates at the end of
              the circuit.
        """
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

    def _transpile(
        self, qc: QuantumCircuit, optimization_level: int = 1
    ) -> QuantumCircuit:
        """Private function to transpile a circuit to target gateset.

        Arguments:
            qc: The circuit to transpile.
            optimization_level: Optimization level to use in Qiskit's transpiler.

        Returns:
            The transpiled circuit.
        """
        basis_gates = self._basis_gates
        coupling_map = self._coupling_map

        # if not given, coupling map is linear
        if coupling_map is None:
            edge_list = []
            for i in range(qc.num_qubits - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])
            coupling_map = CouplingMap(edge_list)
            if qc.num_qubits == 1:
                coupling_map = CouplingMap([[0, 0]])
        transpiled_qc = transpile(
            qc,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            seed_transpiler=12345,  # set seed for testing
            optimization_level=optimization_level,
        )
        return transpiled_qc

    def _schedule(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Private function to run the scheduling passes on a circuit.

        Arguments:
            qc: The circuit to schedule.

        Returns:
            The scheduled circuit.
        """
        return self._pm.run(qc)

    def run(
        self,
        qc: QuantumCircuit,
        separate_moments: bool = True,
        no_end_to_end: bool = True,
        return_dag: bool = False,
        optimization_level: Union[int, None] = 1,
    ) -> QuantumCircuit | DAGCircuit:
        """Run the scheduler on a circuit.

        First tranpsiles the circuit to make sure it uses the target gateset and conforms to the
        desired coupling map. Then, runs the scheduling passes to obtain the desired structure.

        Arguments:
            qc: The circuit to schedule.
            separate_moments: Whether to run separate one- and two-qubit moments.
            no_end_to_end: Whether to get rid of any end-to-end connections between two-qubit gates.
            return_dag: Whether to return the DAG instead of the circuit.
            optimization_level: Optimization level to use in Qiskit's transpiler.

        Returns:
            Scheduled `DAGCircuit` or `QuantumCircuit`.
        """
        if no_end_to_end and not separate_moments:
            raise ValueError(
                "Cannot forbid end-to-end connections between two-qubit gates without "
                "separating one- and two-qubit moments."
            )

        # Initialize all passes
        expand_pass = ExpandVirtualGates()
        rz_pass = MergeAdjacentRzs()
        virtual_pass = AttachVirtualGates()
        separate_pass = SeparateMoments()
        forbid_end_to_end_pass = ForbidEndToEnd()

        if optimization_level is not None:
            transpiled_qc = self._transpile(qc, optimization_level=optimization_level)
        else:
            transpiled_qc = qc

        merged_rzs = rz_pass.run(circuit_to_dag(transpiled_qc))

        # Attach virtual gates and keep track of any virtual gates at the end of the 
        # circuit
        attached_dag = virtual_pass.run(merged_rzs)
        self._final_virtuals = virtual_pass.get_final_virtuals()

        last_qc = dag_to_circuit(attached_dag)
        if separate_moments:
            # Separate circuit into one- and two-qubit moments
            separated_dag = separate_pass.run(attached_dag)
            separated_qc = dag_to_circuit(separated_dag)

            # Slide gates to be executed as soon as possible until no more changes are 
            # being done
            flag = False
            last_qc = separated_qc
            while not flag:
                next_qc = self._schedule(last_qc)
                if str(next_qc.draw()) == str(last_qc.draw()):
                    flag = True
                last_qc = next_qc

        if no_end_to_end:
            # Get rid of any end-to-end connections between two-qubit gates
            full_dag = expand_pass.handle_final_virtuals(
                circuit_to_dag(last_qc), final_virtuals=self._final_virtuals
            )
            no_end_to_end_dag = forbid_end_to_end_pass.run(full_dag)

            # Expand and merge any contiguous Z gates
            expanded_dag = expand_pass.run(no_end_to_end_dag)
            merged_rzs = rz_pass.run(expanded_dag)

            # Reattach newly created Z gates
            last_qc = dag_to_circuit(virtual_pass.run(merged_rzs))
            self._final_virtuals = virtual_pass.get_final_virtuals()

        # Reattach virtual gates only if indicated
        if not self._reattach:
            final_dag = circuit_to_dag(last_qc)
        else:
            # Make virtual gates "real" again and attach any pending at the end of the 
            # circuit
            final_dag = expand_pass.run(circuit_to_dag(last_qc))

        if self._final_virtuals and self._attach_final_virtual:
            final_dag = expand_pass.handle_final_virtuals(
                final_dag, final_virtuals=self._final_virtuals
            )
            self._final_virtuals = None

        if return_dag:
            return final_dag
        return dag_to_circuit(final_dag)
