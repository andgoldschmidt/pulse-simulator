from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library import Barrier
from qiskit.circuit import Qubit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit import ClassicalRegister, transpile

from .expand_virtual import ExpandVirtualGates


class ForbidEndToEnd(TransformationPass):
    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        qr = dag.qregs[list(dag.qregs.keys())[0]]
        if len(dag.cregs.keys()) != 0:
            cr = dag.cregs[list(dag.cregs.keys())[0]]
        else:
            cr = ClassicalRegister(0)

        for layer in dag.layers():
            subdag = layer["graph"]

            new_subdag = DAGCircuit()
            new_subdag.add_qreg(qr)
            new_subdag.add_creg(cr)

            two_q_dag = DAGCircuit()
            two_q_dag.add_qreg(qr)
            two_q_dag.add_creg(cr)

            connections: list[tuple[int, int]] = []
            for gate in subdag.op_nodes(include_directives=True):
                if len(gate.qargs) == 2:
                    connections.append(
                        (
                            dag.find_bit(gate.qargs[0]).index,
                            dag.find_bit(gate.qargs[1]).index,
                        )
                    )
                    two_q_dag.apply_operation_back(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )
                else:
                    new_subdag.apply_operation_back(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )

            if len(connections) != 0:
                invalid_connections = self._check_connections(connections)
                if len(invalid_connections) == 0:
                    new_dag.compose(two_q_dag, qubits=qr, clbits=cr)
                    continue

                expand_virtuals = ExpandVirtualGates()
                expanded_qc = dag_to_circuit(expand_virtuals.run(two_q_dag))

                valid_coupling_map = self._valid_coupling_map(
                    dag.num_qubits(), invalid_connections
                )
                transpiled_qc = transpile(
                    expanded_qc,
                    basis_gates=["rz", "x", "sx", "ecr", "id"],
                    coupling_map=valid_coupling_map,
                    seed_transpiler=12345,  # set seed for testing
                    optimization_level=1,
                )
                transpiled_dag = circuit_to_dag(transpiled_qc)

                # Make sure two-qubit gates are between barriers
                reached_two_qubit = {i: False for i in range(dag.num_qubits())}
                pending_operations = []
                two_qubit_operations = []
                for layer in transpiled_dag.serial_layers():
                    subdag = layer["graph"]
                    for gate in subdag.op_nodes(include_directives=True):
                        if (
                            len(gate.qargs) == 1
                            and reached_two_qubit[gate.qargs[0].index]
                        ):
                            new_subdag.apply_operation_back(
                                gate.op, qargs=gate.qargs, cargs=gate.cargs
                            )
                        elif len(gate.qargs) == 2:
                            two_qubit_operations.append(gate)
                            reached_two_qubit[dag.find_bit(gate.qargs[0]).index] = True
                            reached_two_qubit[dag.find_bit(gate.qargs[1]).index] = True
                        else:
                            pending_operations.append(gate)

                new_subdag.apply_operation_front(Barrier(len(qr)), qargs=qr, cargs=[])
                for gate in two_qubit_operations:
                    new_subdag.apply_operation_front(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )
                new_subdag.apply_operation_front(Barrier(len(qr)), qargs=qr, cargs=[])
                for gate in pending_operations[::-1]:
                    new_subdag.apply_operation_front(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )
            new_dag.compose(new_subdag, qubits=qr, clbits=cr)
        return new_dag

    def _check_connections(
        self, connections: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        previous_target = -2  # -1 would detect (x, 0) as invalid
        invalid_connections: list[tuple[int, int]] = []

        for connection in connections:
            if abs(previous_target - connection[1]) == 1:
                invalid_connections.append(connection)
            previous_target = connection[1]

        return invalid_connections

    def _valid_coupling_map(
        self, n_qubits: int, invalid_connections: list[tuple[int, int]]
    ):
        """
        Generates valid coupling maps
        """
        # TODO: generalize to any connectivity, not just linear

        # remove invalid connections
        edge_list = []
        for i in range(n_qubits - 1):
            if (i, i + 1) not in invalid_connections:
                edge_list.append((i, i + 1))
            if (i + 1, i) not in invalid_connections:
                edge_list.append((i + 1, i))
        return CouplingMap(edge_list)
