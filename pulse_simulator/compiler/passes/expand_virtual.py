from qiskit import ClassicalRegister, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import Operation, Gate
from qiskit.circuit import Qubit
from qiskit.circuit.library import RZGate


class ExpandVirtualGates(TransformationPass):
    def __init__(self, virtual_dict: dict[str, Gate] = {"rz": RZGate}):
        """Deattaches virtual gates from real moments."""
        super().__init__()
        self._virtual_dict = virtual_dict

    def _parse_name_to_instruction(self, virtual: str) -> Operation:
        virtual_dict = self._virtual_dict
        processed = virtual.split("|")
        gate = virtual_dict[processed[0]]
        params = eval(processed[1])
        qargs = eval(processed[2])
        if isinstance(qargs, Qubit):
            qargs = (qargs,)
        return gate(*params), tuple(qargs)

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

            for gate in subdag.op_nodes(include_directives=True):
                if gate.op.label is not None:
                    for gate_string in gate.op.label.split("&"):
                        operation, qargs = self._parse_name_to_instruction(gate_string)
                        new_subdag.apply_operation_back(operation, qargs=qargs)
                    gate.op.label = None
                new_subdag.apply_operation_back(
                    gate.op, qargs=gate.qargs, cargs=gate.cargs
                )
            new_dag.compose(new_subdag, qubits=qr, clbits=cr)

        return new_dag

    def handle_final_virtuals(
        self, dag: DAGCircuit, final_virtuals: dict[Qubit, str]
    ) -> DAGCircuit:
        for qubit in final_virtuals:
            operation, qargs = self._parse_name_to_instruction(final_virtuals[qubit])
            dag.apply_operation_back(operation, qargs=qargs)
        return dag
