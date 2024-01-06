from qiskit import ClassicalRegister, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import Operation, Gate
from qiskit.circuit import Qubit
from qiskit.circuit.library import RZGate


class MergeAdjacentRzs(TransformationPass):
    def __init__(self):
        """Merges adjacent Rz gates together."""
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

        cached_rzs: dict[Qubit, float] = {}
        for layer in dag.layers():
            subdag = layer["graph"]

            new_subdag = DAGCircuit()
            new_subdag.add_qreg(qr)
            new_subdag.add_creg(cr)

            for gate in subdag.op_nodes(include_directives=True):
                if gate.op.name == "barrier":
                    continue
                elif gate.op.name == "rz":
                    if gate.qargs[0] in cached_rzs:
                        cached_rzs[gate.qargs[0]] += gate.op.params[0]
                    else:
                        cached_rzs[gate.qargs[0]] = gate.op.params[0]
                else:
                    for q in gate.qargs:
                        if q in cached_rzs:
                            new_subdag.apply_operation_back(
                                RZGate(cached_rzs[q]), qargs=(q,)
                            )
                            cached_rzs.pop(q)
                    new_subdag.apply_operation_back(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )
            new_dag.compose(new_subdag, qubits=qr, clbits=cr)

        for q in cached_rzs:
            new_dag.apply_operation_back(RZGate(cached_rzs[q]), qargs=(q,))

        return new_dag
