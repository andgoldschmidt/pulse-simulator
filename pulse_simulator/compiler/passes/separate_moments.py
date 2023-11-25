from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import Barrier
from qiskit.transpiler.basepasses import TransformationPass
from qiskit import ClassicalRegister


class SeparateMoments(TransformationPass):
    def __init__(self):
        """Separates one- and two-qubit moments in a circuit."""
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
            for gate in subdag.op_nodes(include_directives=True):
                if len(gate.qargs) == 2:
                    new_subdag.apply_operation_back(
                        Barrier(len(qr)), qargs=qr, cargs=[]
                    )
                    new_subdag.apply_operation_back(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )
                    new_subdag.apply_operation_back(
                        Barrier(len(qr)), qargs=qr, cargs=[]
                    )
                else:
                    new_subdag.apply_operation_back(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )

            new_dag.compose(new_subdag, qubits=qr, clbits=cr)

        return new_dag
