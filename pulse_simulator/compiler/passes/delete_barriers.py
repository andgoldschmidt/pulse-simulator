from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit import ClassicalRegister


class DeleteConsecutiveBarriers(TransformationPass):
    def __init__(self):
        """Deletes any consecutive barriers in a circuit."""
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

        subdags = [layer["graph"] for layer in dag.serial_layers()]
        previous_gate_name = ""
        remove = []

        for i in range(len(subdags)):
            subdag = subdags[i]

            for gate in subdag.op_nodes(include_directives=True):
                if gate.op.name == "barrier" and previous_gate_name == "barrier":
                    remove += [i, i - 1]
                previous_gate_name = gate.op.name

        for i in range(len(subdags)):
            if i in remove:
                continue
            new_dag.compose(subdags[i], qubits=qr, clbits=cr)

        return new_dag
