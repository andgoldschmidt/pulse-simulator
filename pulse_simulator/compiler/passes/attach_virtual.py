from qiskit import ClassicalRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class AttachVirtualGates(TransformationPass):
    def __init__(self, virtual_names: set[str] = ("rz",)):
        """Attaches virtual gates to real moments."""
        super().__init__()
        self._virtual_names = virtual_names
        self._final_virtual = dict()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        virtual_names = self._virtual_names
        ignore = {"barrier", "measurement"}

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

        pending_virtual = {}
        for layer in dag.layers():
            subdag = layer["graph"]

            new_subdag = DAGCircuit()
            new_subdag.add_qreg(qr)
            new_subdag.add_creg(cr)

            for gate in subdag.op_nodes(include_directives=True):
                if gate.name not in ignore:
                    if gate.name in virtual_names:
                        # assumes virtual gates are one-qubit
                        pending_virtual[
                            gate.qargs[0]
                        ] = f"{gate.name}|{gate.op.params}|{gate.qargs[0]}"
                        continue

                    for q in gate.qargs:
                        if q in pending_virtual.keys():
                            gate.op = gate.op.to_mutable()
                            if gate.op.label:
                                gate.op.label = gate.op.label + "&" + pending_virtual[q]
                            else:
                                gate.op.label = pending_virtual[q]
                            del pending_virtual[q]

                new_subdag.apply_operation_back(
                    gate.op, qargs=gate.qargs, cargs=gate.cargs
                )
            new_dag.compose(new_subdag, qubits=qr, clbits=cr)

        if pending_virtual:
            self._final_virtual = pending_virtual
        return new_dag

    def get_final_virtuals(self) -> DAGCircuit | None:
        if self._final_virtual:
            return self._final_virtual
        return None
