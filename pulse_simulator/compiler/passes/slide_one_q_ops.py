from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import Barrier
from qiskit.transpiler.basepasses import TransformationPass
from qiskit import ClassicalRegister


class SlideOneQubitOps(TransformationPass):
    def __init__(self):
        """Sends one-qubit operations to as soon as possible without while leaving two-qubit layers alone."""
        super().__init__()

    def _separate_moments(self, dag: DAGCircuit) -> list[tuple[list[DAGCircuit], str]]:
        moments: list[tuple[list[DAGCircuit], str]] = []
        subdags = [layer["graph"] for layer in dag.layers()]

        current_moment: list[DAGCircuit] = []
        previous_type = None

        for subdag in subdags:
            gate = subdag.op_nodes(include_directives=True)[0]

            if gate.op.name == "barrier":
                moments.append((current_moment, previous_type))
                current_moment = []
                continue

            current_moment.append(subdag)
            previous_type = "two" if len(gate.qargs) == 2 else "one"

        if len(current_moment) != 0:
            moments.append((current_moment, previous_type))

        return moments

    def _dag_from_moments(
        self, dag: DAGCircuit, moments: list[DAGCircuit]
    ) -> DAGCircuit:
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

        for moment in moments:
            if sum([layer.size() for layer in moment[0]]) == 0:
                continue
            elif moment[1] == "two":
                new_dag.apply_operation_back(Barrier(len(qr)), qargs=qr, cargs=[])
                for layer in moment[0]:
                    new_dag.compose(layer, qubits=qr, clbits=cr)
                new_dag.apply_operation_back(Barrier(len(qr)), qargs=qr, cargs=[])
            else:
                for layer in moment[0]:
                    new_dag.compose(layer, qubits=qr, clbits=cr)

        return new_dag

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        moments = self._separate_moments(dag)

        # ensure we start with a one-qubit moment
        current_index = 0
        if moments[0][1] == "two":
            current_index += 1

        last_one_q_moment = None
        used_qargs = set()
        for i in range(current_index, len(moments), 2):
            # grab one- and two-qubit moments
            one_q_moment = moments[i][0]
            if i + 1 >= len(moments):
                two_q_moment = [DAGCircuit()]
            else:
                two_q_moment = moments[i + 1][0]

            # slide back to available wires
            if i > 1:
                removed = []
                for gate in one_q_moment[0].op_nodes():
                    if all([q not in used_qargs for q in gate.qargs]):
                        last_one_q_moment.apply_operation_back(
                            gate.op, qargs=gate.qargs, cargs=gate.cargs
                        )
                        removed.append(gate)
                for gate in removed:
                    one_q_moment[0].remove_op_node(gate)

            # keep track of used wires
            used_qargs = set()
            for layer in two_q_moment:
                for gate in layer.op_nodes():
                    for q in gate.qargs:
                        used_qargs.add(q)
            if len(one_q_moment) == 0:
                continue
            for gate in one_q_moment[-1].op_nodes():
                for q in gate.qargs:
                    used_qargs.add(q)
            last_one_q_moment = one_q_moment[-1]

        new_dag = self._dag_from_moments(dag, moments)
        return new_dag
