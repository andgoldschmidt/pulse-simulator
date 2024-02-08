import pulse_simulator as ps
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit
from .scheduler import RobustScheduler
from qiskit.transpiler.passes import RemoveBarriers

ONE_QUBIT_GATES = ["sx", "x"]
TWO_QUBIT_GATES = ["ecr"]
VIRTUAL_GATES = ["rz"]

GATE_DICT = dict[int, str] | dict[tuple[int, int], str]
VIRTUAL_ZS = dict[int, float]
SINGLE_MOMENT = tuple[GATE_DICT, VIRTUAL_ZS]
CIRCUIT_MOMENTS = list[tuple[GATE_DICT, VIRTUAL_ZS]]


class MomentBuilder:
    def __init__(self, basis_gates: list[str] = ["rz", "sx", "x", "ecr"]) -> None:
        # set scheduler to not attach virtual gates since we sill extract
        # those from the label of the real gates and treat them accordingly
        self._scheduler = RobustScheduler(
            basis_gates=basis_gates,
            coupling_map=None,
            reattach=False,
            attach_final_virtual=False,
        )

    def build_moments(
        self,
        circuit: QuantumCircuit,
        separate_moments: bool = True,
        no_end_to_end: bool = True,
    ) -> CIRCUIT_MOMENTS:
        n = circuit.num_qubits
        one_q_coloring, two_q_coloring = self._get_coloring(n)

        # scheduled circuits and get output dag
        circuit = RemoveBarriers()(circuit)
        scheduled_dag = self._scheduler.run(
            circuit,
            separate_moments=separate_moments,
            no_end_to_end=no_end_to_end,
            return_dag=True,
        )
        subdags: list[DAGCircuit] = [layer["graph"] for layer in scheduled_dag.layers()]

        # build moments dicts
        moments: CIRCUIT_MOMENTS = []
        for subdag in subdags:
            gates = subdag.op_nodes(include_directives=True)

            gates_dict = {}
            virtual_zs = {}
            for gate in gates:
                if gate.op.name == "barrier":
                    break
                qargs = gate.qargs

                # one-qubit operation
                if len(qargs) == 1:
                    qubit = qargs[0]
                    index = scheduled_dag.find_bit(qubit).index
                    color = one_q_coloring[index]
                    gates_dict[index] = f"{gate.op.name}{color}".upper()

                    if virtual := gate.op.label:
                        virtual_zs.update(self._virtual_str_to_dict(virtual))

                # two-qubit operation
                elif len(qargs) == 2:
                    q0, q1 = qargs[0], qargs[1]
                    i0 = scheduled_dag.find_bit(q0).index
                    i1 = scheduled_dag.find_bit(q1).index
                    color = two_q_coloring[tuple({i0, i1})]
                    gates_dict[(i0, i1)] = f"{gate.op.name}{color}".upper()

                    if virtuals := gate.op.label:
                        for virtual in virtuals.split("&"):
                            virtual_zs.update(self._virtual_str_to_dict(virtual))

            if virtual_zs:
                moments.append(virtual_zs)
            if gates_dict:
                moments.append(gates_dict)

        if final_virtuals := self._scheduler._final_virtuals:
            virtual_zs = {}
            for qubit in final_virtuals:
                virtual_zs.update(self._virtual_str_to_dict(final_virtuals[qubit]))
            moments.append(virtual_zs)

        return moments

    def _virtual_str_to_dict(self, virtual: str) -> dict[int, str]:
        processed = virtual.split("|")
        params = eval(processed[1])
        qargs = eval(processed[2])
        return {qargs._index: params[0]}

    # TODO: generalize for any connectivity?
    def _get_coloring(
        self, n: int
    ) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
        one_q_coloring = {
            qubit: "1" if qubit % 2 == 0 else "2" for qubit in range(n)
        }

        flag = True
        two_q_coloring = {}
        for qubit in range(n - 1):
            if qubit % 2 == 0:
                flag = not flag
            two_q_coloring[tuple({qubit, qubit + 1})] = "1" if flag else "2"

        return one_q_coloring, two_q_coloring
