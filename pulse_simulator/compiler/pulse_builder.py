from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.pulse import SymbolicPulse

from .scheduler import RobustScheduler


class PulseBuilder:
    def __init__(
        self,
        basis_gates: list[str],
        coupling_map: CouplingMap | None,
        one_q_pulses: dict[int, SymbolicPulse],
        control_pulses: dict[int, SymbolicPulse],
        target_pulses: dict[int, SymbolicPulse],
    ):
        self._basis_gates = basis_gates
        self._coupling_map = coupling_map
        self._one_q_pulses = one_q_pulses
        self._control_pulses = control_pulses
        self._target_pulses = target_pulses

        # set the scheduler to not attach virtual gates since we will extract
        # those from the label of the real gates and treat them accordingly
        self._scheduler = RobustScheduler(
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            reattach=False,
            attach_final_virtual=False,
        )

    def build(self, circuit: QuantumCircuit):
        moments = self._build_moments_dicts(circuit)

        # TODO: actually build pulses

        return moments

    def _build_moments_dicts(
        self, circuit: QuantumCircuit
    ) -> list[tuple[dict[int, str], dict[int, float]]]:
        n = circuit.num_qubits
        one_q_coloring, two_q_coloring = self._get_coloring(n)

        # scheduled circuits and get output dag
        circuit = RemoveBarriers()(circuit)
        scheduled_dag = self._scheduler.run(circuit, return_dag=True)
        subdags: list[DAGCircuit] = [layer["graph"] for layer in scheduled_dag.layers()]

        # build moment dicts
        moments = []
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
                    gates_dict[index] = f"{gate.op.name}_{color}"

                    if virtual := gate.op.label:
                        virtual_zs.update(self._virtual_str_to_dict(virtual))

                # two-qubit operation
                elif len(qargs) == 2:
                    q0, q1 = qargs[0], qargs[1]
                    i0 = scheduled_dag.find_bit(q0).index
                    i1 = scheduled_dag.find_bit(q1).index
                    color = two_q_coloring[tuple({i0, i1})]
                    gates_dict[(i0, i1)] = f"{gate.op.name}_{color}"

                    if virtuals := gate.op.label:
                        for virtual in virtuals.split("&"):
                            virtual_zs.update(self._virtual_str_to_dict(virtual))

            if gates_dict:
                moments.append((gates_dict, virtual_zs))

        if final_virtuals := self._scheduler._final_virtuals:
            virtual_zs = {}
            for qubit in final_virtuals:
                virtual_zs.update(self._virtual_str_to_dict(final_virtuals[qubit]))
            moments.append(({}, virtual_zs))

        return moments

    # TODO: generalize for any connectivity?
    def _get_coloring(self, n: int) -> tuple[dict[int, str], dict[int, str]]:
        one_q_coloring = {
            qubit: "red" if qubit % 2 == 0 else "blue" for qubit in range(n)
        }

        flag = True
        two_q_coloring = {}
        for qubit in range(n - 1):
            if qubit % 2 == 0:
                flag = not flag
            two_q_coloring[tuple({qubit, qubit + 1})] = "blue" if flag else "red"

        return one_q_coloring, two_q_coloring

    def _virtual_str_to_dict(self, virtual: str) -> dict[int, str]:
        processed = virtual.split("|")
        params = eval(processed[1])
        qargs = eval(processed[2])
        return {qargs._index: params[0]}
