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
TIMED_CIRCUIT_MOMENTS = dict[int, dict[int, str | float]]


class MomentBuilder:
    def __init__(
        self,
        basis_gates: list[str] = ["rz", "sx", "x", "ecr"],
        gate_durations: dict[int, int] = {1: 50, 2: 270},
    ) -> None:
        """Creates a `MomentBuilder` instance.

        Args:
            - basis_gates (list[str]): the basis gates to transpile to
            - gate_durations (dict[int, int]): one- and two-qubit gates durations in dt 
            units. By default, this is set to 50 for one qubit gates and 270 for two qubit gates.
        """
        # set scheduler to not attach virtual gates since we sill extract
        # those from the label of the real gates and treat them accordingly
        self._scheduler = RobustScheduler(
            basis_gates=basis_gates,
            coupling_map=None,
            reattach=False,
            attach_final_virtual=False,
        )
        self._gate_durations = gate_durations

    def build_moments(
        self,
        circuit: QuantumCircuit,
        separate_moments: bool = True,
        no_end_to_end: bool = True,
        timed: bool = False,
    ) -> CIRCUIT_MOMENTS | TIMED_CIRCUIT_MOMENTS: 
        """Builds list of dictionary moments. The default format is:
        ```
        [
            {0: 1.5707963267948966, 1: -3.141592653589793},
            {0: "SX1", 1: "SX2"},
            {1: -3.141592653589793},
            {(0, 1): "ECR2"},
            {0: "X1"},
        ]
        ```

        Args:
            - circuit (QuantumCircuit): the input circuit
            - separate_moments (bool): whether to separate one- and two-qubit gates into separate 
            moments
            - no_end_to_end (bool): whether to forbid any adjacent two-qubit gates to have
            end-to-end connections
            - timed (bool): whether to return the moments in a timed format, for example:
            ```
            {
                0: {0: "SX1", 50: "ECR2_0", 320: "X1", 370: "ECR2_1"},
                1: {0: "SX2", 50: "ECR2_1", 370: "ECR2_0", 640: "X2"},
            }
            ```

        Returns:
            Moments in the format specified by the `timed` flag.
        """
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

        if timed:
            return self._convert_to_timed(moments, n)
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
        one_q_coloring = {qubit: "1" if qubit % 2 == 0 else "2" for qubit in range(n)}

        flag = True
        two_q_coloring = {}
        for qubit in range(n - 1):
            if qubit % 2 == 0:
                flag = not flag
            two_q_coloring[tuple({qubit, qubit + 1})] = "1" if flag else "2"

        return one_q_coloring, two_q_coloring

    def _convert_to_timed(self, moments: CIRCUIT_MOMENTS, n_qubits: int) -> TIMED_CIRCUIT_MOMENTS:
        timed_moments: dict[int, dict[str]] = {i: {} for i in range(n_qubits)}
        current_time: dict[int: float] = {i: 0 for i in range(n_qubits)}

        gate_durations = self._gate_durations
        for moment in moments:
            for qubit in moment:
                gate = moment[qubit]
                if isinstance(gate, float): # virtual gates
                    timed_moments[qubit].update({current_time[qubit]: gate})
                    continue 
                gate_name = gate[:-1].lower()
                if gate_name in ONE_QUBIT_GATES:
                    timed_moments[qubit].update({current_time[qubit]: gate})
                    current_time[qubit] += gate_durations[1]
                elif gate_name in TWO_QUBIT_GATES:
                    gate_time = max([current_time[qubit[0]], current_time[qubit[1]]])
                    timed_moments[qubit[0]].update({gate_time: f"{gate}_0"})
                    timed_moments[qubit[1]].update({gate_time: f"{gate}_1"})
                    current_time[qubit[0]] = gate_time + gate_durations[2]
                    current_time[qubit[1]] = gate_time + gate_durations[2]                    

        return timed_moments
