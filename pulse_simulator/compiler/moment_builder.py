from typing import Union

# Used in the `MomentBuilder` class as types
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import RemoveBarriers

from .scheduler import RobustScheduler
from ..pulse_simulator_types import (
    CIRCUIT_MOMENT,
    TIMED_CIRCUIT_MOMENTS,
    ONE_QUBIT_GATE_NAMES,
    TWO_QUBIT_GATE_NAMES,
    VIRTUAL_GATE_NAMES,
    IDENTITY_GATE_NAME,
)


class MomentBuilder:

    def __init__(
        self,
        registers: list[int],
        coupling_map: list[list[int]],
    ) -> None:
        """Creates a `MomentBuilder` instance.

        Args:
            - basis_gates (list[str]): the basis gates to transpile to
        """
        self._registers = registers
        self._basis_gates = (
            ONE_QUBIT_GATE_NAMES + TWO_QUBIT_GATE_NAMES + VIRTUAL_GATE_NAMES
        )

        # set scheduler to not attach virtual gates since we will extract
        # those from the label of the real gates and treat them accordingly
        self._scheduler = RobustScheduler(
            self._basis_gates,
            coupling_map=coupling_map,
            reattach=False,
            attach_final_virtual=False,
        )

    def build_moments(
        self,
        circuit: QuantumCircuit,
        separate_moments: bool = True,
        no_end_to_end: bool = True,
        optimization_level: Union[int, None] = 1,
        insert_identities: bool = True,
        draw: bool = False,
    ) -> list[CIRCUIT_MOMENT]:
        """Builds list of dictionary moments. The default format is:
        ```
        [
            {0: 1.5707963267948966, 1: -3.141592653589793},
            {0: "SX", 1: "SX"},
            {1: -3.141592653589793},
            {(0, 1): "ECR"},
            {0: "X"},
        ]
        ```

        Args:
            -   circuit (QuantumCircuit): the input circuit
            -   separate_moments (bool): whether to separate one- and two-qubit gates
                into separate moments
            -   no_end_to_end (bool): whether to forbid any adjacent two-qubit gates to
                have end-to-end connections
            -   optimization_level (Union[int, None]): the optimization level to use in
                Qiskit's transpiler. Useful to set to 0 when sending circuits that are
                equal to the identity or for which you don't want any changes other than
                transpilation and scheduling. Can be set to None if no transpilation is
                desired. Defaults to 1.
            -   insert_identities (bool): whether to insert identity gates in moments of
                single qubit gates where no gate is applied.
            -   draw (bool): whether to draw the circuit after scheduling.

        Returns:
            Moments
        """
        # scheduled circuits and get output dag
        circuit = RemoveBarriers()(circuit)
        scheduled_dag = self._scheduler.run(
            circuit,
            separate_moments=separate_moments,
            no_end_to_end=no_end_to_end,
            return_dag=True,
            optimization_level=optimization_level,
        )
        if draw:
            print(dag_to_circuit(scheduled_dag).draw())
        subdags: list[DAGCircuit] = [layer["graph"] for layer in scheduled_dag.layers()]

        # build moments dicts
        moments: list[CIRCUIT_MOMENT] = []
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
                    gates_dict[index] = f"{gate.op.name}".upper()

                    if virtual := gate.op.label:
                        virtual_zs.update(self._virtual_str_to_dict(virtual))

                    if insert_identities:
                        for qubit in self._registers:
                            if qubit not in gates_dict:
                                gates_dict[qubit] = IDENTITY_GATE_NAME

                # two-qubit operation
                elif len(qargs) == 2:
                    q0, q1 = qargs[0], qargs[1]
                    i0 = scheduled_dag.find_bit(q0).index
                    i1 = scheduled_dag.find_bit(q1).index
                    gates_dict[(i0, i1)] = f"{gate.op.name}".upper()

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

    def convert_to_timed(
        self,
        moments: list[CIRCUIT_MOMENT],
        n_qubits: int,
        gate_durations: dict[int, int] = {1: 50, 2: 270},
    ) -> TIMED_CIRCUIT_MOMENTS:
        """
        Converts the moments to a timed format. The format is:
        ```
        {
            0: {0: "SX", 50: "ECR_0", 320: "X", 370: "ECR_1"},
            1: {0: "SX", 50: "ECR_1", 370: "ECR_0", 640: "X"},
        }
        ```

        Args:
            - moments (list[CIRCUIT_MOMENT]): the moments to convert
            - n_qubits (int): the number of qubits in the circuit
            - gate_durations (dict[int, int]): 1: duration of one-qubit gates, 2: duration
                of two-qubit gates

        Returns:
            The moments in a timed format.
        """
        timed_moments: dict[int, dict[str]] = {i: {} for i in range(n_qubits)}
        current_time: dict[int:float] = {i: 0 for i in range(n_qubits)}

        for moment in moments:
            for qubit in moment:
                gate = moment[qubit]
                if isinstance(gate, float):  # virtual gates
                    if (
                        qubit in timed_moments
                        and current_time[qubit] in timed_moments[qubit]
                    ):
                        previous_gate = timed_moments[qubit][current_time[qubit]]
                        timed_moments[qubit][current_time[qubit]] = (
                            previous_gate,
                            gate,
                        )
                    else:
                        timed_moments[qubit].update({current_time[qubit]: gate})
                    continue

                gate_name = gate.lower()
                if gate_name in ONE_QUBIT_GATE_NAMES:
                    if (
                        qubit in timed_moments
                        and current_time[qubit] in timed_moments[qubit]
                    ):
                        previous_gate = timed_moments[qubit][current_time[qubit]]
                        timed_moments[qubit][current_time[qubit]] = (
                            previous_gate,
                            gate,
                        )
                    else:
                        timed_moments[qubit].update({current_time[qubit]: gate})
                    current_time[qubit] += gate_durations[1]
                elif gate_name in TWO_QUBIT_GATE_NAMES:
                    gate_time = max([current_time[qubit[0]], current_time[qubit[1]]])
                    timed_moments[qubit[0]].update({gate_time: f"{gate}_0"})
                    timed_moments[qubit[1]].update({gate_time: f"{gate}_1"})
                    current_time[qubit[0]] = gate_time + gate_durations[2]
                    current_time[qubit[1]] = gate_time + gate_durations[2]

            # adjust time since moments are supposed to be parallel
            max_time = max([current_time[qubit] for qubit in current_time])
            for qubit in current_time:
                current_time[qubit] = max_time

        return timed_moments
