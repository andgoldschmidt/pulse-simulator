import qiskit
import qiskit_dynamics
import pulse_simulator as ps
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, DensityMatrix
from qiskit.circuit import Qubit
from qiskit.providers import BackendV2
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import RemoveBarriers

from compiler.scheduler import RobustScheduler

# not sure if this should go here or where
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
qiskit_dynamics.array.Array.set_default_backend("jax")

ONE_QUBIT_GATES = ["sx", "x"]
TWO_QUBIT_GATES = ["cx"]
VIRTUAL_GATES = ["rz"]

# custom types
GATE_DICT = dict[int, str] | dict[tuple[int, int], str]
VIRTUAL_ZS = dict[int, float]
SINGLE_MOMENT = tuple[GATE_DICT, VIRTUAL_ZS]
CIRCUIT_MOMENTS = list[tuple[GATE_DICT, VIRTUAL_ZS]]


class Simulator:
    def __init__(
        self, basis_gates: list[str], solver: qiskit_dynamics.Solver, backend: BackendV2
    ):
        required_pulses = []
        for gate in basis_gates:
            if gate in VIRTUAL_GATES:
                continue
            elif gate in ONE_QUBIT_GATES:
                required_pulses += [gate + "_blue", gate + "_red"]
            elif gate in TWO_QUBIT_GATES:
                continue
                # ignored for now, should be able to add once we figure out 2q gates
                required_pulses += [
                    gate + "_control_blue",
                    gate + "_control_red",
                    gate + "_target_blue",
                    gate + "_target_red",
                ]
        self._pulses = dict.fromkeys(required_pulses)
        self._dt = backend.configuration().dt * 1e9
        self._basis_gates = basis_gates
        self._solver = solver

        # set scheduler to not attach viertual gates since we sill extract
        # those from the label of the real gates and treat them accordingly
        self._scheduler = RobustScheduler(
            basis_gates=basis_gates,
            coupling_map=None,
            reattach=False,
            attach_final_virtual=False,
        )

    def set_pulse(self, name: str, pulse: qiskit.pulse.Waveform) -> None:
        if name not in self._pulses.keys():
            raise Exception(f"Pulse {name} not required for simulation.")
        self._pulses[name] = pulse

    def simulate_circuit(self, circuit: QuantumCircuit) -> DensityMatrix:
        # check that all pulses are loaded correctly
        pulses = self._pulses
        for gate_name in pulses:
            if pulses[gate_name] is None:
                raise Exception(f"Pulse {gate_name} not loaded.")

        # get moments dicts from scheduler
        moments = self._get_moments(circuit=circuit)

        # simulate each moment
        num_qubits = circuit.num_qubits
        out = Operator(QuantumCircuit(num_qubits))
        for moment in moments:
            gates = moment[0]
            virtual_zs = moment[1]
            n_qubits = moment[2]
            if n_qubits == 1:
                op = self._simulate_one_qubit_moment(gates, virtual_zs, num_qubits)
            if n_qubits == 2:
                op = self._simulate_two_qubit_moment(gates, virtual_zs, num_qubits)
            # print("\n")
            # print(moment)
            # print(op)
            out = op @ out

        # testing
        # a = RobustScheduler(
        #     basis_gates=self._basis_gates,
        #     coupling_map=None,
        #     reattach=True,
        #     attach_final_virtual=True,
        # )
        # print("\n", Operator(a.run(circuit)).data)

        return out

    def _simulate_one_qubit_moment(
        self, gates: GATE_DICT, virtual_zs: VIRTUAL_ZS, num_qubits: int
    ) -> Operator:
        solver = self._solver
        dt = self._dt
        pulses = self._pulses

        with qiskit.pulse.build() as pulse_moment:
            for qubit in gates:
                channel = qiskit.pulse.DriveChannel(qubit)
                qiskit.pulse.play(pulses[gates[qubit]], channel)

        U0 = ps.qiskit_identity_operator(num_qubits)
        duration = pulse_moment.duration
        sol = solver.solve(
            t_span=[0.0, duration],
            y0=U0,
            signals=pulse_moment,
            max_dt=dt,
            t_eval=np.linspace(0, duration, int(duration / dt) + 1, endpoint=True),
            method="jax_expm",
            magnus_order=1,
        )
        op = sol.y[-1]

        # TODO: handle z gates correctly
        for i in range(num_qubits):
            if i not in virtual_zs:
                virtual_zs[i] = 0.0
        rzs = ps.rz_moment(virtual_zs, [i for i in range(num_qubits)])
        return (op @ rzs).reverse_qargs()

    def _simulate_two_qubit_moment(
        self, gates: GATE_DICT, virtual_zs: VIRTUAL_ZS, num_qubits: int
    ) -> Operator:
        # TODO: use actual two-qubit pulses
        qc = QuantumCircuit(num_qubits)
        for control, target in gates:
            qc.cx(control, target)
        op = Operator(qc)

        for i in range(num_qubits):
            if i not in virtual_zs:
                virtual_zs[i] = 0.0
        rzs = ps.rz_moment(virtual_zs, [i for i in range(num_qubits)])
        return op @ rzs

    def _get_moments(self, circuit: QuantumCircuit) -> CIRCUIT_MOMENTS:
        n = circuit.num_qubits
        one_q_coloring, two_q_coloring = self._get_coloring(n)

        # scheduled circuits and get output dag
        circuit = RemoveBarriers()(circuit)
        scheduled_dag = self._scheduler.run(circuit, return_dag=True)
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
                moments.append((gates_dict, virtual_zs, len(qargs)))

        if final_virtuals := self._scheduler._final_virtuals:
            virtual_zs = {}
            for qubit in final_virtuals:
                virtual_zs.update(self._virtual_str_to_dict(final_virtuals[qubit]))
            moments.append(({}, virtual_zs, 1))

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
            qubit: "red" if qubit % 2 == 0 else "blue" for qubit in range(n)
        }

        flag = True
        two_q_coloring = {}
        for qubit in range(n - 1):
            if qubit % 2 == 0:
                flag = not flag
            two_q_coloring[tuple({qubit, qubit + 1})] = "blue" if flag else "red"

        return one_q_coloring, two_q_coloring
