# Packages
import jax
import qiskit_dynamics

from .pulses import build_pulse_schedule, get_moment_case, build_circuit
from .models import build_solver, backend_edges
from .compiler import MomentBuilder
from .qiskit_simulation_utils import qiskit_identity_operator

# Types
from typing import Union
from qiskit.providers import BackendV2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.pulse.library.symbolic_pulses import ScalableSymbolicPulse
from qiskit.pulse.schedule import ScheduleBlock

from .pulse_simulator_types import CIRCUIT_MOMENT, CIRCUIT_MOMENT_NAMES


class ExperimentalSimulator:
    def __init__(
        self,
        registers: list[int],
        backend: BackendV2,
        basis_gates: [list[str], None] = None,
        coupling_map: Union[list[list[int]], None] = None,
        solver_kwargs: Union[dict, None] = None,
    ):
        # Necessary for this solver to work
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platform_name", "cpu")
        qiskit_dynamics.array.Array.set_default_backend("jax")

        self._registers = registers
        self._backend = backend

        # Top level fields to populate
        self._unique_schedules_by_case: dict[str, ScheduleBlock] = {}
        self._moments: list[CIRCUIT_MOMENT] = []

        # set coupling map using backend if not provided
        if coupling_map is None:
            coupling_map: list[list[int]] = []
            for edge in backend_edges(backend):
                if edge[0] in registers and edge[1] in registers:
                    coupling_map.append(edge)

        self._moment_builder = MomentBuilder(
            basis_gates=basis_gates,
            coupling_map=coupling_map,
        )

        self._solver_kwargs = solver_kwargs
        self._solver = build_solver(registers, backend, **solver_kwargs)
        self._open_solver = False
        if self._open_solver:
            self._solver.model.evaluation_mode = "sparse_vectorized"
        else:
            self._solver.model.evaluation_mode = "sparse"

    def simulate_circuit(
        self,
        circuit: QuantumCircuit,
        single_qubit_gates: dict[str, ScalableSymbolicPulse],
        two_qubit_gates: dict[str, ScalableSymbolicPulse],
        dt: float,
        verbose: bool = False,
        atol=1e-9,
        rtol=1e-9,
    ) -> list[Operator]:
        """Simulate the moments of a circuit."""
        # build moments (hardcode necessary parameters)
        self._moments = self._moment_builder.build_moments(
            circuit,
            separate_moments=True,
            no_end_to_end=True,
            optimization_level=0,
            draw=False,
        )

        # get unique moments
        unique_moments: list[CIRCUIT_MOMENT] = []
        unique_frozen_moments: list[frozenset] = []
        moment_to_unique_moment_index: list[int] = []
        for m in self._moments:
            frozen_m = frozenset(m.items())
            if frozen_m not in unique_frozen_moments:
                unique_moments.append(m)
                unique_frozen_moments.append(frozen_m)
            # store how to find each moment
            moment_to_unique_moment_index.append(unique_frozen_moments.index(frozen_m))

        # build unique schedules
        # group by case to match simulation duration
        self._unique_schedules_by_case = {case: [] for case in CIRCUIT_MOMENT_NAMES}
        unique_moment_index_to_case_index: list[[str, int]] = []
        for moment in unique_moments:
            schedule = build_pulse_schedule(
                [moment],
                qubit_gates=single_qubit_gates,
                cr_gates=two_qubit_gates,
                registers=self._registers,
                backend=self._backend,
                name=f"unique-moment-{unique_moments.index(moment)}",
                dt=dt,
                insert_identities=True,
                verbose=verbose,
            )
            case = get_moment_case(moment)
            self._unique_schedules_by_case[case].append(schedule)
            # store how to find each unique schedule
            unique_moment_index_to_case_index.append(
                [case, len(self._unique_schedules_by_case[case]) - 1]
            )

        # TODO: switch based on open / closed system
        y0 = qiskit_identity_operator(len(self._registers), open=self._open_solver)

        # simulate each case
        unique_results_dict = {}
        for case, schedules in self._unique_schedules_by_case.items():
            if len(schedules) == 0:
                continue
            else:
                final_time = schedules[0].duration * dt
                results = self._solver.solve(
                    t_span=[0.0, final_time],
                    t_eval=[final_time],
                    y0=y0,
                    signals=schedules,
                    atol=atol,
                    rtol=rtol,
                    method="jax_odeint",
                )
                unique_results_dict[case] = [r.y[-1] for r in results]

        # multiply results in order
        out: list[Operator] = [None] * (len(self._moments) + 1)
        out[0] = Operator(QuantumCircuit(len(self._registers)))
        for i, index in enumerate(moment_to_unique_moment_index):
            case, case_index = unique_moment_index_to_case_index[index]
            results = unique_results_dict[case]
            yf = results[case_index]
            out[i + 1] = yf @ out[i]

        return out

    def get_moments(self) -> list[CIRCUIT_MOMENT]:
        return self._moments

    def get_schedules(self, case: Union[str, None] = None) -> list[ScheduleBlock]:
        if case is None:
            return [
                schedule
                for schedules in self._unique_schedules_by_case.values()
                for schedule in schedules
            ]
        else:
            return self._unique_schedules_by_case[case]

    def build_circuit(self, **kwargs) -> QuantumCircuit:
        return build_circuit(self._moments, self._registers, **kwargs)
