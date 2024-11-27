# Packages
from .pulses import build_pulse_schedule, get_moment_case, build_circuit, build_operator
from .models import build_solver, get_backend_edges, BackendSampler
from .compiler import MomentBuilder
from .qiskit_simulation_utils import qiskit_identity_operator

# Types
from typing import Union
from qiskit.providers import Backend
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SuperOp
from qiskit.pulse.library.symbolic_pulses import ScalableSymbolicPulse
from qiskit.pulse.schedule import ScheduleBlock

from .pulse_simulator_types import CIRCUIT_MOMENT, CIRCUIT_MOMENT_NAMES

import jax
import numpy as np


class Simulator:
    def __init__(
        self,
        registers: list[int],
        backend: Backend,
        units: float = 1e9,
    ):
        """Creates a `Simulator` instance."""
        self._registers = registers
        self._backend = backend

        # Get backend coupling map as a list of edges
        self._coupling_map = get_backend_edges(
            self._backend, registers=self._registers, directed=True
        )

        # build moment builder
        self._moment_builder = MomentBuilder(self._registers, self._coupling_map)

        # default solver variables (NOTE: Qiskit V1.0)
        self._simulation_units = units
        self._backend_sampler = BackendSampler(self._simulation_units, seed=17)
        self._simulation_vars = self._backend_sampler.generate_simulation_vars(
            self._registers, self._coupling_map
        )

        # Filled by `simulate_circuit`
        # NOTE: If the BackendSampler is included here, it will be used to generate
        #       the simulation variables from mean values instead of samples.
        self._solver_kwargs = {
            "backend_sampler": self._backend_sampler,
        }

        # top level fields to populate
        self._unique_schedules_by_case: dict[str, ScheduleBlock] = {}
        self._moments: list[CIRCUIT_MOMENT] = []

    def simulate_circuit(
        self,
        circuit: QuantumCircuit,
        single_qubit_gates: dict[str, ScalableSymbolicPulse],
        two_qubit_gates: dict[str, ScalableSymbolicPulse],
        dt: float,
        include_crosstalk: bool = False,
        include_dissipation: bool = False,
        insert_identities: bool = True,
        insert_dynamical_decoupling: bool = False,
        verbose: bool = False,
        simulation_vars: Union[dict, None] = None,
        cross_resonance_model_name: str = "Toy",
        atol=1e-9,
        rtol=1e-9,
        gpu: bool = False,
        rotary_gates: Union[dict[str, ScalableSymbolicPulse], None] = None,
        virtual_cases: list[str] = ["phase"],
        return_operators: bool = False,
    ) -> Union[Operator, SuperOp]:
        """Simulate the moments of a circuit.

        Args:
            circuit (QuantumCircuit): the input circuit
            single_qubit_gates (dict[str, ScalableSymbolicPulse]): the single-qubit gates
            two_qubit_gates (dict[str, ScalableSymbolicPulse]): the two-qubit gates
            dt (float): the simulation time step
            include_crosstalk (bool): whether to include crosstalk
            include_dissipation (bool): whether to include qubit decay
            verbose (bool): whether to print additional information
            simulation_vars (dict): the configuration variables
            atol (float): the absolute tolerance
            rtol (float): the relative tolerance
            gpu (bool): whether to use GPU acceleration

        Note on cases:
            "single" if the moment is a single qubit moment, "ecr" if it is a two
            qubit moment, 'phase' if it is a Z gate moment.

        Returns:
            Union[Operator, SuperOp]: the circuit operator after each moment
        """
        # build solver (add args)
        self._solver_kwargs["include_crosstalk"] = include_crosstalk
        self._solver_kwargs["include_dissipation"] = include_dissipation
        self._solver_kwargs["simulation_vars"] = (
            self._simulation_vars if simulation_vars is None else simulation_vars
        )
        self._solver_kwargs["simulation_dt"] = dt
        self._solver_kwargs["cross_resonance_model_name"] = cross_resonance_model_name

        self._solver = build_solver(
            self._registers, self._backend, **self._solver_kwargs
        )

        # set evaluation mode and initial state
        sparse_or_dense = "dense" if gpu else "sparse"
        self._open_solver = include_dissipation
        if self._open_solver:
            self._solver.model.evaluation_mode = f"{sparse_or_dense}_vectorized"
        else:
            self._solver.model.evaluation_mode = sparse_or_dense

        if gpu:
            method = {"method": "jax_expm", "max_dt": dt}
            # Also consider: jax_RK4_parallel
        else:
            method = {
                "atol": atol,
                "rtol": rtol,
                "method": "jax_odeint",
            }

        # set circuit moments
        self._moments = self.build_moments(circuit)

        # get unique moments
        unique_moments: list[CIRCUIT_MOMENT] = []
        unique_frozen_moments: list[frozenset] = []
        self._moment_to_unique_moment_index: list[int] = []
        for m in self._moments:
            frozen_m = frozenset(m.items())
            if frozen_m not in unique_frozen_moments:
                unique_moments.append(m)
                unique_frozen_moments.append(frozen_m)
            # store how to find each moment
            self._moment_to_unique_moment_index.append(
                unique_frozen_moments.index(frozen_m)
            )

        # build unique schedules
        # group by case to match simulation duration
        self._unique_schedules_by_case = {case: [] for case in CIRCUIT_MOMENT_NAMES}
        self._unique_moment_by_case = {case: [] for case in CIRCUIT_MOMENT_NAMES}
        unique_moment_index_to_case_index: list[list[str, int]] = []
        for moment in unique_moments:
            schedule = build_pulse_schedule(
                [moment],
                qubit_gates=single_qubit_gates,
                cr_gates=two_qubit_gates,
                registers=self._registers,
                backend=self._backend,
                name=f"unique-moment-{unique_moments.index(moment)}",
                dt=dt,
                insert_identities=insert_identities,
                insert_dynamical_decoupling=insert_dynamical_decoupling,
                verbose=verbose,
                rotary_gates=rotary_gates,
            )
            case = get_moment_case(moment)
            self._unique_schedules_by_case[case].append(schedule)
            self._unique_moment_by_case[case].append(moment)
            # store how to find each unique schedule
            unique_moment_index_to_case_index.append(
                [case, len(self._unique_schedules_by_case[case]) - 1]
            )

        # simulate each case
        y0 = qiskit_identity_operator(len(self._registers), open=self._open_solver)
        self._unique_results_dict = {}
        for case, schedules in self._unique_schedules_by_case.items():
            if len(schedules) == 0 or case in virtual_cases:
                continue
            else:
                final_time = schedules[0].duration * dt
                results = self._solver.solve(
                    t_span=[0.0, final_time],
                    t_eval=[final_time],
                    y0=y0,
                    signals=schedules,
                    **method,
                )
                self._unique_results_dict[case] = [r.y[-1] for r in results]

        # simulate any virtual cases
        for case, moments in self._unique_moment_by_case.items():
            if len(moments) == 0 or case not in virtual_cases:
                continue
            else:
                self._unique_results_dict[case] = []
                for m in moments:
                    if self._open_solver:
                        self._unique_results_dict[case].append(
                            SuperOp(build_operator([m], self._registers))
                        )
                    else:
                        self._unique_results_dict[case].append(
                            build_operator([m], self._registers)
                        )

        # multiply results in order
        # if gpu:
        #     op_init = jax.numpy.array(
        #         qiskit_identity_operator(len(self._registers), open=self._open_solver).data
        #     )
        #     final, res = jax.lax.scan(self._scan_fn, op_init, jax.numpy.stack(ops))
        #     res = jax.numpy.vstack([res, final[None, :, :]])
        #     res = [SuperOp(np.array(r)) if self._open_solver else Operator(np.array(r)) 
        #            for r in res]
        # else:

        res: list[Union[Operator, SuperOp]] = [None] * (len(self._moments) + 1)
        res[0] = qiskit_identity_operator(len(self._registers))
        for i, index in enumerate(self._moment_to_unique_moment_index):
            op_or_superop = self._get_op_or_superop(
                index, unique_moment_index_to_case_index, gpu=False
            )
            res[i + 1] = op_or_superop @ res[i]

        if return_operators:
            # instantiate the operator list
            ops = [
                self._get_op_or_superop(index, unique_moment_index_to_case_index, gpu=gpu)
                for index in self._moment_to_unique_moment_index
            ]
            return res, ops
        else:
            return res

    def _scan_fn(self, carry, x):
        y = x @ carry
        # carry, save
        return y, x

    def _get_op_or_superop(
        self, 
        index: int,
        unique_moment_index_to_case_index: list[list[str, int]],
        gpu=False
    ) -> Union[Operator, SuperOp]:
        case, case_index = unique_moment_index_to_case_index[index]
        if gpu:
            return self._unique_results_dict[case][case_index].data
        else:
            return self._unique_results_dict[case][case_index]

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

    def build_moments(self, circuit: QuantumCircuit) -> list[CIRCUIT_MOMENT]:
        # build moments (hardcode necessary parameters)
        return self._moment_builder.build_moments(
            circuit,
            separate_moments=True,
            no_end_to_end=True,
            insert_identities=True,
            optimization_level=0,
            draw=False,
        )

    def build_circuit(
        self, moments: Union[list[CIRCUIT_MOMENT], None] = None, **kwargs
    ) -> QuantumCircuit:
        if moments is None:
            moments = self._moments
        return build_circuit(moments, self._registers, **kwargs)
