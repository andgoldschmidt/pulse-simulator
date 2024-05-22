import functools
import numpy as np
from qiskit_dynamics import Solver
from typing import Union

from .qiskit_backend_utils import backend_simulation_vars, backend_edges
from .one_qubit_models import rx_model, rz_model, qubit_decay_model
from .two_qubit_models import cross_resonance_model, crosstalk_model


def build_solver(*args, **kwargs) -> Solver:
    """
    Build a solver for the given Qiskit backend and registers.

    See `build_model` for the arguments.

    Returns:
        Solver: The solver for the given backend and registers.
    """
    return Solver(**build_model(*args, **kwargs))


def build_model(
    registers: list,
    backend,
    include_crosstalk: bool = False,
    include_dissipation: bool = False,
    verbose: bool = False,
    simulation_units: float = 1e9,
    simulation_rabi: bool = False,
    simulation_dt: Union[float, None] = None,
    simulation_vars: Union[dict, None] = None,
) -> dict:
    """
    Build a model for the given Qiskit backend and registers.

    This function can be useful for inspecting the model parameters, or for
    making custom modifications to the model before passing to the solver.

    The simulation variables are are constructed from the provided backend if they are
    not provided directly. Note that if the `simulation_rabi` is True, you have to
    calibrate. The `simulation_units` are used to convert the backend units to the
    desired units. The `simulation_dt` is used to set the simulation time step, and
    can be set to the backend time step if not provided.

    Args:
        registers (list): The qubit registers to simulate.
        backend (Backend): The backend to simulate.
        include_crosstalk (bool): Include crosstalk in the simulation.
        include_dissipation (bool): Include qubit decay in the simulation.
        verbose (bool): Print additional information.
        simulation_units (float): The units to use for the simulation.
        simulation_rabi (bool): Use the backend Rabi frequency.
        simulation_dt (float): The simulation time step.
        simulation_vars (dict): The simulation variables.

    Returns:
        dict: The model parameters for the given backend and registers.
    """
    if simulation_dt is None:
        simulation_dt = backend.configuration().dt * simulation_units

    if simulation_vars is None:
        simulation_vars = backend_simulation_vars(
            backend, rabi=simulation_rabi, units=simulation_units
        )

    if verbose:
        print(f"Building {backend.name()} model...")
        print(f"\tTimestep: {simulation_dt: 0.2f}")
        print(f"\tRabi: {simulation_rabi}")
        print(f"\tCrosstalk: {include_crosstalk}")
        print(f"\tDissipation: {include_dissipation}")

    # Partially compile model functions with parameters
    # =====
    H_rx = functools.partial(
        rx_model,
        registers=registers,
        backend=backend,
        variables=simulation_vars,
        rotating_frame=True,
    )

    H_rz = functools.partial(
        rz_model,
        registers=registers,
        backend=backend,
    )

    H_cr = functools.partial(
        cross_resonance_model,
        registers=registers,
        backend=backend,
        variables=simulation_vars,
        model_name="Toy",
    )

    A_decay = functools.partial(
        qubit_decay_model, registers=registers, variables=simulation_vars
    )

    edges = backend_edges(backend)

    # Create a system model
    # =====
    # Single qubit terms
    H_drift = 0.0
    Hs_control = []
    Hs_channels = []
    As_static = []
    for qubit in registers:
        # Single qubit terms
        Hj_drift, Hjs_control, Hjs_channel = H_rx(qubit)
        Hj_phase, Hj_phase_ch = H_rz(qubit)
        Ajs_static = A_decay(qubit)

        H_drift += Hj_drift
        Hs_control += Hjs_control + [Hj_phase]
        Hs_channels += Hjs_channel + [Hj_phase_ch]
        As_static += Ajs_static

    # Two qubit terms
    added_edges = []
    for control, target in edges:
        # Get both directions
        if control in registers and target in registers:
            added_edges.append((control, target))
            for pair in [(control, target), (target, control)]:
                Hj_drift, Hjs_control, Hjs_channel = H_cr(pair)
                H_drift += Hj_drift
                Hs_control += Hjs_control
                Hs_channels += Hjs_channel
    if verbose:
        print(f"\tEdges: {added_edges}")

    # Crosstalk
    H_xtalk = crosstalk_model(registers, edges, simulation_vars)

    # Control channels
    # =====
    # Some models may contribute repeat control channels.
    # Drop any repeates; duplication adds control contributions.
    Hs_channels, indices = np.unique(Hs_channels, return_index=True)
    Hs_control = [Hs_control[i] for i in indices]

    # For RWA / Rotating frame. No carriers in this example
    # =====
    registered_carriers = {c: 0 for c in Hs_channels}

    # Resolve args
    # =====
    return {
        "static_hamiltonian": H_drift + H_xtalk if include_crosstalk else H_drift,
        "hamiltonian_operators": Hs_control,
        "hamiltonian_channels": Hs_channels,
        "static_dissipators": As_static if include_dissipation else None,
        "rotating_frame": None,
        "rwa_cutoff_freq": None,
        "channel_carrier_freqs": registered_carriers,
        "dt": simulation_dt,
    }
