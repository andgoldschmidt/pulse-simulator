import functools
import numpy as np
from qiskit.pulse import DriveChannel


def backend_simulation_vars(backend, angular=False, rabi=False, units=1.0):
    """Extract variables for the pulse simulation from the backend.

    NOTE:   It is unclear what happens when Rabi rates are unequal and robust
            control pulses are applied. Does calibration involve simple
            rescaling?

    Args:
        backend (qk.providers.fake_provider.FakePulseBackend): Backend
        angular (bool, optional): If False, returns parameters in frequency ω,
            used as `2πω GHz`. If True, returns parameters in angular
            frequency, f GHz, used as `f GHz`. Defaults to False.
        rabi (bool, optional): If True, use backend Rabi rates. If False,
            use universal Rabi rate of 1. Defaults to False.
        units (int): Conversion factor (E.g. for GHz and ns, units
            would be 1e9).

    Returns:
        Dict: Backend variables for pulse simulation.
    """
    GHz = 1 / units
    ns = units
    config = backend.configuration()
    props = backend.properties()
    n = config.n_qubits
    t1 = {vars_t1(i): props.t1(i) * ns for i in range(n)}
    t2 = {vars_t2(i): props.t2(i) * ns for i in range(n)}
    H = {
        k: v * GHz if angular else v * GHz / 2 / np.pi
        for k, v in config.hamiltonian["vars"].items()
    }
    if not rabi:
        for i in range(n):
            if angular:
                H[vars_rabi(i)] = 1.0
            else:
                H[vars_rabi(i)] = 1 / 2 / np.pi
    return functools.reduce(lambda a, b: {**a, **b}, [H, t1, t2])


def vars_coupling(i, j):
    assert i != j
    return f"jq{min(i,j)}q{max(i,j)}"


def vars_t1(i):
    return f"t1q{i}"


def vars_t2(i):
    return f"t2q{i}"


def vars_frequency(i):
    return f"wq{i}"


def vars_rabi(i):
    return f"omegad{i}"


def vars_anharmonicity(i):
    return f"delta{i}"


def get_drive_channel(i, backend, name=False):
    # Standard drive channel
    channels = []
    for ch in backend.configuration().get_qubit_channels(i):
        if type(ch) is DriveChannel:
            channels.append(ch)
    if name:
        return channels[0].name
    else:
        return channels[0]


def get_control_channel(c, t, backend, name=False):
    # Cross resonance channel
    channels = backend.configuration().control_channels[(c, t)]
    if name:
        return channels[0].name
    else:
        return channels[0]


def backend_carriers(backend, variables):
    n = backend.configuration().n_qubits
    carriers = {
        get_drive_channel(i, backend, name=True): variables[vars_frequency(i)]
        for i in range(n)
    }
    for c, t in backend_edges(backend, directed=True):
        uct = get_control_channel(c, t, backend, name=True)
        carriers.update({uct: variables[vars_frequency(t)]})
    return carriers


def backend_edges(backend, directed=False):
    """Get the unique edges of the provided backend based on allowed two-qubit
    gates.

    Arguments:
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend

    Returns:
        List of unique edges (as tuples)
    """
    dir_graph = backend.configuration().coupling_map
    if not directed:
        dir_graph = np.unique([set(edge) for edge in dir_graph])
    return [tuple(edge) for edge in dir_graph]
