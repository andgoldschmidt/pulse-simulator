import functools
import numpy as np
from qiskit.pulse.channels import ControlChannel, DriveChannel

# Global offset for phase channels
PHASE_CHANNEL_INDEX_SHIFT = 1000

# Types
from typing import Union, Callable


def draw_t1(rng):
    # units: s (source: FakeManila)
    mean = 1e-4
    std = 1e-05
    return rng.normal(mean, std)


def draw_t2(rng):
    # units: s (source: FakeManila)
    mean = 5e-5
    std = 5e-06
    return rng.normal(mean, std)


def draw_frequency(rng):
    # units: Hz (source: FakeManila)
    mean = 2 * np.pi * 5e9
    std = 2 * np.pi * 0.05e9
    return rng.normal(mean, std)


def draw_anharmonicity(rng):
    # units: Hz (source: FakeManila)
    mean = -2 * np.pi * 0.05e9
    std = 2 * np.pi * 0.0001e9
    return rng.normal(mean, std)    


def draw_coupling(rng):
    # units: Hz (source: FakeManila)
    mean = 2 * np.pi * 2e6
    std = 2 * np.pi * 0.03e6
    return rng.normal(mean, std)


def draw_rabi(rng):
    # calibrated to 1.0
    return 1e9


def vars_t1(i):
    return f"t1q{i}"


def vars_t2(i):
    return f"t2q{i}"


def vars_frequency(i):
    return f"wq{i}"


def vars_coupling(i, j):
    assert i != j
    return f"jq{min(i,j)}q{max(i,j)}"


def vars_rabi(i):
    return f"omegad{i}"


def vars_anharmonicity(i):
    return f"delta{i}"


def backend_simulation_vars(backend, angular=False, rabi=False, units=1e9):
    """Qiskit versions < 1.0 only.

    Extract variables for the pulse simulation from the backend.

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



def generate_simulation_vars(
    registers: list[int],
    coupling_map: list[list[int]],
    t1_fn: Callable = draw_t1,
    t2_fn: Callable = draw_t2,
    w_fn: Callable = draw_frequency,
    j_fn: Callable = draw_coupling,
    omega_fn: Callable = draw_rabi,
    alpha_fn: Callable = draw_anharmonicity,
    angular: bool = False,
    units: float = 1e9,
    seed: int = None,
):
    """
    This function creates the necessary variables for the pulse simulation. The
    required variables are:

    - wqi: Frequency of the qubit i.
    - jqiqj: Coupling strength between qubits i and j.
    - omegadi: Rabi rate for the qubit i.
    - alphaqi: Anharmonicity of the qubit i.
    - t1qi: T1 time of the qubit i.
    - t2qi: T2 time of the qubit i.

    Optionally, we could model higher levels of the qubits or other
    properties. This will depend on the Hamiltonian we want to simulate.
    Currently, only qubits are supported.

    Args:
        registers (list[int]): List of qubits.
        coupling_map (list[list[int]]): List of qubit pairs.
        t1_fn (function, optional): Function to draw T1 times.
        t2_fn (function, optional): Function to draw T2 times.
        w_fn (function, optional): Function to draw qubit frequencies.
        j_fn (function, optional): Function to draw coupling strengths.
        alpha_fn (function, optional): Function to draw anharmonicities.
        omega_fn (function, optional): Function to draw Rabi rates.
        angular (bool, optional): If False, returns parameters in frequency ω,
            used as `2πω GHz`. If True, returns parameters in angular
            frequency, f GHz, used as `f GHz`. Defaults to False.
        units (int): Conversion factor (E.g. for GHz and ns, units
            would be 1e9).
        seed (int, optional): Seed for the random number generator.

    Returns:
        Dict: Variables for pulse simulation.
    """
    GHz = 1 / units
    ns = units

    # Assert functions are not None
    assert t1_fn is not None
    assert t2_fn is not None
    assert w_fn is not None
    assert j_fn is not None
    assert omega_fn is not None
    assert alpha_fn is not None

    # Set seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Qubit frequencies
    GHz2pi = GHz / 2 / np.pi
    wq = {
        vars_frequency(i): w_fn(rng) * GHz if angular else w_fn(rng) * GHz2pi
        for i in registers
    }

    # Anharmonicities
    deltaq = {
        vars_anharmonicity(i): alpha_fn(rng) * GHz if angular else alpha_fn(rng) * GHz2pi
        for i in registers
    }

    # Coupling strengths
    jqq = {
        vars_coupling(i, j): j_fn(rng) * GHz if angular else j_fn(rng) * GHz2pi
        for i, j in coupling_map
    }

    # Rabi rates (1.0 => calibration is correct)
    omegad = {
        vars_rabi(i): omega_fn(rng) * GHz if angular else omega_fn(rng) * GHz2pi
        for i in registers
    }

    # T1 and T2 times
    t1 = {vars_t1(i): t1_fn(rng) * ns for i in registers}
    t2 = {vars_t2(i): t2_fn(rng) * ns for i in registers}

    return functools.reduce(lambda a, b: {**a, **b}, [t1, t2, wq, jqq, deltaq, omegad])


def generate_default_simulation_vars(
    registers: list[int],
    coupling_map: list[list[int]],
    angular: bool = False,
    units: float = 1e9,
):
    seed = 17
    return generate_simulation_vars(
        registers,
        coupling_map,
        t1_fn=draw_t1,
        t2_fn=draw_t2,
        w_fn=draw_frequency,
        j_fn=draw_coupling,
        alpha_fn=draw_anharmonicity,
        omega_fn=draw_rabi,
        angular=angular,
        units=units,
        seed=seed,
    )


def get_channels(registers, backend, allowed_channels=(ControlChannel, DriveChannel)):
    channel_dict = {}
    for i in registers:
        for ch in backend.configuration().get_qubit_channels(i):
            if isinstance(ch, allowed_channels):
                channel_dict[ch.name] = ch
    return channel_dict


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


def get_phase_channel(i, backend, name=False):
    # Arbitrary index shift based on device qubits
    n = backend.configuration().n_qubits
    index_shift = int(10 ** np.ceil(np.log10(n)))

    # Phase is a mock drive channel
    d = DriveChannel(i + index_shift)
    if name:
        return d.name
    else:
        return d


def get_control_channel(c, t, backend, name=False):
    # Cross resonance channel
    try:
        channels = backend.configuration().control_channels[(c, t)]
    except KeyError as e:
        raise Exception(f"Backend edge not found for {c} and {t}") from e
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
    return [list(edge) for edge in dir_graph]


def backend_coupling_map(registers, backend):
    coupling_map = []
    for edge in backend_edges(backend):
        if edge[0] in registers and edge[1] in registers:
            coupling_map.append(edge)
    return coupling_map