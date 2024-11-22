import functools
import numpy as np
from qiskit.pulse.channels import ControlChannel, DriveChannel
from qiskit import __version__ as qiskit_version
from packaging.version import Version

# Global offset for phase channels
PHASE_CHANNEL_INDEX_SHIFT = 1000

# Types
from typing import Callable


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


class BackendSampler:
    """
    This class generates simulation variables for the pulse simulator.

    The default simulation variables are taken from IBM Brisbane, see:
    https://arxiv.org/abs/2410.00916

    The coupling strength could be taken from:
    https://doi.org/10.1103/PhysRevA.93.060302
    ...but this turned out to be too strong.
    An observed coupling strength from XY4 experiments was used instead.

    For T1 and T2, we want good qubits, so drop the deviation by a factor of 10.

    A calibrated Rabi uses the units as the value (e.g. 1e9).

    Note that models may rely on default values for all qubits because constructing
    a device with good frequency layouts (bounded 1/Δω) cannot be done randomly.
    """

    def __init__(self, simulation_units, seed=None):
        self.units = simulation_units
        self.rng = np.random.default_rng(seed)

        # Default parameters for IBM Brisbane (SI units)
        self.params = {
            "t1": {"mean": 216 * 1e-06, "std": 80 * 1e-06 / 10},
            "t2": {"mean": 154 * 1e-06, "std": 82 * 1e-06 / 10},
            "frequency": {"mean": 2 * np.pi * 4.90e9, "std": 2 * np.pi * 0.11e9},
            "anharmonicity": {"mean": -2 * np.pi * 0.31e9, "std": 2 * np.pi * 0.0054e9},
            "coupling": {"mean": 2 * np.pi * 1.72e6, "std": 0.0},
            "rabi": {"mean": 1e9, "std": 0.0},
        }

    def draw(self, key, mean=False, std=False):
        if key not in self.params:
            raise ValueError(f"Key {key} not found in the parameters.")

        mu = self.params[key]["mean"]
        sigma = self.params[key]["std"]

        if mean and std:
            return mu, sigma
        elif mean:
            return mu
        elif std:
            return sigma
        else:
            if np.isclose(sigma, 0.0):
                return mu
            else:
                return self.rng.normal(mu, sigma)

    def draw_t1(self, mean=False, std=False):
        return self.draw("t1", mean=mean, std=std)

    def draw_t2(self, t1=0.0, mean=False, std=False):
        t2 = self.draw("t2", mean=mean, std=std)
        if 2 * t1 <= t2 or t2 <= 0:
            raise ValueError(f"Bad draw. Require: 0<2*t1<t2, got t1={t1}, t2={t2}")
        return t2

    def draw_frequency(self, mean=False, std=False):
        return self.draw("frequency", mean=mean, std=std)

    def draw_anharmonicity(self, mean=False, std=False):
        return self.draw("anharmonicity", mean=mean, std=std)

    def draw_coupling(self, mean=False, std=False):
        return self.draw("coupling", mean=mean, std=std)

    def draw_rabi(self, mean=False, std=False):
        return self.draw("rabi", mean=mean, std=std)

    def time_units(self, var):
        # Convert to the simulation units (e.g. ns)
        ns = self.units
        return var * ns

    def frequency_units(self, var, angular=False):
        # Convert to the simulation units (e.g. GHz)
        GHz = 1 / self.units
        return var * GHz if angular else var * GHz / 2 / np.pi

    def generate_simulation_vars(
        self,
        registers: list[int],
        coupling_map: list[list[int]],
        angular: bool = False,
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

        See the `draw_` functions for the default values.

        NOTE: It's possible that to do this correctly (i.e. with a good frequency layout
        and appropriate correlation among t1 and t2), we would want to have the random
        samples depend on the qubit index. This is not done here, but it's a possibility.

        Args:
            registers (list[int]): List of qubits.
            coupling_map (list[list[int]]): List of qubit pairs.
            angular (bool, optional): If False, returns parameters in frequency ω,
                used as `2πω GHz`. If True, returns parameters in angular
                frequency, f GHz, used as `f GHz`. Defaults to False (check models).
        Returns:
            Dict: Variables for pulse simulation.
        """
        time = lambda x: self.time_units(x)
        freq = lambda x: self.frequency_units(x, angular=angular)

        # Qubit frequencies
        wq = {vars_frequency(i): freq(self.draw_frequency()) for i in registers}

        # Anharmonicities
        deltaq = {
            vars_anharmonicity(i): freq(self.draw_anharmonicity()) for i in registers
        }

        # Coupling strengths
        jqq = {vars_coupling(i, j): freq(self.draw_coupling()) for i, j in coupling_map}

        # Rabi rates (1.0 => calibration is correct)
        omegad = {vars_rabi(i): freq(self.draw_rabi()) for i in registers}

        # T1 and T2 times (correlated)
        t1 = {}
        t2 = {}
        for i in registers:
            t1_val = self.draw_t1()
            t2_val = self.draw_t2(t1=t1_val)
            t1[vars_t1(i)] = time(t1_val)
            t2[vars_t2(i)] = time(t2_val)

        return functools.reduce(
            lambda a, b: {**a, **b}, [t1, t2, wq, jqq, deltaq, omegad]
        )


def backend_simulation_vars(backend, angular=False, rabi=False, units=1e9):

    """Qiskit versions < 1.0 only.

    Extract variables for the pulse simulation from the fake backend.

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
    if Version(qiskit_version) > Version("1.0.0"):
        raise Exception("`backend_simulation_vars` requires Qiskit versions < 1.0.0")

    ns_fn = lambda x: x * units
    GHz_fn = lambda x: x / units if angular else x / units / 2 / np.pi
    config = backend.configuration()
    props = backend.properties()
    n = config.n_qubits
    t1 = {vars_t1(i): ns_fn(props.t1(i)) for i in range(n)}
    t2 = {vars_t2(i): ns_fn(props.t2(i)) for i in range(n)}
    H = {k: GHz_fn(v) for k, v in config.hamiltonian["vars"].items()}
    if not rabi:
        for i in range(n):
            H[vars_rabi(i)] = GHz_fn(units)
    return functools.reduce(lambda a, b: {**a, **b}, [H, t1, t2])


def get_channels(registers, backend, allowed_channels=(ControlChannel, DriveChannel)):
    channel_dict = {}
    if Version(qiskit_version) > Version("1.0.0"):
        allowed_channel_names = []
        # Ignore acquire and measure channels
        if ControlChannel in allowed_channels:
            allowed_channel_names.append("control")
        if DriveChannel in allowed_channels:
            allowed_channel_names.append("drive")
        # channels_map
        # {'acquire': {(0,): [AcquireChannel(0)],...}}
        for kind, elems in backend.channels_map.items():
            if kind in allowed_channel_names:
                for qs, chs in elems.items():
                    if set(qs).issubset(registers):
                        for ch in chs:
                            channel_dict[ch.name] = ch
    else:
        for i in registers:
            for ch in backend.configuration().get_qubit_channels(i):
                if isinstance(ch, allowed_channels):
                    channel_dict[ch.name] = ch

    return channel_dict


def get_drive_channel(i, backend, name=False):
    # Standard drive channel
    if Version(qiskit_version) > Version("1.0.0"):
        ch = backend.drive_channel(i)
    else:
        channels = []
        for ch in backend.configuration().get_qubit_channels(i):
            if type(ch) is DriveChannel:
                channels.append(ch)
        if len(channels) != 1:
            raise Exception(f"Invalid number of drive channels for qubit {i}")
        ch = channels[0]

    if name:
        return ch.name
    else:
        return ch


def get_phase_channel(i, backend, name=False):
    # Arbitrary index shift based on device qubits
    if Version(qiskit_version) > Version("1.0.0"):
        n = backend.num_qubits
    else:
        n = backend.configuration().n_qubits
    index_shift = int(10 ** np.ceil(np.log10(n)))

    # Phase is a mock drive channel
    d = DriveChannel(i + index_shift)
    if name:
        return d.name
    else:
        return d


def get_control_channel(c, t, backend, name=False):
    message = f"Backend edge not found for {c} and {t}"
    if Version(qiskit_version) > Version("1.0.0"):
        try:
            ch = backend.control_channel([c, t])[0]
        except IndexError as e:
            raise Exception(message) from e
    else:
        # Cross resonance channel
        try:
            ch = backend.configuration().control_channels[(c, t)][0]
        except KeyError as e:
            raise Exception(message) from e
    if name:
        return ch.name
    else:
        return ch


def backend_carriers(backend, variables):
    # TODO: Why is this function needed?
    if Version(qiskit_version) > Version("1.0.0"):
        raise Exception("`backend_carriers` requires Qiskit versions < 1.0.0")

    n = backend.configuration().n_qubits
    carriers = {
        get_drive_channel(i, backend, name=True): variables[vars_frequency(i)]
        for i in range(n)
    }
    for c, t in get_backend_edges(backend, directed=True):
        uct = get_control_channel(c, t, backend, name=True)
        carriers.update({uct: variables[vars_frequency(t)]})
    return carriers


def get_backend_edges(backend, registers=None, directed=False):
    """Get the unique edges of the provided backend based on allowed two-qubit
    gates.

    Arguments:
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend
        registers (list[int], optional) -- List of qubits. If None, all qubits
            are considered. Defaults to None.
        directed (bool, optional) -- If True, returns directed edges. Defaults to False.

    Returns:
        List of unique edges (as tuples)
    """
    if Version(qiskit_version) > Version("1.0.0"):
        dir_graph = backend.coupling_map
    else:
        dir_graph = backend.configuration().coupling_map

    if not directed:
        dir_graph = np.unique([set(edge) for edge in dir_graph])

    edges = []
    for edge in dir_graph:
        edge = list(edge)
        if registers is not None:
            if edge[0] in registers and edge[1] in registers:
                edges.append(edge)
        else:
            edges.append(edge)

    return edges
