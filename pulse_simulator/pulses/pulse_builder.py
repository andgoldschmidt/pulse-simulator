from numbers import Number
import numpy as np
from qiskit import pulse

from .qiskit_shapes import zero_envelope, constant_envelope

from ..models.qiskit_backend_utils import (
    get_channels,
    get_drive_channel,
    get_control_channel,
    get_phase_channel,
)


VALID_MOMENT_TYPES = ["single", "ecr", "phase"]


def valid_moment_types():
    return VALID_MOMENT_TYPES


def parse_moments(list_of_moments, registers):
    """Parse a list of moments to ensure that they are valid.

    Arguments:
        list_of_moments -- the moments to be parsed
        registers -- the qubits used by the simulation

    Returns:
        A dictionary with the moment types as keys and the number of moments
        of each type as values.
    """
    case_list = {case: 0 for case in VALID_MOMENT_TYPES}
    for moment in list_of_moments:
        check_moment(moment, registers)
        case = get_moment_case(moment)
        case_list[case] += 1
        if case not in VALID_MOMENT_TYPES:
            raise ValueError(f"Invalid moment type: {case}")
    return case_list


def duration_of_moments(list_of_moments, qubit_gates, cr_gates, registers):
    """Get the duration of a list of moments.

    Arguments:
        list_of_moments -- the moments to be parsed
        qubit_gates -- the qubit pulse shapes
        cr_gates -- the signed CR pulses
        registers -- the qubits used by the simulation

    Returns:
        The duration of the moments
    """
    duration = 0
    duration_1qb = qubit_gates[next(iter(qubit_gates))].duration
    duration_2qb = cr_gates[next(iter(cr_gates))].duration
    types_and_durations = {
        "single": duration_1qb,
        "ecr": 2 * duration_2qb + duration_1qb,
        "phase": 1,
    }
    types_and_counts = parse_moments(list_of_moments, registers)
    for case, count in types_and_counts.items():
        duration += count * types_and_durations[case]
    return duration


def build_pulse_schedule(
    list_of_moments, qubit_gates, cr_gates, registers, backend, dt=1.0, name=None
):
    """Build a pulse schedule from a list of moments. The moments themselves are
    dictionaries with qubit labels as keys and gate labels as values. The one qubit
    gates are indexed by integers and the two qubit gates are indexed by tuples.
    The values of the dictionaries are the labels of the pulse shapes to be played,
    or in the case of the two qubit gates, the label of the single qubit gate to be
    use for the echo.

    Moments should be exclusively single qubit or two qubit gates.

    For example, the following is a valid list of moments:
    ```Python
        [
            {0: "X1", 1: "X2", 2: "X1", 3: "X2"},
            {(0,1): "X1", (2,3): "X1"},
            {(1,2): "X2"},
            {0: "X1", 1: "SX2", 2: "SX1", 3: "X2"},
        ]
    ```

    Supported moments are:
    - Single qubit gates `{qubit: gate_label}` of type `{int: str}`
    - Two qubit gates: `{(control, target): gate_label}` of type `{(int, int): str}`
    - Phase gates: `{qubit: angle}` of type `{int: float}`

    The schedule is built by playing the gates in the moments in the order they
    are given. The gates are played on the corresponding channels of the backend.
    Zero envelopes are played on the remaining channels to ensure that the schedule
    respects the ordering of the moments.

    Currently, this function is designed to play X and :math:`\sqrt{X}` gates
    during single qubit moments, and ECR gates during two qubit moments.

    Arguments:
        list_of_moments -- moments to be played in the circuit
        qubit_gates -- key, value pairs containing qubit pulse shapes
        cr_gates -- the keys are + or - and the values are the signed CR pulses
        registers -- the qubits used by the simulation
        backend -- the Qiskit backend (for control channels)

    Keyword Arguments:
        name -- the schedule name (default: {None})
        dt -- simulation time step (see phase moments) (default: {1.0})

    Returns:
        A pulse schedule
    """
    with pulse.build(name=name, backend=backend) as schedule:
        for moment in list_of_moments:
            # Inspect format
            check_moment(moment, registers)

            case = get_moment_case(moment)
            if case == "single":
                play_1qb_moment(moment, qubit_gates, registers, backend)
            elif case == "phase":
                play_phase_moment(moment, dt, registers, backend)
            elif case == "ecr":
                play_ecr_moment(
                    moment, qubit_gates, cr_gates, registers, backend, dt=dt
                )
            else:
                raise ValueError("Invalid moment")
    return schedule


def check_moment(moment, registers):
    keys = list(moment.keys())
    assert len(keys) > 0, "Empty moment"
    played_keys = keys if np.isscalar(keys[0]) else flatten_tuples(keys)
    assert len(played_keys) == len(
        set(played_keys)
    ), f"Multiple gates in {keys} involve the same qubit."
    assert np.all([q in registers for q in played_keys]), f"Invalid qubit in {keys}"


def flatten_tuples(iter_of_tuples):
    """Flatten an iterable of tuples into a list of elements

    Arguments:
        iter_of_tuples -- Iterable of tuples.

    Returns:
        List of elements.
    """
    return list(sum(iter_of_tuples, ()))


def get_moment_case(moment):
    """Get the type of a moment.

    Arguments:
        moment -- the moment to be checked

    Returns:
        "single" if the moment is a single qubit moment, "ecr" if it is a two
        qubit moment, 'phase' if it is a Z gate moment.
    """
    index = next(iter(moment))
    if np.isscalar(index):
        if isinstance(moment[index], Number):
            return "phase"
        else:
            return "single"
    elif len(index) == 2:
        return "ecr"
    else:
        raise ValueError("Invalid moment")


def play_1qb_moment(moment, qubit_gates, registers, backend):
    """To be run within a pulse.build block. Play a single qubit moment.

    Arguments:
        moment -- the moment to be played
        qubit_gates -- the qubit pulse shapes
        registers -- the qubits used by the simulation
        backend -- the Qiskit backend (for control channels)
    """
    played_channels = []
    for qubit, gate_label in moment.items():
        drive_channel = get_drive_channel(qubit, backend)
        pulse.play(qubit_gates[gate_label], drive_channel)
        played_channels.append(drive_channel.name)

    duration = qubit_gates[next(iter(qubit_gates))].duration
    fill_unplayed_phases([], duration, registers, backend)
    fill_unplayed_channels(played_channels, duration, registers, backend)


def play_phase_moment(moment, dt, registers, backend):
    played_qubits = []
    for qubit, angle in moment.items():
        phase_channel = get_phase_channel(qubit, backend)
        pulse.play(constant_envelope(1, angle=angle / dt, name="phase"), phase_channel)
        played_qubits.append(qubit)

    fill_unplayed_phases(played_qubits, 1, registers, backend)
    # No standard channels are used in phase moments
    fill_unplayed_channels([], 1, registers, backend)


def fill_unplayed_channels(played_channel_names, duration, registers, backend):
    """To be run within a pulse.build block. Play zero envelopes on the channels
    that were not used in the moment.

    Arguments:
        played_channel_names -- the channels that were used in the moment
        duration -- the duration of the zero pulse to be played
        registers -- the qubits used by the simulation
        backend -- the Qiskit backend (for control channels)
    """
    channels = get_channels(registers, backend)
    for c in channels.keys():
        if c not in played_channel_names:
            pulse.play(zero_envelope(duration), channels[c])


def fill_unplayed_phases(played_qubits, duration, registers, backend):
    """To be run within a pulse.build block. Play zero envelopes on the phase
    channels that were not used in the moment.

    Arguments:
        played_qubits -- the qubits that were used in the moment
        duration -- the duration of the zero pulse to be played
        registers -- the qubits used by the simulation
        backend -- the Qiskit backend (for control channels)
    """
    for qubit in registers:
        if qubit not in played_qubits:
            phase_channel = get_phase_channel(qubit, backend)
            pulse.play(zero_envelope(duration), phase_channel)


def play_ecr_moment(moment, qubit_gates, cr_gates, registers, backend, dt=1.0):
    """To be run within a pulse.build block. Play a two qubit echo cross resonance
    moment.

    Arguments:
        moment -- the moment to be played
        qubit_gates -- the qubit pulse shapes
        cr_gates -- the signed CR pulses
        registers -- the qubits used by the simulation
        backend -- the Qiskit backend (for control channels)
    """
    # Gate durations
    duration_1qb = qubit_gates[next(iter(qubit_gates))].duration
    duration_2qb = cr_gates[next(iter(cr_gates))].duration

    # +ZX
    played_channels = []
    for qubit, gate_label in moment.items():
        q0, q1 = qubit
        control_channel = get_control_channel(q0, q1, backend)
        pulse.play(cr_gates["+"], control_channel)
        played_channels.append(control_channel.name)
    fill_unplayed_phases([], duration_2qb, registers, backend)
    fill_unplayed_channels(played_channels, duration_2qb, registers, backend)

    # Echo
    played_channels = []
    for qubit, gate_label in moment.items():
        q0, _ = qubit
        drive_channel = get_drive_channel(q0, backend)
        pulse.play(qubit_gates[gate_label], drive_channel)
        played_channels.append(drive_channel.name)
    fill_unplayed_phases([], duration_1qb, registers, backend)
    fill_unplayed_channels(played_channels, duration_1qb, registers, backend)

    # -ZX
    played_channels = []
    for qubit, gate_label in moment.items():
        q0, q1 = qubit
        control_channel = get_control_channel(q0, q1, backend)
        pulse.play(cr_gates["-"], control_channel)
        played_channels.append(control_channel.name)
    fill_unplayed_phases([], duration_2qb, registers, backend)
    fill_unplayed_channels(played_channels, duration_2qb, registers, backend)
