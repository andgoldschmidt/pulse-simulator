from itertools import combinations
from numbers import Number
import numpy as np
from qiskit import pulse, QuantumCircuit
from qiskit.quantum_info import Operator
import re

from .qiskit_shapes import zero_envelope, constant_envelope

from ..models.qiskit_backend_utils import (
    backend_edges,
    get_channels,
    get_drive_channel,
    get_control_channel,
    get_phase_channel,
)

VALID_GATE_TYPES = ("X", "SX", "I", "ECR")
VALID_MOMENT_TYPES = ("single", "ecr", "phase")


def alternating_coloring(qubit):
    return 1 if qubit % 2 == 0 else 2


def valid_moment_types():
    return VALID_MOMENT_TYPES


def valid_gate_types():
    return VALID_GATE_TYPES


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
    return case_list


def check_moment(moment, registers):
    keys = list(moment.keys())
    assert len(keys) > 0, "Empty moment"
    played_keys = keys if np.isscalar(keys[0]) else flatten_tuples(keys)
    assert len(played_keys) == len(
        set(played_keys)
    ), f"Multiple gates in {keys} involve the same qubit."
    assert np.all([q in registers for q in played_keys]), f"Invalid qubit in {keys}"


def check_gate_label(label, number, allowed_numbers=[1, 2]):
    assert label in VALID_GATE_TYPES, f"Invalid gate label: {label}"
    assert number in allowed_numbers, f"Invalid qubit number: {number}"


def parse_gate_label(label):
    """Parse a gate label into a prefix and a number.

    Arguments:
        label -- the label to be parsed, e.g. SX1

    Returns:
        A tuple with the prefix and the number, e.g. ("SX", "1")
    """
    try:
        label, number = re.match(r"([a-z]+)([0-9]+)", label, re.I).groups()
    except AttributeError:
        raise ValueError(f"Gate label {label} is not prefix and number")
    number = int(number)
    check_gate_label(label, number)
    return label, number


def color_moments(list_of_moments, coloring=alternating_coloring):
    """Color the moments in a list of moments.

    Arguments:
        list_of_moments -- the moments to be colored

    Keyword Arguments:
        coloring -- the coloring rule for the qubits (default: {alternating_coloring})

    Returns:
        The colored moments
    """
    colored_moments = []
    for moment in list_of_moments:
        colored_moment = {}
        case = get_moment_case(moment)
        if case == "phase":
            colored_moment = moment
        elif case == "single":
            for qubit, label in moment.items():
                colored_moment[qubit] = f"{label}{coloring(qubit)}"
        elif case == "ecr":
            for qubits, label in moment.items():
                colored_moment[qubits] = f"{label}{coloring(qubits[0])}"
        colored_moments.append(colored_moment)
    return colored_moments


def uncolor_moments(list_of_moments):
    """Remove the coloring from the moments in a list of moments.

    Arguments:
        list_of_moments -- the moments to be uncolored

    Returns:
        The uncolored moments
    """
    uncolored_moments = []
    for moment in list_of_moments:
        uncolored_moment = {}
        if get_moment_case(moment) == "phase":
            uncolored_moment = moment
        else:
            for qubit, label in moment.items():
                prefix, _ = parse_gate_label(label)
                uncolored_moment[qubit] = prefix
        uncolored_moments.append(uncolored_moment)
    return uncolored_moments


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
    case = None
    if np.isscalar(index):
        if isinstance(moment[index], Number):
            case = "phase"
        else:
            case = "single"
    elif len(index) == 2:
        case = "ecr"
    else:
        raise ValueError("Could not parse moment case.")

    if case not in VALID_MOMENT_TYPES:
        return f"Invalid moment type: {case}"
    else:
        return case


def durations_of_moments(list_of_moments, duration_1qb, duration_2qb):
    """Get the list of durations corresponding to a list of moments.

    Can be combined with accumulate to get the timestep after each moment.

    Arguments:
        list_of_moments -- the moments to be parsed
        qubit_gates -- the qubit pulse shapes
        cr_gates -- the signed CR pulses

    Returns:
        The list of durations of the moments
    """
    durations = []
    types_and_durations = {
        "single": duration_1qb,
        "ecr": 2 * duration_2qb + duration_1qb,
        "phase": 1,
    }
    for moment in list_of_moments:
        case = get_moment_case(moment)
        durations.append(types_and_durations[case])
    return durations


def build_pulse_schedule(
    list_of_moments,
    qubit_gates,
    cr_gates,
    registers,
    backend,
    dt=1.0,
    name=None,
    insert_identities=True,
    coloring=alternating_coloring,
    verbose=True,
):
    """Build a pulse schedule from a list of moments. The moments themselves are
    dictionaries with qubit labels as keys and gate labels as values. The one qubit
    gates are indexed by integers and the two qubit gates are indexed by tuples.
    The values of the dictionaries are the labels of the pulse shapes to be played,
    or in the case of the two qubit gates, the label ECR.

    Moments should be exclusively single qubit or two qubit gates.

    For example, the following is a valid list of moments:
    ```Python
        [
            {0: "X", 1: "X", 2: "X", 3: "X"},
            {(0,1): "ECR", (2,3): "ECR"},
            {(1,2): "ECR"},
            {0: "X", 1: "SX", 2: "SX", 3: "X"},
        ]
    ```
    In this case, the coloring function will be used to determine the colored pulses.
    The coloring function should take a qubit index and return an integer that is
    used to index the pulse shape types for that qubit. The default coloring function
    is `default_coloring`, which colors the qubits alternately.

    Optionally, `insert_identities` is a flag that determines whether identity gates
    should be inserted for qubits that are not used in a moment. This is useful for ECR
    gates, where an identity can be inserted for the qubit that is not being used in the
    echo. The coloring function is used to determine the identity gates to be played.

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
        coloring -- the coloring rule for the qubits (default: {default_coloring})

    Returns:
        A pulse schedule
    """
    if verbose:
        print(f"Building schedule with {len(list_of_moments)} moments. ")
        print(f"\t Qubit gateset: {qubit_gates.keys()}")
        print(f"\t CR gateset: {cr_gates.keys()}")
        print(f"\t Registers: {registers}")
        print(f"\t Insert identities: {insert_identities}")

    with pulse.build(name=name, backend=backend) as schedule:
        for moment in list_of_moments:
            # Inspect format
            check_moment(moment, registers)
            case = get_moment_case(moment)
            if case == "single":
                play_1qb_moment(
                    moment, qubit_gates, registers, coloring, insert_identities, backend
                )
            elif case == "phase":
                play_phase_moment(moment, dt, registers, backend)
            elif case == "ecr":
                play_ecr_moment(
                    moment,
                    qubit_gates,
                    cr_gates,
                    registers,
                    coloring,
                    insert_identities,
                    backend,
                    dt=dt,
                    verbose=verbose,
                )
            else:
                raise ValueError("Invalid moment")
    return schedule


def play_1qb_moment(
    moment, qubit_gates, registers, coloring, insert_identities, backend
):
    """To be run within a pulse.build block. Play a single qubit moment.

    Arguments:
        moment -- the moment to be played
        qubit_gates -- the qubit pulse shapes
        registers -- the qubits used by the simulation
        coloring -- the coloring rule for the qubits
        insert_identities -- whether to insert identity gates for unplayed qubits
        backend -- the Qiskit backend (for control channels)
    """
    played_channels = []
    played_qubits = []
    for qubit, gate_label in moment.items():
        drive_channel = get_drive_channel(qubit, backend)
        pulse.play(qubit_gates[f"{gate_label}{coloring(qubit)}"], drive_channel)
        played_qubits.append(qubit)
        played_channels.append(drive_channel.name)

    duration = qubit_gates[next(iter(qubit_gates))].duration
    fill_unplayed_phases([], duration, registers, backend)
    # Insert identities, or fill remaining unplayed channels
    if insert_identities:
        played_channels += fill_unplayed_single_qubit_gates(
            played_qubits, qubit_gates, registers, coloring, backend
        )
    fill_unplayed_channels(played_channels, duration, registers, backend)


def play_phase_moment(moment, dt, registers, backend):
    played_qubits = []
    for qubit, angle in moment.items():
        phase_channel = get_phase_channel(qubit, backend)
        # R_Z gates are played with 1/2 the value
        pulse.play(
            constant_envelope(1, angle=angle / 2 / dt, name="phase"), phase_channel
        )
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


def fill_unplayed_single_qubit_gates(
    played_qubits, qubit_gates, registers, coloring, backend
):
    """
    To be run within a pulse.build block. Play identity gates on any registers that
    were not used in the moment. Use the coloring to determine the correct identity
    gate to use.

    Arguments:
        played_qubits -- the qubits that were used in the moment
        qubit_gates -- the qubit pulse shapes
        registers -- the qubits used by the simulation
        coloring -- the coloring rule for the qubits
        backend -- the Qiskit backend (for control channels)

    Returns:
        A list of the channels that were played
    """
    played_channels = []
    for q in registers:
        if q not in played_qubits:
            drive_channel = get_drive_channel(q, backend)
            pulse.play(qubit_gates[f"I{coloring(q)}"], drive_channel)
            played_channels.append(drive_channel.name)
    return played_channels


def has_edge_with(q, partners, edge_set):
    return np.any([set((q, p)) in edge_set for p in partners])


def fill_unplayed_dd_gates(
    played_channels,
    played_pairs,
    qubit_gates,
    duration_1qb,
    duration_2qb,
    coloring,
    registers,
    backend,
    verbose=False,
):
    """
    To be run within a pulse.build block. Play dynamical decoupling (DD) sequences
    on idle pairs of qubits. The DD sequence is centered in the middle of the 2QB
    gate. In this way, the DD sequence evenly accumulates and deaccumulates coherent
    errors from the crosstalk.

    The ECR echo will not affect the DD sequence, but the target rotary can; therefore,
    we do not play a DD sequence on a target spectator.

    TODO:
    This is not an optimal DD approach. For example, consider:
    >    Idle = [0, 1, 2], Played = [(4, 3)]
    >    If we resolve the (0,1) pair first: Selecting [0] or [1] for
    >    DD sequences is valid, but only [1] will work; after [0], the
    >    pair (1,2) are both next to an active neighbor.

    Arguments:
        played_channels -- the channels that were used in the moment
        played_pairs -- the qubit pairs that were used in the moment
        qubit_gates -- the qubit pulse shapes
        duration_1qb -- the duration of the single qubit gates
        duration_2qb -- the duration of the two qubit gates
        coloring -- the coloring rule for the qubits
        registers -- the qubits used by the simulation
        backend -- the Qiskit backend (for control channels)

    Keyword Arguments:
        verbose -- whether to print information about the selected qubits (default: {False})
    """
    if duration_2qb - duration_1qb <= 0:
        raise ValueError("DD Error: 2QB duration must be greater than 1QB duration.")

    # Get idle qubits
    idles = [q for q in registers if q not in flatten_tuples(played_pairs)]

    # Center the DD sequence
    duration_idle1 = int((duration_2qb - duration_1qb) / 2)
    duration_idle2 = duration_2qb - duration_1qb - duration_idle1
    fill_unplayed_channels(played_channels, duration_idle1, registers, backend)

    # Play DD sequence on empty pairs of qubits
    dd_qubits = []
    dd_channels = []
    target_qubits = [q2 for _, q2 in played_pairs]
    edges = [set(e) for e in backend_edges(backend)]
    for pair in combinations(idles, 2):
        if set(pair) in edges:
            q1, q2 = pair
            if q2 in dd_qubits or q1 in dd_qubits:
                # DD not needed if a DD sequence already exists.
                continue
            else:
                # Do not play a DD sequence on a target spectator or next to existing DD.
                active_neighbors = target_qubits + dd_qubits
                if not has_edge_with(q2, active_neighbors, edges):
                    qd = q2
                elif not has_edge_with(q1, active_neighbors, edges):
                    qd = q1
                else:
                    raise ValueError(
                        f"DD Error: Idle pair ({q1, q2}) are next to active neighbors (target or DD)."
                    )

                dd_channel = get_drive_channel(qd, backend)
                pulse.play(qubit_gates[f"X{coloring(qd)}"], dd_channel)
                dd_channels.append(dd_channel.name)
                dd_qubits.append(qd)

    # Print selected qubits
    if verbose and len(dd_qubits) > 0:
        print(f"\tSelcting {dd_qubits} for DD sequences when playing {played_pairs}.")

    # Fill remaining time with idle
    fill_unplayed_channels(played_channels, duration_idle2, registers, backend)
    fill_unplayed_channels(
        played_channels + dd_channels,
        duration_1qb,
        registers,
        backend,
    )


def play_ecr_moment(
    moment,
    qubit_gates,
    cr_gates,
    registers,
    coloring,
    insert_identities,
    backend,
    dt=1.0,
    verbose=False,
):
    """To be run within a pulse.build block. Play a two qubit echo cross resonance
    moment.

    TODO: Currently the gate label is unused because there are not yet different types
    of cross resonance pulses.

    Arguments:
        moment -- the moment to be played
        qubit_gates -- the qubit pulse shapes
        cr_gates -- the signed CR pulses
        registers -- the qubits used by the simulation
        coloring -- the coloring rule for the qubits
        insert_identities -- whether to insert identity gates for unplayed qubits
        backend -- the Qiskit backend (for control channels)
    """
    # Gate durations
    duration_1qb = qubit_gates[next(iter(qubit_gates))].duration
    duration_2qb = cr_gates[next(iter(cr_gates))].duration

    # +ZX
    played_pairs = []
    played_channels = []
    for qubit, _ in moment.items():
        q0, q1 = qubit
        control_channel = get_control_channel(q0, q1, backend)
        pulse.play(cr_gates["+"], control_channel)
        played_channels.append(control_channel.name)
        played_pairs.append(qubit)
    fill_unplayed_phases([], duration_2qb, registers, backend)
    if insert_identities:
        fill_unplayed_dd_gates(
            played_channels,
            played_pairs,
            qubit_gates,
            duration_1qb,
            duration_2qb,
            coloring,
            registers,
            backend,
            verbose=verbose,
        )
    else:
        fill_unplayed_channels(played_channels, duration_2qb, registers, backend)

    # Echo
    played_qubits = []
    played_channels = []
    for qubit, label in moment.items():
        q0, _ = qubit
        drive_channel = get_drive_channel(q0, backend)
        pulse.play(qubit_gates[f"X{coloring(q0)}"], drive_channel)
        played_qubits.append(q0)
        played_channels.append(drive_channel.name)
    fill_unplayed_phases([], duration_1qb, registers, backend)
    if insert_identities:
        played_channels += fill_unplayed_single_qubit_gates(
            played_qubits, qubit_gates, registers, coloring, backend
        )
    fill_unplayed_channels(played_channels, duration_1qb, registers, backend)

    # -ZX
    played_pairs = []
    played_channels = []
    for qubit, _ in moment.items():
        q0, q1 = qubit
        control_channel = get_control_channel(q0, q1, backend)
        pulse.play(cr_gates["-"], control_channel)
        played_channels.append(control_channel.name)
        played_pairs.append(qubit)
    fill_unplayed_phases([], duration_2qb, registers, backend)
    if insert_identities:
        fill_unplayed_dd_gates(
            played_channels,
            played_pairs,
            qubit_gates,
            duration_1qb,
            duration_2qb,
            coloring,
            registers,
            backend,
            verbose=False,
        )
    else:
        fill_unplayed_channels(played_channels, duration_2qb, registers, backend)


def qiskit_gate_dispatch(label, q0, circuit, q1=None):
    """Dispatch a gate to the Qiskit circuit.

    Arguments:
        label -- the label of the gate
        q0 -- the qubit to be acted upon
        circuit -- the Qiskit circuit

    Keyword Arguments:
        q1 -- the second qubit (default: {None})

    Raises:
        ValueError: if the gate label is unknown

    Returns:
        The Qiskit circuit with the gate applied
    """
    if label not in VALID_GATE_TYPES:
        raise ValueError(f"Invalid gate label: {label}")

    if label == "X":
        return circuit.x(q0)
    elif label == "SX":
        return circuit.sx(q0)
    elif label == "I":
        return circuit.id(q0)
    elif label == "ECR":
        return circuit.ecr(q0, q1)
    else:
        raise ValueError(f"No dispatch for gate label: {label}")


def build_circuit(
    moments, registers, includes_coloring=False, insert_identities=False, barriers=False
):
    """Build a Qiskit circuit from a list of moments.

    Arguments:
        moments -- the moments to be played in the circuit
        registers -- the qubits used by the simulation

    Raises:
        ValueError: if the moment type is unknown

    Returns:
        A Qiskit circuit with the moments applied
    """
    circ = QuantumCircuit(len(registers))

    for moment in moments:
        case = get_moment_case(moment)
        if case == "single":
            played_qubits = []
            for q0, label in moment.items():
                if includes_coloring:
                    prefix = parse_gate_label(label)[0]
                else:
                    prefix = label
                played_qubits.append(q0)
                qiskit_gate_dispatch(prefix, q0, circ)
            if insert_identities:
                for q0 in registers:
                    if q0 not in played_qubits:
                        qiskit_gate_dispatch("I", q0, circ)
        elif case == "phase":
            for q0, value in moment.items():
                circ.rz(value, q0)
        elif case == "ecr":
            for (q0, q1), label in moment.items():
                qiskit_gate_dispatch(label="ECR", q0=q0, q1=q1, circuit=circ)
        else:
            raise ValueError(f"Unknown case: {case}")
        if barriers:
            circ.barrier()
    return circ


def build_operator(moments, registers):
    """Build a Qiskit operator from a list of moments.

    Arguments:
        moments -- the moments to be played in the circuit
        registers -- the qubits used by the simulation

    Returns:
        A Qiskit operator with the moments applied
    """
    return Operator(build_circuit(moments, registers))
