import numpy as np
from qiskit import pulse

from .qiskit_shapes import (
    zero_envelope,
)

from ..models.qiskit_backend_utils import (
    get_channels,
    get_drive_channel,
    get_control_channel,
)


def flatten_tuples(iter_of_tuples):
    """Flatten an iterable of tuples into a list of elements

    Arguments:
        iter_of_tuples -- Iterable of tuples.

    Returns:
        List of elements.
    """
    return list(sum(iter_of_tuples, ()))


def build_pulse_schedule(
    list_of_moments, qubit_gates, cr_gates, registers, backend, name=None
):
    """Build a pulse schedule from a list of moments. The moments themselves are
    dictionaries with qubit labels as keys and gate labels as values. The one qubit
    gates are indexed by integers and the two qubit gates are indexed by tuples.
    Moments should be exclusively single qubit or two qubit gates.

    The schedule is built by playing the gates in the moments in the order they
    are given. The gates are played on the corresponding channels of the backend.
    Zero envelopes are played on the remaining channels to ensure that the schedule
    respects the ordering of the moments.

    Currently, this function plays X and :math:`\sqrt{X}` gates during single qubit
    moments, and ECR gates during two qubit moments.

    Arguments:
        list_of_moments -- _description_
        qubit_gates -- key, value pairs containing qubit pulse shapes
        cr_gates -- the keys are + or - and the values are the signed CR pulses
        registers -- the qubits used by the simulation
        backend -- the Qiskit backend (for control channels)

    Keyword Arguments:
        name -- The schedule name (default: {None})

    Returns:
        A pulse schedule
    """

    with pulse.build(name=name, backend=backend) as schedule:
        for moment in list_of_moments:
            # Check moment structure
            # =====
            keys = list(moment.keys())
            assert len(keys) > 0, "Empty moment"
            played_keys = keys if np.isscalar(keys[0]) else flatten_tuples(keys)
            assert len(played_keys) == len(
                set(played_keys)
            ), f"Multiple gates in {keys} involve the same qubit."
            assert np.all(
                [q in registers for q in played_keys]
            ), f"Invalid qubit in {keys}"

            # Get control data
            # =====
            # Available channels for the backend
            channels = get_channels(registers, backend)
            duration_1qb = qubit_gates[next(iter(qubit_gates))].duration
            duration_2qb = cr_gates[next(iter(cr_gates))].duration

            if np.isscalar(keys[0]):
                # One qubit case
                # =====
                played_channels = []
                for qubit, gate_label in moment.items():
                    drive_channel = get_drive_channel(qubit, backend)
                    pulse.play(qubit_gates[gate_label], drive_channel)
                    played_channels.append(drive_channel.name)
                for c in channels.keys():
                    if c not in played_channels:
                        pulse.play(zero_envelope(duration_1qb), channels[c])
            else:
                # Two qubit case
                # =====
                # +ZX
                played_channels = []
                for qubit, gate_label in moment.items():
                    q0, q1 = qubit
                    control_channel = get_control_channel(q0, q1, backend)
                    pulse.play(cr_gates["+"], control_channel)
                    played_channels.append(control_channel.name)
                for c in channels.keys():
                    if c not in played_channels:
                        pulse.play(zero_envelope(duration_2qb), channels[c])

                # Echo
                played_channels = []
                for qubit, gate_label in moment.items():
                    q0, q1 = qubit
                    drive_channel = get_drive_channel(q0, backend)
                    pulse.play(qubit_gates[gate_label], drive_channel)
                    played_channels.append(drive_channel.name)
                for c in channels.keys():
                    if c not in played_channels:
                        pulse.play(zero_envelope(duration_1qb), channels[c])

                # -ZX
                played_channels = []
                for qubit, gate_label in moment.items():
                    q0, q1 = qubit
                    control_channel = get_control_channel(q0, q1, backend)
                    pulse.play(cr_gates["-"], control_channel)
                    played_channels.append(control_channel.name)
                for c in channels.keys():
                    if c not in played_channels:
                        pulse.play(zero_envelope(duration_2qb), channels[c])
    return schedule
