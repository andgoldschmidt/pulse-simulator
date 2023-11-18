from .qiskit_operator_labels import from_label, to_label
from .qiskit_backend_utils import (
    get_drive_channel,
    vars_frequency,
    vars_rabi,
    vars_t1,
    vars_t2,
)

import numpy as np
import warnings


def qubit_decay_model(qubit, registers, variables):
    """Construct a single qubit error model for pulse gates from t1 and t2
    times. The functionality is similar to the behavior of QuTip pulse.

    Arguments:
        qubit (Int) -- Qubit index
        registers (List[Int]) -- Qubits in circuit
        variables (Dict{Str, Int}) -- Backend configuration properties.

    Keyword arguments:
        t1 [optional] (float) -- Damping decay constant.
        t2 [optional] (float) -- Dephasing decay constant.

    Raises:
        ValueError: Inconsistent t1 and t2.
        Warning: Both t1 and t2 unset.

    Returns:
        List[Static jump operators]
    """
    # Unpack parameters (allow default None)
    t1 = variables.get(vars_t1(qubit), None)
    t2 = variables.get(vars_t2(qubit), None)

    if t1 is None and t2 is None:
        warnings.warn(f"Neither T1 nor T2 is set for Qubit {qubit}.")

    # Construct operator labels
    # Destroy D is an augmented label (not in Qiskit's from_label)
    damp_label = to_label({qubit: "D"}, registers)
    dephase_label = to_label({qubit: "1"}, registers)

    # Construct dissipation operators (QuTip pulse)
    static_op = []
    if t1 is not None:
        static_op += [1 / np.sqrt(t1) * from_label(damp_label)]
    if t2 is not None:
        # Keep the total dephasing ~ exp(-t/t2)
        if t1 is not None:
            if 2 * t1 < t2:
                raise ValueError(f"t1={t1}, t2={t2} does not fulfill 2*t1>t2")
            T2_eff = 1 / (1 / t2 - 1 / 2 / t1)
        else:
            T2_eff = t2
        static_op += [1 / np.sqrt(2 * T2_eff) * 2 * from_label(dephase_label)]

    return static_op


def rx_model(qubit, registers, backend, variables, rotating_frame=False):
    """Construct a single qubit model for pulse gates. This model is
    provided in the lab frame by default.

    Arguments:
        qubit (Int) -- Qubit index
        registers (List[Int]) -- Qubits in circuit
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend
            needed for drive channels.
        variables (Dict{Str, Int}) -- Backend configuration properties.
        rotating_frame (Bool) -- Use the rotating frame. Default false.

    Keyword arguments:
        frequency (float) -- Energy of qubit.

    Returns:
        Drift operator, List[Control operators], List[Drive channels]
    """
    # Unpack parameters
    try:
        w = 2 * np.pi * variables[vars_frequency(qubit)]
        r = 2 * np.pi * variables[vars_rabi(qubit)]
    except Exception as e:
        print(f"Missing required parameter for R_X model on qubit {qubit}.")
        raise e

    # Construct operator labels
    drift_label = to_label({qubit: "Z"}, registers)
    control_label = to_label({qubit: "X"}, registers)

    # Construct Hamiltonian operators
    if rotating_frame:
        drift_op = 0.0  # TODO: Better shaped default
    else:
        drift_op = w / 2 * from_label(drift_label)

    control_op = r * from_label(control_label)

    # Get drive channel
    control_ch = get_drive_channel(qubit, backend, name=True)

    return drift_op, [control_op], [control_ch]
