from .qiskit_operator_labels import from_label, to_label
from .qiskit_backend_utils import (
    get_drive_channel,
    get_phase_channel,
    vars_frequency,
    vars_rabi,
    vars_detuning,
    vars_t1,
    vars_t2,
)

import numpy as np
import warnings


def qubit_decay_model(qubit, registers, variables, qiskit_order=True):
    """Construct a single qubit error model for pulse gates from t1 and t2
    times. The functionality is similar to the behavior of QuTip pulse.

    Linblad master equation:
     d_t ρ = -i[H, ρ] + ∑_k γ_k (A_k ρ A_k† - 1/2 {A_k† A_k, ρ})

    with:
        γ_k = √(1/T_k)
        A_k = jump operators
        T_k = decay times

    Damping (T_1):
        A_k = damp
        T_k = T_1

    Dephasing (T_2):
        A_k = 2 * dephase
        T_k = 2 T_eff
        Εquivalently, d_t ρ = √(1/2T_eff) (Z ρ Z - ρ = Z ρ Z - 1/2 {Z† Z, ρ})

    Arguments:
        qubit (Int) -- Qubit index
        registers (List[Int]) -- Qubits in circuit
        variables (Dict{Str, Int}) -- Backend configuration properties.
        qiskit_order (Bool) -- Use Qiskit's endianness. Default True.

    Keyword variables:
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

    # Construct dissipation operators (c.f. QuTip's pulse class)
    static_op = []
    if t1 is not None:
        static_op += [1 / np.sqrt(t1) * from_label(damp_label, reverse=qiskit_order)]
    if t2 is not None:
        # Keep the total dephasing ~ exp(-t/t2)
        if t1 is not None:
            if 2 * t1 < t2:
                raise ValueError(f"t1={t1}, t2={t2} does not fulfill 2*t1>t2")
            t2_eff = 1 / (1 / t2 - 1 / 2 / t1)
        else:
            t2_eff = t2

        if t2_eff < 0.0:
            warnings.warn(f"Bad 2*t1<t2 occurred, t2_eff={t2_eff}.")
        static_op += [
            1
            / np.sqrt(2 * t2_eff)
            * 2
            * from_label(dephase_label, reverse=qiskit_order)
        ]

    return static_op


def rx_model(
    qubit,
    registers,
    backend,
    variables,
    rotating_frame=False,
    return_params=False,
    qiskit_order=True,
):
    """Construct a single qubit model for pulse gates. This model is
    provided in the lab frame by default.

    TODO: remove dependence on backend by handling channels via a lookup

    Arguments:
        qubit (Int) -- Qubit index
        registers (List[Int]) -- Qubits in circuit
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend
            needed for drive channels.
        variables (Dict{Str, Int}) -- Backend configuration properties.
        rotating_frame (Bool) -- Use the rotating frame. Default false.
        qiskit_order (Bool) -- Use Qiskit's endianness. Default True.

    Keyword variables:
        frequency (float) -- Energy of qubit.

    Returns:
        Drift operator, List[Control operators], List[Drive channels]
    """
    # Unpack parameters
    try:
        w = 2 * np.pi * variables[vars_frequency(qubit)]
        r = 2 * np.pi * variables[vars_rabi(qubit)]
        d = variables[vars_detuning(qubit)]
    except Exception as e:
        print(f"Missing required parameter for R_X model on qubit {qubit}.")
        raise e

    params = {}

    # Construct operator labels
    drift_label = to_label({qubit: "Z"}, registers)
    control_label = to_label({qubit: "X"}, registers)

    # Get drift
    if rotating_frame:
        w_rot = 0.0
        drift_op = w_rot * from_label(drift_label, reverse=qiskit_order)
        params["Drift"] = f"{w_rot: .2e} * Z_{qubit}"
    else:
        drift_op = w / 2 * from_label(drift_label, reverse=qiskit_order)
        params["Drift"] = f"{w / 2: .2e} * Z_{qubit}"

    # Get detuning
    drift_op += d * from_label(drift_label, reverse=qiskit_order)
   
    # Get drive
    control_ch = get_drive_channel(qubit, backend, name=True)
    control_op = r * from_label(control_label, reverse=qiskit_order)
    params[f"{control_ch}"] = f"{r: .2e} * X_{qubit}"

    if return_params:
        return drift_op, [control_op], [control_ch], params
    else:
        return drift_op, [control_op], [control_ch]


def rz_model(
    qubit,
    registers,
    backend,
    return_params=False,
    qiskit_order=True,
):
    """Construct Z phase rotation gates.

    Arguments:
        qubit (Int) -- Qubit index
        registers (List[Int]) -- Qubits in circuit
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend
            needed for drive channels.
        qiskit_order (Bool) -- Use Qiskit's endianness. Default True.

    Keyword variables:
        frequency (float) -- Energy of qubit.

    Returns:
        Drift operator, List[Control operators], List[Drive channels]
    """
    # Construct operator labels
    phase_label = to_label({qubit: "Z"}, registers)
    phase_op = from_label(phase_label, reverse=qiskit_order)
    phase_ch = get_phase_channel(qubit, backend, name=True)
    params = {phase_ch: f"Z_{qubit}"}

    if return_params:
        return phase_op, phase_ch, params
    else:
        return phase_op, phase_ch
