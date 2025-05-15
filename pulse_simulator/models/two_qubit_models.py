from .qiskit_operator_labels import from_label, to_label, zero_operator
from .qiskit_backend_utils import (
    BackendSampler,
    vars_anharmonicity,
    vars_coupling,
    vars_frequency,
    vars_rabi,
    get_control_channel,
    get_drive_channel,
)

import numpy as np

# Types
from typing import Union


def zz_coupling(edge, variables, sampler: Union[BackendSampler, None] = None):
    """Return the crosstalk coupling amount.

    Arguments:
        edge -- tuple
        qubit_props (Dict{Str, Number}) -- Qubit properties dictionary.
        coupling_props (Dict{Str, Number}) -- Coupling properties dictionary.

    Returns:
        Value of ZZ coupling for edge
    """
    i1, i2 = edge
    if sampler is not None:
        # Need units of simulation
        J12 = sampler.frequency_units(sampler.draw_coupling(mean=True), angular=True)
        α = sampler.frequency_units(sampler.draw_anharmonicity(mean=True), angular=True)
        Δ12 = sampler.frequency_units(sampler.draw_frequency(std=True), angular=True)
    else:
        try:
            J12 = 2 * np.pi * variables[vars_coupling(i1, i2)]
            α1 = 2 * np.pi * variables[vars_anharmonicity(i1)]
            α2 = 2 * np.pi * variables[vars_anharmonicity(i2)]
            # Edges are assumed to be undirected
            α = (α1 + α2) / 2
            ω1 = 2 * np.pi * variables[vars_frequency(i1)]
            ω2 = 2 * np.pi * variables[vars_frequency(i2)]
            Δ12 = ω1 - ω2
        except Exception as e:
            print(f"Missing required parameter for crosstalk edge {edge}.")
            raise e

    # Carries units of frequency
    return 2 * α * J12**2 / (Δ12**2 - α**2)


def crosstalk_model(
    registers,
    graph,
    variables,
    sampler: Union[BackendSampler, None] = None,
    qiskit_order=True,
):
    """The crosstalk Hamiltonian of the circuit. The crosstalk is limited
    to the active registers provided, even if the graph includes additional
    edges.

    Arguments:
        registers -- The allowed qubits from the backend.
        graph (List[Tuple(Int, Int)]) -- Undirected edge list
        variables (Dict{Str, Int}) -- Backend configuration properties.
        qiskit_order (Bool) -- Use Qiskit's endianness. Default True.

    Returns:
        Operator of crosstalk
    """
    operator = zero_operator(len(registers))
    for ZZ_edge in graph:
        if ZZ_edge[0] in registers and ZZ_edge[1] in registers:
            ZZ_value = zz_coupling(ZZ_edge, variables, sampler=sampler)
            ZZ_label = to_label({i: "Z" for i in ZZ_edge}, registers)
            operator += ZZ_value * from_label(ZZ_label, reverse=qiskit_order)
    return operator


def calibrate_cross_resonance_drive(
    qubits, variables, model_name="Toy", angle=np.pi / 8
):
    i_c, i_t = qubits
    try:
        J = 2 * np.pi * variables[vars_coupling(i_c, i_t)]
        αc = 2 * np.pi * variables[vars_anharmonicity(i_c)]
        ωc = 2 * np.pi * variables[vars_frequency(i_c)]
        ωt = 2 * np.pi * variables[vars_frequency(i_t)]
        Δct = ωc - ωt
    except Exception as e:
        print(f"Missing parameter for CR model on {qubits}.")
        raise e

    if "SWPT" in model_name:
        zx_coef = J / (Δct + αc) * (αc / Δct)
    elif "Simple" in model_name:
        zx_coef = J / Δct
    elif "Toy" in model_name:
        zx_coef = 1.0
    else:
        raise ValueError(f"Unknown CR gate model with name {model_name}")

    # Return the drive calibration (the area under the pulse)
    return angle / zx_coef


# TODO: This is a very hacky function default.
def calibrate_default_cross_resonance_drive(
    model_name: str, angle=np.pi / 8, sampler: BackendSampler = BackendSampler(1e9)
):
    # returned value will cancel units & 2π's -- conversion is not strictly necessary
    J = sampler.frequency_units(sampler.draw_coupling(mean=True), angular=True)
    αc = sampler.frequency_units(sampler.draw_anharmonicity(mean=True), angular=True)
    Δct = sampler.frequency_units(sampler.draw_frequency(std=True), angular=True)

    if "SWPT" in model_name:
        zx_coef = J / (Δct + αc) * (αc / Δct)
    elif "Simple" in model_name:
        zx_coef = J / Δct
    elif "Toy" in model_name:
        zx_coef = 1.0
    else:
        raise ValueError(f"Unknown CR gate model with name {model_name}")

    # Return the drive calibration (the area under the pulse)
    return angle / zx_coef


def cross_resonance_model(
    qubits,
    registers,
    backend,
    variables,
    model_name="Toy",
    sampler: Union[BackendSampler, None] = None,
    return_params=False,
    qiskit_order=True,
):
    """Construct a two qubit model for pulse CR gates.

    Note: These models keep ZZ coupling separate! Any ZZ coupling should be
    added to the drift Hamiltonian via the crosstalk model.

    Note: The default device parameters enable a single calibrated 2QB gate, instead of
    using the backend parameters, which would require a separate calibration for each pair
    of qubits.

    Arguments:
        qubits (List[Int]) -- The two qubits, ordered as (control, target).
        registers (List[Int]) -- Qubits in circuit
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend for
            control channels.
        variables (Dict{Str, Int}) -- Backend configuration properties.
        name (Str) -- The name of the model("SWPT", "Simple", "Toy").
            Default is "Toy".
        qiskit_order (Bool) -- Use Qiskit's endianness. Default false.

    Returns:
        Drift operator, List[Control operators], List[Drive channels]
    """
    # Parameters dictionary
    params = {}

    # Unpack qubits
    i_c, i_t = qubits

    # Get channel labels
    cr_drive_l = get_control_channel(i_c, i_t, backend, name=True)
    targ_drive_l = get_drive_channel(i_t, backend, name=True)
    ctrl_drive_l = get_drive_channel(i_c, backend, name=True)

    # Unpack parameters
    if sampler is not None:
        J = sampler.frequency_units(sampler.draw_coupling(mean=True), angular=True)
        αc = sampler.frequency_units(
            sampler.draw_anharmonicity(mean=True), angular=True
        )
        Δct = sampler.frequency_units(sampler.draw_frequency(std=True), angular=True)
        rc = sampler.frequency_units(sampler.draw_rabi(mean=True), angular=True)
        rt = sampler.frequency_units(sampler.draw_rabi(mean=True), angular=True)
    else:
        try:
            J = 2 * np.pi * variables[vars_coupling(i_c, i_t)]
            αc = 2 * np.pi * variables[vars_anharmonicity(i_c)]
            ωc = 2 * np.pi * variables[vars_frequency(i_c)]
            ωt = 2 * np.pi * variables[vars_frequency(i_t)]
            rc = 2 * np.pi * variables[vars_rabi(i_c)]
            rt = 2 * np.pi * variables[vars_rabi(i_t)]
            Δct = ωc - ωt
        except Exception as e:
            print(f"Missing parameter for CR model on {qubits}.")
            raise e

    # Unitless coefficients
    c1 = J / (Δct + αc)
    c2 = αc / Δct
    c3 = J / Δct

    # Construct operator labels
    ZI_label = to_label({i_c: "Z", i_t: "I"}, registers)
    XI_label = to_label({i_c: "X", i_t: "I"}, registers)
    IX_label = to_label({i_c: "I", i_t: "X"}, registers)
    ZX_label = to_label({i_c: "Z", i_t: "X"}, registers)

    # Construct Hamiltonian operators
    Zero = zero_operator(len(registers))
    ZI = from_label(ZI_label, reverse=qiskit_order)
    XI = from_label(XI_label, reverse=qiskit_order)
    IX = from_label(IX_label, reverse=qiskit_order)
    ZX = from_label(ZX_label, reverse=qiskit_order)
    if model_name == "SWPT":
        # Schriefer-Wolff perturbation theory
        params = {
            "Model": f"{model_name}",
            "Drift": "0",
            f"{cr_drive_l}": f"{c1: .2e} * (-I_{i_c} X_{i_t} + {c2: .2e} * Z_{i_c} X_{i_t})",
            f"{targ_drive_l}": f"{rt} * X_{i_t}",
            f"{ctrl_drive_l}": f"{rc} * X_{i_c}",
        }

        drift_op = Zero
        cr_drive_op = c1 * (-IX + c2 * ZX)
        targ_drive_op = rt * IX
        ctrl_drive_op = rc * XI

        control_ops = [cr_drive_op, ctrl_drive_op, targ_drive_op]
        control_channels = [cr_drive_l, ctrl_drive_l, targ_drive_l]

    elif model_name == "Simple":
        # Simple all-μwave entangling gate for fixed-frequency SC qubits
        params = {
            "Model": f"{model_name}",
            "Drift": f"{Δct} * ZI",
            f"{cr_drive_l}": f"X_{i_c} I_{i_t} + {c3: .2e} * Z_{i_c} X_{i_t}",
            f"{targ_drive_l}": f"{rt: .2e} * X_{i_t}",
            f"{ctrl_drive_l}": f"{rc: .2e} * X_{i_c}",
        }

        drift_op = Δct * ZI
        cr_drive_op = XI + c3 * ZX
        ctrl_drive_op = rc * XI
        targ_drive_op = rt * IX

        control_ops = [cr_drive_op, ctrl_drive_op, targ_drive_op]
        control_channels = [cr_drive_l, ctrl_drive_l, targ_drive_l]

    elif model_name == "Toy":
        rct = 1.0
        params = {
            "Model": f"{model_name}",
            "Drift": "0",
            f"{cr_drive_l}": f"{rct: .2e} Z_{i_c} X_{i_t}",
            f"{targ_drive_l}": f"{rt: .2e} * X_{i_t}",
            f"{ctrl_drive_l}": f"{rc: .2e} * X_{i_c}",
        }

        drift_op = Zero
        cr_drive_op = rct * ZX
        targ_drive_op = rt * IX
        ctrl_drive_op = rc * XI

        control_ops = [cr_drive_op, ctrl_drive_op, targ_drive_op]
        control_channels = [cr_drive_l, ctrl_drive_l, targ_drive_l]

    else:
        raise ValueError(f"Unknown CR gate model with name {model_name}")

    if return_params:
        return drift_op, control_ops, control_channels, params
    else:
        return drift_op, control_ops, control_channels


def get_ix_coef(simlator, sampler, J_factor=1.0):
    J = J_factor * simlator._backend_sampler.frequency_units(sampler.draw_coupling(mean=True), angular=True)
    αc = simlator._backend_sampler.frequency_units(sampler.draw_anharmonicity(mean=True), angular=True)
    Δct = simlator._backend_sampler.frequency_units(sampler.draw_frequency(std=True), angular=True)
    return -J / (Δct + αc)


def get_zz_coef(simlator, sampler, J_factor=1.0):
    J = J_factor * simlator._backend_sampler.frequency_units(sampler.draw_coupling(mean=True), angular=True)
    αc = simlator._backend_sampler.frequency_units(sampler.draw_anharmonicity(mean=True), angular=True)
    Δct = simlator._backend_sampler.frequency_units(sampler.draw_frequency(std=True), angular=True)
    return 2 * αc * J**2 / (Δct**2 - αc**2) # Note units!