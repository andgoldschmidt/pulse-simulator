from .qiskit_operator_labels import from_label, to_label, zero_operator
from .qiskit_backend_utils import (
    get_drive_channel,
    get_control_channel,
    vars_anharmonicity,
    vars_coupling,
    vars_frequency,
    vars_rabi,
)

import numpy as np


def zz_coupling(edge, variables):
    """Return the crosstalk coupling amount.

    Arguments:
        edge -- tuple
        qubit_props (Dict{Str, Number}) -- Qubit properties dictionary.
        coupling_props (Dict{Str, Number}) -- Coupling properties dictionary.

    Returns:
        Value of ZZ coupling for edge
    """
    i1, i2 = edge
    try:
        α1 = 2 * np.pi * variables[vars_anharmonicity(i1)]
        α2 = 2 * np.pi * variables[vars_anharmonicity(i2)]
        # Edges are assumed to be undirected
        α = (α1 + α2) / 2
        J12 = 2 * np.pi * variables[vars_coupling(i1, i2)]
        ω1 = 2 * np.pi * variables[vars_frequency(i1)]
        ω2 = 2 * np.pi * variables[vars_frequency(i2)]
    except Exception as e:
        print(f"Missing required parameter for crosstalk edge {edge}.")
        raise e

    Δ12 = ω1 - ω2
    return -2 * α * J12**2 / (Δ12**2 - α**2)


def crosstalk_model(registers, graph, variables):
    """The crosstalk Hamiltonian of the circuit. The crosstalk is limited
    to the active registers provided, even if the graph includes additional
    edges.

    Arguments:
        registers -- The allowed qubits from the backend.
        graph (List[Tuple(Int, Int)]) -- Undirected edge list
        variables (Dict{Str, Int}) -- Backend configuration properties.

    Returns:
        Operator of crosstalk
    """
    operator = 0.0
    for ZZ_edge in graph:
        if ZZ_edge[0] in registers and ZZ_edge[1] in registers:
            ZZ_value = zz_coupling(ZZ_edge, variables)
            ZZ_label = to_label({i: "Z" for i in ZZ_edge}, registers)
            operator += ZZ_value * from_label(ZZ_label)
    return operator


def cross_resonance_model(
    qubits, registers, backend, variables, model_name="Toy", return_params=False
):
    """Construct a two qubit model for pulse CR gates.

    Note: These models keep ZZ coupling separate! Any ZZ coupling should be
    added to the drift Hamiltonian via the crosstalk model.

    Arguments:
        qubits (List[Int]) -- The two qubits, ordered as (control, target).
        registers (List[Int]) -- Qubits in circuit
        backend (qk.providers.fake_provider.FakePulseBackend) -- Backend for
            control channels.
        variables (Dict{Str, Int}) -- Backend configuration properties.
        name (Str) -- The name of the model("SWPT", "Simple", "Toy").
            Default is "Toy".

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
    try:
        J = 2 * np.pi * variables[vars_coupling(i_c, i_t)]
        αc = 2 * np.pi * variables[vars_anharmonicity(i_c)]
        αt = 2 * np.pi * variables[vars_anharmonicity(i_c)]
        ωc = 2 * np.pi * variables[vars_frequency(i_c)]
        ωt = 2 * np.pi * variables[vars_frequency(i_t)]
        rc = 2 * np.pi * variables[vars_rabi(i_c)]
        rt = 2 * np.pi * variables[vars_rabi(i_t)]
    except Exception as e:
        print(f"Missing parameter for CR model on {qubits}.")
        raise e

    # Modify params
    Δct = ωc - ωt
    α = (αc + αt) / 2

    # Construct operator labels
    ZI_label = to_label({i_c: "Z", i_t: "I"}, registers)
    XI_label = to_label({i_c: "X", i_t: "I"}, registers)
    IX_label = to_label({i_c: "I", i_t: "X"}, registers)
    ZX_label = to_label({i_c: "Z", i_t: "X"}, registers)

    # Construct Hamiltonian operators
    Zero = zero_operator(len(registers))
    ZI = from_label(ZI_label)
    XI = from_label(XI_label)
    IX = from_label(IX_label)
    ZX = from_label(ZX_label)
    if model_name == "SWPT":
        # Schriefer-Wolff perturbation theory
        params = {
            "Model": f"{model_name}",
            "Drift": "0",
            f"{cr_drive_l}": f"{J / (Δct + α): .2e} * (I_{i_c} X_{i_t} + {(α / Δct): .2e} * Z_{i_c} X_{i_t})",
            f"{targ_drive_l}": f"{rt} * X_{i_t}",
            f"{ctrl_drive_l}": f"{rc} * X_{i_c}",
        }

        drift_op = Zero
        cr_drive_op = J / (Δct + α) * (IX + (α / Δct) * ZX)
        targ_drive_op = rt * IX
        ctrl_drive_op = rc * XI

        control_ops = [cr_drive_op, ctrl_drive_op, targ_drive_op]
        control_channels = [cr_drive_l, ctrl_drive_l, targ_drive_l]

    elif model_name == "Simple":
        # Simple all-μwave entangling gate for fixed-frequency SC qubits
        params = {
            "Model": f"{model_name}",
            "Drift": f"{Δct} * ZI",
            f"{cr_drive_l}": f"X_{i_c} I_{i_t} + {J / Δct: .2e} * Z_{i_c} X_{i_t}",
            f"{targ_drive_l}": f"{rt: .2e} * X_{i_t}",
            f"{ctrl_drive_l}": f"{rc: .2e} * X_{i_c}",
        }

        drift_op = Δct * ZI
        cr_drive_op = XI + J / Δct * ZX
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
