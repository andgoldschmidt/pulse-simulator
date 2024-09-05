# Namespace: pulse_simulator_types

# Allowed gates
# look for `TODO: Hardcoded names`
IDENTITY_GATE_NAME = "i"
ONE_QUBIT_GATE_NAMES = ["sx", "x"] + [IDENTITY_GATE_NAME]
TWO_QUBIT_GATE_NAMES = ["ecr"]
VIRTUAL_GATE_NAMES = ["rz"]

# Allowed moments
CIRCUIT_MOMENT_NAMES = ["single", "ecr", "phase"]

# Type aliases
GATE_MOMENT = dict[int, str] | dict[tuple[int, int], str]
VIRTUAL_MOMENT = dict[int, float]
CIRCUIT_MOMENT = tuple[GATE_MOMENT, VIRTUAL_MOMENT]
TIMED_CIRCUIT_MOMENTS = dict[int, dict[int, str | float]]
