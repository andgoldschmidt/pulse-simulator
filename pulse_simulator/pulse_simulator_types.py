# Namespace: pulse_simulator_types

# Allowed gates
# look for `TODO: Hardcoded names`
# need to fix case sensitivity
IDENTITY_GATE_NAME = "ID"
ONE_QUBIT_GATE_NAMES = ["SX", "X"] + [IDENTITY_GATE_NAME]
TWO_QUBIT_GATE_NAMES = ["ECR"]
VIRTUAL_GATE_NAMES = ["RZ"]

# Allowed moments
# look for `TODO: Hardcoded names`
CIRCUIT_MOMENT_NAMES = ["single", "ecr", "phase"]

# Type aliases
GATE_MOMENT = dict[int, str] | dict[tuple[int, int], str]
VIRTUAL_MOMENT = dict[int, float]
CIRCUIT_MOMENT = tuple[GATE_MOMENT, VIRTUAL_MOMENT]
TIMED_CIRCUIT_MOMENTS = dict[int, dict[int, str | float]]
