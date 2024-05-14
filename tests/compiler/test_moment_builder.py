from qiskit import QuantumCircuit
from pulse_simulator.compiler import MomentBuilder
from pulse_simulator import build_circuit
import pytest


@pytest.fixture
def moment_builder():
    return MomentBuilder()


@pytest.fixture
def cyclic_moment_builder():
    return MomentBuilder(coupling_map=[[0, 1], [1, 2], [2, 3], [0, 3]])


def test_moment_builder(moment_builder: MomentBuilder):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    expected = [
        {0: 1.5707963267948966, 1: -3.141592653589793},
        {0: "SX", 1: "SX"},
        {1: -3.141592653589793},
        {(0, 1): "ECR"},
        {0: "X"},
    ]
    assert moment_builder.build_moments(qc) == expected


def test_moment_builder_timed(moment_builder: MomentBuilder):
    qc = QuantumCircuit(2)
    qc.sx(0)
    qc.sx(1)
    qc.barrier()
    qc.ecr(0, 1)
    qc.barrier()
    qc.x(0)
    qc.barrier()
    qc.ecr(1, 0)
    qc.barrier()
    qc.x(1)

    expected = {
        0: {0: "SX", 50: "ECR_0", 320: "X", 370: "ECR_1"},
        1: {0: "SX", 50: "ECR_1", 370: "ECR_0", 640: "X"},
    }
    assert moment_builder.build_moments(qc, timed=True) == expected


def test_equal_to_identity(moment_builder: MomentBuilder):
    N = 2
    layers = 1

    qc = QuantumCircuit(N)
    for _ in range(layers):
        for q in range(N):
            qc.x(q)
            qc.y(q)
            qc.x(q)
            qc.y(q)

    expected = {
        0: {
            0: "X",
            50: (3.141592653589793, "X"),
            100: "X",
            150: (3.141592653589793, "X"),
        },
        1: {
            0: "X",
            50: (3.141592653589793, "X"),
            100: "X",
            150: (3.141592653589793, "X"),
        },
    }
    assert (
        moment_builder.build_moments(qc, timed=True, optimization_level=0) == expected
    )


def test_only_two_qubit(moment_builder: MomentBuilder):
    qc = QuantumCircuit(4)
    qc.ecr(0, 1)
    qc.ecr(3, 2)

    # compiler will fix end-to-end connection
    expected = {
        0: {50: "ECR_0"},
        1: {50: "ECR_1"},
        2: {
            0: (-1.5707963267948966, "SX"),
            50: "ECR_0",
            320: (1.5707963267948966, "SX"),
            370: 1.5707963267948966,
        },
        3: {
            0: (1.5707963267948966, "SX"),
            50: "ECR_1",
            320: (1.5707963267948966, "SX"),
            370: 1.5707963267948966,
        },
    }
    assert moment_builder.build_moments(qc, timed=True) == expected


def test_no_transpile(
    moment_builder: MomentBuilder, cyclic_moment_builder: MomentBuilder
):
    # Build test circuit with a cyclic connectivity
    registers = [0, 1, 2, 3]
    test_moments = [
        {0: "SX", 1: "SX", 2: "SX", 3: "SX"},
        {0: "SX", 1: "SX", 2: "SX", 3: "SX"},
        {(0, 3): "ECR", (1, 2): "ECR"},
    ]
    circ = build_circuit(
        test_moments,
        registers,
        insert_identities=False,
        includes_coloring=False,
        barriers=False,
    )

    # Check that no modifications occur
    test_moments_after = moment_builder.build_moments(
        circ,
        no_end_to_end=False,
        optimization_level=None,
    )
    assert test_moments_after == test_moments

    # Check that with the cyclic connectivity, the moments are the same
    test_moments_connected = cyclic_moment_builder.build_moments(
        circ,
        no_end_to_end=False,
        optimization_level=0,
    )
    assert test_moments_connected == test_moments

    # But, without the expected connectivity, the moments are different
    test_moments_disconnected = moment_builder.build_moments(
        circ,
        no_end_to_end=False,
        optimization_level=0,
    )
    assert test_moments_disconnected != test_moments
