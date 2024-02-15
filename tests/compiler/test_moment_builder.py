from qiskit import QuantumCircuit
from pulse_simulator.compiler import MomentBuilder
import pytest


@pytest.fixture
def moment_builder():
    return MomentBuilder()


def test_moment_builder(moment_builder: MomentBuilder):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    expected = [
        {0: 1.5707963267948966, 1: -3.141592653589793},
        {0: "SX1", 1: "SX2"},
        {1: -3.141592653589793},
        {(0, 1): "ECR2"},
        {0: "X1"},
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
        0: {0: "SX1", 50: "ECR2_0", 320: "X1", 370: "ECR2_1"},
        1: {0: "SX2", 50: "ECR2_1", 370: "ECR2_0", 640: "X2"},
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
            0: "X1",
            50: (3.141592653589793, "X1"),
            100: "X1",
            150: (3.141592653589793, "X1"),
        },
        1: {
            0: "X2",
            50: (3.141592653589793, "X2"),
            100: "X2",
            150: (3.141592653589793, "X2"),
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
        0: {50: "ECR2_0"},
        1: {50: "ECR2_1"},
        2: {
            0: (-1.5707963267948966, "SX1"),
            50: "ECR1_0",
            320: (1.5707963267948966, "SX1"),
            370: 1.5707963267948966,
        },
        3: {
            0: (1.5707963267948966, "SX2"),
            50: "ECR1_1",
            320: (1.5707963267948966, "SX2"),
            370: 1.5707963267948966,
        },
    }
    assert moment_builder.build_moments(qc, timed=True) == expected
