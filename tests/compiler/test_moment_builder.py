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
        {0: 'SX1', 1: 'SX2'},
        {1: -3.141592653589793},
        {(0, 1): 'ECR2'},
        {0: 'X1'},
    ]
    assert moment_builder.build_moments(qc) == expected
        