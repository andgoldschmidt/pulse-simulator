import pytest
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator
from pulse_simulator.compiler import RobustScheduler


@pytest.fixture
def basis_gates():
    return ["sx", "x", "rz", "ecr", "id"]

@pytest.fixture
def coupling_map():
    return [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]


def test_scheduler_equiv(basis_gates, coupling_map):
    qc = random_circuit(5, 20)
    sch = RobustScheduler(basis_gates, coupling_map)
    out_qc1 = sch.run(qc, separate_moments=True, no_end_to_end=True)
    out_qc2 = sch.run(qc, separate_moments=True, no_end_to_end=False)
    out_qc3 = sch.run(qc, separate_moments=False, no_end_to_end=False)
    transpiled_qc = sch._transpile(qc)
    assert Operator(transpiled_qc).equiv(Operator(out_qc1))
    assert Operator(transpiled_qc).equiv(Operator(out_qc2))
    assert Operator(transpiled_qc).equiv(Operator(out_qc3))
