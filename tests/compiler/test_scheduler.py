from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator
from pulse_simulator.compiler import RobustScheduler

BASIS_GATES = ["sx", "x", "rz", "ecr", "id"]


def test_scheduler_equiv():
    qc = random_circuit(5, 20)
    sch = RobustScheduler(BASIS_GATES)
    out_qc = sch.run(qc)
    transpiled_qc = sch._transpile(qc)
    assert Operator(transpiled_qc).equiv(Operator(out_qc))
