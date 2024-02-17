from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, process_fidelity
from pulse_simulator.compiler import RobustScheduler

BASIS_GATES = ["sx", "x", "rz", "ecr", "id"]


def test_scheduler_equiv():
    qc = random_circuit(5, 20)
    sch = RobustScheduler(BASIS_GATES)
    out_qc1 = sch.run(qc, separate_moments=True, no_end_to_end=True)
    out_qc2 = sch.run(qc, separate_moments=True, no_end_to_end=False)
    out_qc3 = sch.run(qc, separate_moments=False, no_end_to_end=False)
    transpiled_qc = sch._transpile(qc)
    assert Operator(transpiled_qc).equiv(Operator(out_qc1))
    assert Operator(transpiled_qc).equiv(Operator(out_qc2))
    assert Operator(transpiled_qc).equiv(Operator(out_qc3))
