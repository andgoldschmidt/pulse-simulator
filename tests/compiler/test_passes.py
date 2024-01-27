from qiskit import QuantumCircuit
from qiskit.converters import dag_to_circuit, circuit_to_dag
from pulse_simulator.compiler.passes import (
    DeleteConsecutiveBarriers,
    SeparateMoments,
    SlideOneQubitOps,
    SlideTwoQubitOps,
)


def test_delete_consecutive_barrier():
    tp = DeleteConsecutiveBarriers()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.barrier()
    qc.h(1)

    expected_qc = QuantumCircuit(2)
    expected_qc.h(0)
    expected_qc.h(1)

    dag = circuit_to_dag(qc)
    new_qc = dag_to_circuit(tp.run(dag))

    assert new_qc == expected_qc


def test_separate_moments():
    tp = SeparateMoments()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cz(0, 1)
    qc.h(1)

    expected_qc = QuantumCircuit(2)
    expected_qc.h(0)
    expected_qc.barrier()
    expected_qc.cz(0, 1)
    expected_qc.barrier()
    expected_qc.h(1)

    dag = circuit_to_dag(qc)
    new_qc = dag_to_circuit(tp.run(dag))

    assert new_qc == expected_qc


def test_slide_one_q_ops():
    tp = SlideOneQubitOps()
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.barrier()
    qc.cz(0, 1)
    qc.barrier()
    qc.h(2)

    expected_qc = QuantumCircuit(3)
    expected_qc.h(0)
    expected_qc.h(2)
    expected_qc.barrier()
    expected_qc.cz(0, 1)
    expected_qc.barrier()

    dag = circuit_to_dag(qc)
    new_qc = dag_to_circuit(tp.run(dag))

    assert new_qc == expected_qc

def test_slide_two_q_ops():
    tp = SlideTwoQubitOps()
    qc = QuantumCircuit(4)
    qc.cz(0, 1)
    qc.barrier()
    qc.h(0)
    qc.h(1)
    qc.barrier()
    qc.cz(2, 3)

    expected_qc = QuantumCircuit(4)
    expected_qc.barrier()
    expected_qc.cz(0, 1)
    expected_qc.cz(2, 3)
    expected_qc.barrier()
    expected_qc.h(0)
    expected_qc.h(1)

    dag = circuit_to_dag(qc)
    new_qc = dag_to_circuit(tp.run(dag))

    assert new_qc == expected_qc