import networkx as nx
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import Barrier
from qiskit.circuit import Qubit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_circuit
from qiskit import ClassicalRegister


class ForbidEndToEnd(TransformationPass):
    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        qr = dag.qregs[list(dag.qregs.keys())[0]]
        if len(dag.cregs.keys()) != 0:
            cr = dag.cregs[list(dag.cregs.keys())[0]]
        else:
            cr = ClassicalRegister(0)

        for layer in dag.layers():
            subdag = layer["graph"]

            new_subdag = DAGCircuit()
            new_subdag.add_qreg(qr)
            new_subdag.add_creg(cr)

            print(dag_to_circuit(subdag))

            connections: list[tuple[Qubit, Qubit]] = []
            for gate in subdag.op_nodes(include_directives=True):
                if len(gate.qargs) == 2:
                    connections.append((gate.qargs[0], gate.qargs[1]))
                else:
                    new_subdag.apply_operation_back(
                        gate.op, qargs=gate.qargs, cargs=gate.cargs
                    )
            if len(connections) > 0:
                connection_graph = nx.DiGraph()
                
                # TODO: generalize to any connectivity
                for i in range(dag.num_qubits() - 1):
                    connection_graph.add_node((i, i + 1))
                for connection in connections:
                    qubit0, qubit1 = dag.find_bit(connection[0]).index, dag.find_bit(connection[1]).index
                    if qubit1 + 2 >= dag.num_qubits():
                        continue
                    end_connection = tuple(frozenset([qubit1 + 1, qubit1 + 2]))
                    tip_connection = tuple(frozenset([qubit0, qubit1]))
                    connection_graph.add_edge(tip_connection, end_connection)
                    
                print(connection_graph.edges)
                print(list(nx.topological_sort(connection_graph)))
                import matplotlib.pyplot as plt
                nx.draw(connection_graph, with_labels=True)
                plt.show()

            new_dag.compose(new_subdag, qubits=qr, clbits=cr)    

        return new_dag
