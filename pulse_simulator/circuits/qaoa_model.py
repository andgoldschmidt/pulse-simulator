import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from typing import Union


class QAOAModel:
    def __init__(self, reps: int=1, seed: int=42):
        """
        Follow the qiskit tutorial: ".../tutorials/algorithms/05_qaoa.ipynb"
        See also: https://arxiv.org/abs/1709.03489

        """
        self.ADJACENCY_MATRIX = np.array([
            [0., 1., 1., 0.],
            [1., 0., 1., 1.],
            [1., 1., 0., 1.],
            [0., 1., 1., 0.]
        ])

        qubit_op, offset = self.get_operator()

        self.qaoa = QAOAAnsatz(cost_operator=qubit_op, reps=reps)

        self._rng = np.random.default_rng(seed=seed)


    def get_circuit(
            self, 
            parameters: Union[np.ndarray, None] = None,
            return_parameters: bool = False
        ) -> QAOAAnsatz:
        """
        Get the QAOA circuit with the given parameters.
        """
        if parameters is None:
            parameters = self._rng.standard_normal(len(self.qaoa.parameters)) * np.pi
        
        if return_parameters:
            return self.qaoa.assign_parameters(parameters), parameters
        else:
            return self.qaoa.assign_parameters(parameters)


    def max_cut_cost(self, x: np.ndarray) -> float:
        """Compute the value of a cut.
        Args:
            x: Binary string as numpy array.
            w: Adjacency matrix.
        Returns:
            Value of the cut.
        """
        X = np.outer(x, (1 - x))
        w_01 = np.where(self.ADJACENCY_MATRIX != 0, 1, 0)
        return np.sum(w_01 * X)


    def get_operator(self) -> tuple[SparsePauliOp, float]:
        r"""Generate Hamiltonian for the graph partitioning
        Notes:
            Goals:
                1 Separate the vertices into two set of the same size.
                2 Make sure the number of edges between the two set is minimized.
            Hamiltonian:
                H = H_A + H_B
                H_A = sum\_{(i,j)\in E}{(1-ZiZj)/2}
                H_B = (sum_{i}{Zi})^2 = sum_{i}{Zi^2}+sum_{i!=j}{ZiZj}
                H_A is for achieving goal 2 and H_B is for achieving goal 1.
        Args:
            weight_matrix: Adjacency matrix.
        Returns:
            Operator for the Hamiltonian
            A constant shift for the obj function.
        """
        num_nodes = len(self.ADJACENCY_MATRIX)
        pauli_list = []
        coeffs = []
        shift = 0

        for i in range(num_nodes):
            for j in range(i):
                if self.ADJACENCY_MATRIX[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append(Pauli((z_p, x_p)))
                    coeffs.append(-0.5)
                    shift += 0.5

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append(Pauli((z_p, x_p)))
                    coeffs.append(1.0)
                else:
                    shift += 1
                    
        return SparsePauliOp(pauli_list, coeffs=coeffs), shift
