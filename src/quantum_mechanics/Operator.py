from mpmath import isint

from src.quantum_mechanics.Basis import *

class Operator:
    def __init__(self, basis, matrix, symmetry_group=None, irrep=None):
        self.basis = basis
        self.matrix = matrix
        self.symmetry_group = symmetry_group
        self.irrep = irrep

    def matrix_element(self, state1, state2):
        return np.conj(state1.coeff) @ self.matrix @ state2.coeff

    def __getitem__(self, key):
        s1, s2 = key
        if isinstance(s1, int) and isinstance(s2, int):
            return self.matrix[s1, s2]
        elif isinstance(s1, QuantumState) and isinstance(s2, QuantumState):
            return self.matrix_element(s1, s2)
        else:
            raise TypeError("Operator[i,j] inputs must be integers or QuantumState")

    def __setitem__(self, key, value):
        s1, s2 = key
        self.matrix[s1, s2] = value

    def tensor(self, other):
        return Operator(self.basis * other.basis, np.kron(self.matrix, other.matrix))

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return str(self.matrix)

    def __add__(self, other):
        assert self.basis == other.basis
        return Operator(self.basis, self.matrix + other.matrix)

    def __sub__(self, other):
        assert self.basis == other.basis
        return Operator(self.basis, self.matrix)

    def __mul__(self, other):
        if isinstance(other, Operator):
            assert self.basis == other.basis
            return Operator(self.basis, self.matrix @ other.matrix)
        else:
            return Operator(self.basis, self.matrix * other)

    def __truediv__(self, c):
        return Operator(self.basis, self.matrix / c)

    def get_connected_states(self):
        pairs = []
        for i, b in enumerate(self.basis):
            for i1, b1 in enumerate(self.basis):
                if self.matrix[i,i1] != 0 and (b,b1) not in pairs and (b1,b) not in pairs:
                    pairs.append((b,b1))
        return pairs

    def diagonalize(self):
        eigenvalues, eigenvectors = self.sorted_eig(self.matrix)
        eigenstates = []
        for i in range(self.basis.dimension):
            v = eigenvectors[:,i]
            eigenstates.append(QuantumState(f"φ_{i}",v, self.basis))
        return eigenvalues, eigenstates

    def sorted_eig(self,A):
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = eigvals.real

        # Get the sorted indices of eigenvalues
        idx = np.argsort(eigvals)

        # Sort eigenvalues and eigenvectors accordingly
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        return eigvals_sorted, eigvecs_sorted

    def change_basis(self, other_basis, matrix):
        """ coeff in other basis = matrix @ coeff in current basis"""
        new_operator_matrix = np.linalg.inv(np.transpose(matrix)) @ self.matrix @ np.transpose(matrix)
        return Operator(other_basis, new_operator_matrix)

class ZeroOperator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((len(basis), len(basis)),dtype=np.complex128)
        super().__init__(basis, matrix)