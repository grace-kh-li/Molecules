from src.quantum_mechanics.Basis import *

class Operator:
    def __init__(self, basis, matrix):
        self.basis = basis
        self.matrix = matrix

    def matrix_element(self, state1, state2):
        return np.conj(state1.coeff) @ self.matrix @ state2.coeff

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

        # Get the sorted indices of eigenvalues
        idx = np.argsort(eigvals)

        # Sort eigenvalues and eigenvectors accordingly
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        return eigvals_sorted, eigvecs_sorted