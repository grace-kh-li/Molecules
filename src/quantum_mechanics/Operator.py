from src.quantum_mechanics.Basis import *


class Operator:
    def __init__(self, basis, matrix, symmetry_group=None, irrep=None):
        self.basis = basis
        self.matrix = matrix
        self.symmetry_group = symmetry_group
        self.irrep = irrep

    def matrix_element(self, state1, state2):
        assert state1.defining_basis == self.basis == state2.defining_basis
        return np.conj(state1.coeff.T) @ self.matrix @ state2.coeff

    def __getitem__(self, key):
        s1, s2 = key
        if isinstance(s1, int) and isinstance(s2, int):
            return self.matrix[s1, s2]
        elif isinstance(s1, BasisVector) and isinstance(s2, BasisVector):
            assert s1 in self.basis and s2 in self.basis
            return self.matrix[self.basis.get_index(s1), self.basis.get_index(s2)]
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
        return Operator(self.basis, self.matrix - other.matrix)

    def __mul__(self, other):
        if isinstance(other, Operator):
            assert self.basis == other.basis
            return Operator(self.basis, self.matrix @ other.matrix)
        else:
            return Operator(self.basis, self.matrix * other)

    def __truediv__(self, c):
        return Operator(self.basis, self.matrix / c)

    def get_connected_states(self, threshold=1e-5):
        pairs = []
        for i, b in enumerate(self.basis):
            for i1, b1 in enumerate(self.basis):
                if np.abs(self.matrix[i, i1]) > threshold and (b, b1) not in pairs and (b1, b) not in pairs:
                    if "elec" not in b.quantum_numbers:
                        pairs.append((b, b1))
                    else:
                        if b1.quantum_numbers["elec"] == "X":
                            pairs.append((b1, b))
                        elif b1.quantum_numbers["elec"] < b.quantum_numbers["elec"] != "X":
                            pairs.append((b1, b))
                        else:
                            pairs.append((b, b1))
        return pairs

    def diagonalize(self, get_matrix=False, get_states=True):
        """ Return the eigenvalues eigen states as quantum states, and eigenvectors of the matrix (coefficients)"""
        eigenvalues, eigenvectors = self.sorted_eig(self.matrix)

        if not get_matrix and not get_states:
            return eigenvalues

        if get_matrix and not get_states:
            return eigenvalues, eigenvectors
        else:
            eigenstates = []
            for i in range(self.basis.dimension):
                v = eigenvectors[:, i]
                eigenstates.append(QuantumState(f"φ_{i}", v, self.basis))
            if get_matrix:
                return eigenvalues, eigenstates, eigenvectors
            else:
                return eigenvalues, eigenstates

    def sorted_eig(self, A):
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
        """ Convention for the orthonormal basis change matrix: coeff in other basis = matrix @ coeff in current basis"""
        new_operator_matrix = matrix @ self.matrix @ np.conj(matrix.T)
        return Operator(other_basis, new_operator_matrix)


class ZeroOperator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
        super().__init__(basis, matrix)
