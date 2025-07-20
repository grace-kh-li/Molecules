import numpy as np

class BasisVector:
    def __init__(self, label):
        self.label = label
        self.basis = None

    def dot(self, other):
        if self.basis != other.states:
            raise ValueError("Dot product between basis vectors is not defined")
        else:
            if self.label == other.name:
                return 1
            else:
                return 0

    def __str__(self):
        return "|" + self.label + ">"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.label == other.name

    def copy(self):
        return BasisVector(self.label)

class OrthogonalBasis:
    def __init__(self, basis_vectors, name = "basis"):
        self.basis_vectors = basis_vectors
        self.name = name
        for b in basis_vectors:
            b.states = self
        self.dimension = len(basis_vectors)

    def __str__(self):
        s = self.name + " = \n{"
        for b in self.basis_vectors:
            s += str(b) + ", \n "
        s = s[:-4] + " }"
        return s

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.basis_vectors)

    def __getitem__(self, i):
        return self.basis_vectors[i]

    def __mul__(self, other):
        return OrthogonalBasis.tensor_product(self, other)

    def __add__(self, other):
        return OrthogonalBasis.direct_sum(self, other)

    @staticmethod
    def tensor_product(basis1, basis2):
        tensor_basis = []
        for b1 in basis1:
            for b2 in basis2:
                label = b1.name + ", " + b2.name
                tensor_basis.append(BasisVector(label))
        return OrthogonalBasis(tensor_basis, name=basis1.name + " x " + basis2.name)

    @staticmethod
    def direct_sum(basis1, basis2):
        sum_basis = []
        for b1 in basis1:
            sum_basis.append(b1.copy())
        for b2 in basis2:
            sum_basis.append(b2.copy())
        return OrthogonalBasis(sum_basis, name=basis1.name + " + " + basis2.name)

class HilbertSpace:
    def __init__(self, states, name, allowed_basis_classes = None):
        self.states = states
        self.name = name
        self.dimension = states.dimension
        self.other_bases = allowed_basis_classes

    def __add__(self, other):
        return HilbertSpace(self.states + other.states, name=f"{self.name} + {other.name}")

    def __mul__(self, other):
        return HilbertSpace(self.states * other.states, name=f"{self.name} x {other.name}")

    def change_to_basis(self, vector, new_basis):
        pass



class QuantumState:
    def __init__(self, name, coeff, basis, sorted=False, Hilbert_space=None):
        self.coeff = np.array(coeff)
        assert self.coeff.shape[0] == basis.dimension
        self.basis = basis
        self.name = name
        self.Hilbert_space = Hilbert_space

        self.non_zero_coeffs = []
        self.non_zero_basis = []
        self.sorted = sorted

        for i, b in enumerate(self.basis):
            if np.abs(self.coeff[i]) > 1e-9:
                self.non_zero_basis.append(b)
                self.non_zero_coeffs.append(coeff[i])

        if sorted:
            self.non_zero_coeffs, self.non_zero_basis = self._get_sorted_by_magnitude()

    def braket(self, other):
        """<self|other>"""
        if self.basis != other.basis: #TODO: add change of basis, including different basis and subbasis
            raise ValueError("Dot product between basis vectors is not defined")
        else:
            return np.dot(np.conj(self.coeff), other.coeff)

    def __add__(self, other):
        assert self.basis == other.basis # todo: add change of basis
        return QuantumState(self.coeff + other.coeff, self.basis)

    def __mul__(self, c):
        return QuantumState(self.coeff * c, self.basis)

    def __truediv__(self,c):
        return QuantumState(self.coeff / c, self.basis)

    def __sub__(self, other):
        assert self.basis == other.basis  # todo: add change of basis
        return QuantumState(self.coeff - other.coeff, self.basis)

    def project_onto(self, other):
        assert self.basis == other.basis  # todo: add change of basis
        return other * other.braket(self) / other.norm()**2

    def __eq__(self, other):
        return self.basis == other.basis and np.sum(np.abs(self.coeff - other.coeff)) < 1e-9

    def __str__(self):
        s = self.name + " = "
        for i, b in enumerate(self.non_zero_basis):
            if abs(np.real(self.non_zero_coeffs[i])) < 1e-9:
                s += "{:.2f}j {} + ".format(np.imag(self.non_zero_coeffs[i]), str(b))
            elif abs(np.imag(self.non_zero_coeffs[i])) < 1e-9:
                s += "{:.2f} {} + ".format(np.real(self.non_zero_coeffs[i]), str(b))
        return s[:-2]

    def _get_sorted_by_magnitude(self):
        # Pair each coefficient with its corresponding basis element
        paired = list(zip(self.non_zero_coeffs, self.non_zero_basis))

        # Sort based on the magnitude (norm) of the complex coefficient
        sorted_pairs = sorted(paired, key=lambda pair: -abs(pair[0]))

        # Unzip the sorted pairs back into two separate lists
        sorted_coeffs, sorted_basis = zip(*sorted_pairs) if sorted_pairs else ([], [])

        return list(sorted_coeffs), list(sorted_basis)

    def __repr__(self):
        return str(self)

    def norm(self):
        return np.linalg.norm(self.coeff)

    def __getitem__(self, i):
        return self.non_zero_basis[i]

    def __contains__(self, basis_vector):
        if not isinstance(basis_vector, BasisVector):
            return False
        return basis_vector in self.non_zero_basis

    def __iter__(self):
        return iter(self.non_zero_basis)

    def sort(self):
        """ Sort the printed state based on the magnitude of the basis vectors. """
        self.non_zero_coeffs, self.non_zero_basis = self._get_sorted_by_magnitude()
