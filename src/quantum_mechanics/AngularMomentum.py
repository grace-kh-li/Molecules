from src.quantum_mechanics.Basis import *
from src.tools.WignerSymbols import wigner_3j
from src.quantum_mechanics.Operator import Operator

class AngularMomentumState(BasisVector):
    def __init__(self, J, m, J_symbol="J_total", m_symbol="m_total", other_quantum_numbers = None):
        self.J_total = J
        self.m_total = m
        self.J_symbol = J_symbol
        self.m_symbol = m_symbol
        self.__setattr__(J_symbol, J)
        self.__setattr__(m_symbol, m)
        if other_quantum_numbers is None:
            self.other_quantum_numbers = {}
        else:
            self.other_quantum_numbers = other_quantum_numbers # dictionary {quantum_number_name: value}
            for qn in self.other_quantum_numbers:
                self.__setattr__(qn, other_quantum_numbers[qn])

        s = ""
        for qn in self.other_quantum_numbers:
            s += f"{qn}={other_quantum_numbers[qn]}, "
        s += f"{J_symbol}={J}, {m_symbol}={m}"
        super().__init__(s)

        self.quantum_numbers = {self.J_symbol: self.J_total, self.m_symbol: self.m_total} | self.other_quantum_numbers
        self.reorder_quantum_numbers()
        self.AM_tensor_components = [self]

    def rename_symbols(self, J, m):
        self.quantum_numbers.pop(self.J_symbol)
        self.quantum_numbers.pop(self.m_symbol)

        self.J_symbol = J
        self.m_symbol = m
        self.__setattr__(J, self.J_total)
        self.__setattr__(m, self.m_total)
        self.quantum_numbers = {self.J_symbol: self.J_total, self.m_symbol: self.m_total} | self.other_quantum_numbers
        self.reorder_quantum_numbers()

    # def __str__(self):
    #     s = "|"
    #     for qn in self.other_quantum_numbers:
    #         if qn == "other":
    #             s += f"{self.other_quantum_numbers[qn]}, "
    #         else:
    #             s += f"{qn}={self.other_quantum_numbers[qn]}, "
    #     s += f"{self.J_symbol}={self.J_total}, {self.m_symbol}={self.m_total}>"
    #     return s


class AngularMomentumBasis(OrthogonalBasis):
    def __init__(self, basis_vectors, name = "basis"):
        """ Note: you are responsible for inputting correct and compete basis vectors.
        Each J space must have m = -J, ..., J."""
        super().__init__(basis_vectors, name)
        self.info = "AngularMomentumBasis" # again, this info is used to pass type information to parent classes to avoid circular imports.
        self.states_sorted = {} # dictionary: {other quantum numbers : {J: [J,m states]}}
        for b in self.basis_vectors:
            if not str(b.other_quantum_numbers) in self.states_sorted:
                self.states_sorted[str(b.other_quantum_numbers)] = {b.J_total: [b]}
            else:
                if b.J_total not in self.states_sorted[str(b.other_quantum_numbers)]:
                    self.states_sorted[str(b.other_quantum_numbers)][b.J_total] = [b]
                else:
                    self.states_sorted[str(b.other_quantum_numbers)][b.J_total].append(b)

        # if this basis came from coupling two angular momentum bases, the following properties should be updated.
        self.coupled = False
        self.tensor_basis = None
        self.uncoupled_bases = [self] # if the current basis came from coupling two angular momentum bases, this keeps track of those two bases.
        self.basis_change_matrix = None

    def __mul__(self, other):
        if not isinstance(other, AngularMomentumBasis):
            new_vectors = []
            for b in self.basis_vectors:
                for b1 in other.basis_vectors:
                    other_qns = b.other_quantum_numbers | b1.quantum_numbers
                    new_state = AngularMomentumState(b.J_total, b.m_total, b.J_symbol, b.m_symbol, other_qns)
                    new_state.tensor_components = b.tensor_components + b1.tensor_components
                    new_vectors.append(new_state)

            new_basis = AngularMomentumBasis(new_vectors, name=f"{self.label} x {other.label}")
            new_basis.coupled = self.coupled
            if self.coupled:
                new_basis.tensor_basis = self.tensor_basis * other
                new_basis.basis_change_matrix = np.kron(self.basis_change_matrix, np.eye(other.dimension))
                new_basis.uncoupled_bases = (self.uncoupled_bases[0], self.uncoupled_bases[1] * other)
                for i, b in enumerate(new_basis.basis_vectors):
                    b.set_defining_basis(new_basis.tensor_basis, new_basis.basis_change_matrix[i, :])


            new_basis.tensor_components = self.tensor_components + other.tensor_components

            return new_basis
        else:
            return self.tensor(other)


    def couple(self, other):
        """ Return a new AngularMomentumBasis that came from coupling self and other. The new basis elements will have
        coefficeints in terms of the uncoupled basis elements in their .coeff attribute. The change of basis matrix is also saved."""
        if isinstance(other, AngularMomentumBasis):
            # check that the angular momentum spaces are have the correct dimension
            for qn1 in self.states_sorted:
                for J in self.states_sorted[qn1]:
                    if not len(self.states_sorted[qn1][J]) == 2 * J + 1:
                        print(f"Warning: Angular momentum space of {self} is incomplete. Returning normal tensor product.")
                        return super().__mul__(other)
            for qn2 in other.states_sorted:
                for J in other.states_sorted[qn2]:
                    if not len(other.states_sorted[qn2][J]) == 2 * J + 1:
                        print(f"Warning: Angular momentum space of {other} is incomplete. Returning normal tensor product.")
                        return super().__mul__(other)

            vectors = []
            for qn1 in self.states_sorted:
                J1_states_dict = self.states_sorted[qn1]
                for qn2 in other.states_sorted:
                    J2_states_dict = other.states_sorted[qn2]
                    for J1 in J1_states_dict:
                        J1_symbol = J1_states_dict[J1][0].J_symbol
                        qn1_dict = J1_states_dict[J1][0].other_quantum_numbers
                        for J2 in J2_states_dict:
                            J2_symbol = J2_states_dict[J2][0].J_symbol
                            qn2_dict = J2_states_dict[J2][0].other_quantum_numbers
                            F = abs(J1 - J2)
                            while F <= J1 + J2:
                                m = -F
                                while m <= F:
                                    new_qn = {**qn1_dict, **qn2_dict, J1_symbol: J1, J2_symbol: J2}
                                    v = AngularMomentumState(F, m, other_quantum_numbers= new_qn)
                                    v.__setattr__(J1_symbol, J1)
                                    v.__setattr__(J2_symbol, J2)
                                    vectors.append(v)
                                    for qn in new_qn:
                                        setattr(v, qn, new_qn[qn])
                                    m += 1
                                F += 1
            new_basis =  AngularMomentumBasis(vectors, f"{self.label} x {other.label}")
            new_basis.came_from_tensor(self, other)
        else: # if one tries to "couple" an angular momentum basis with a non-angular momentum basis, just do the normal tensor product.
            new_basis = self * other
        return new_basis

    def rename_symbols(self, J, m):
        if len(self.basis_vectors) > 0 and isinstance(self.basis_vectors[0], AngularMomentumState):
            for b in self.basis_vectors:
                b.rename_symbols(J, m)

    def tensor(self, other):
        return super().__mul__(other)


    def get_uncoupled_basis(self):
        return self.tensor_basis

    def get_basis_change_matrix(self):
        return self.basis_change_matrix

    def came_from_tensor(self, basis1, basis2):
        """ This method is called to add information that reflect the fact that the current basis came from
        coupling two angular momentum bases. """
        self.coupled = True
        tensor_basis = basis1 * basis2
        matrix = np.zeros((len(self.basis_vectors), len(self.basis_vectors)))
        for i, coupled_b in enumerate(self):
            for j, uncoupled_b in enumerate(tensor_basis):
                phi1 = uncoupled_b.AM_tensor_components[0]
                phi2 = uncoupled_b.AM_tensor_components[1]
                J1, m1, J2, m2 = phi1.J_total, phi1.m_total, phi2.J_total, phi2.m_total
                J3, m3 = coupled_b.J_total, coupled_b.m_total
                same_qns = True
                for qn in phi1.other_quantum_numbers:
                    if coupled_b.other_quantum_numbers[qn] != phi1.other_quantum_numbers[qn]:
                        same_qns = False
                for qn in phi2.other_quantum_numbers:
                    if coupled_b.other_quantum_numbers[qn] != phi2.other_quantum_numbers[qn]:
                        same_qns = False
                if coupled_b.other_quantum_numbers[phi1.J_symbol] != phi1.J_total or coupled_b.other_quantum_numbers[phi2.J_symbol] != phi2.J_total:
                    same_qns = False
                if same_qns:
                    matrix[i, j] = (-1) ** (J1 - J2 + m3) * np.sqrt(2 * J3 + 1) * wigner_3j(J1, J2, J3, m1, m2, -m3)
        self.tensor_basis = tensor_basis
        self.uncoupled_bases = [basis1, basis2]
        self.tensor_components = basis1.tensor_components + basis2.tensor_components
        self.basis_change_matrix = matrix # how to use this matrix: uncoupled coeffs = M @ (coupled coeffs); coupled coeffs = transpose(M) @ (uncoupled coeffs)

        for i,b in enumerate(self.basis_vectors):
            b.set_defining_basis(self.tensor_basis, self.basis_change_matrix[i, :])

    def unravel_basis(self):
        """ By default, if an angular momentum basis C comes from coupling two angular momentum bases A and B,
         the basis vectors as quantum states will have (uncoupled) basis A x B and coefficients given by the
         CG coefficients. A and B can be simple angular momentum bases, or composite angular momentum bases (i.e.
         they also came from coupling multiple angular momentum bases).

         In this case, calling this function will set the basis of the current basis vectors (as quantum states)
         to the completely unraveled basis, i.e. tensor products of a bunch of simple angular momentum bases.

        This method is implemented recursively.
        """
        # base case
        if not self.coupled:
            self.basis_change_matrix = np.eye(self.dimension)
            self.tensor_basis = self
        else: # recursion
            basis1, basis2 = self.uncoupled_bases
            basis1.unravel_basis()
            basis2.unravel_basis()
            self.tensor_basis = basis1.tensor_basis * basis2.tensor_basis
            self.basis_change_matrix = self.basis_change_matrix @ np.kron(basis1.basis_change_matrix, basis2.basis_change_matrix)
        for i,b in enumerate(self.basis_vectors):
            b.set_defining_basis(self.tensor_basis, self.basis_change_matrix[i, :])

    def use_as_defining_basis(self):
        """ Set the basis itself as the defining basis. This means in future calculations, the basis vectors
        will be expressed in this basis, so the coefficients for the basis vectors will be
        [1,0,0,...], [0,1,0,...]. etc
        """
        for b in self.basis_vectors:
            b.set_basis(self)


class AngularMomentumOperator(Operator):
    def __init__(self, basis, axis):
        assert isinstance(basis, AngularMomentumBasis)
        matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
        if axis == 0 or axis == "z" or axis == "Z":
            for i, s1 in enumerate(basis):
                J1, m1 = s1.J_total, s1.m_total
                matrix[i,i] = m1
        elif axis == 1: # raising operator
            for i1, s1 in enumerate(basis):
                J1, m1 = s1.J_total, s1.m_total
                for i2, s2 in enumerate(basis):
                    J2, m2 = s2.J_total, s2.m_total
                    if J1 == J2 and m1 == m2 + 1:
                        matrix[i1, i2] = np.sqrt((J2 - m2) * (J2 + m2 + 1))
        elif axis == -1: # lowering operator
            for i1, s1 in enumerate(basis):
                J1, m1 = s1.J_total, s1.m_total
                for i2, s2 in enumerate(basis):
                    J2, m2 = s2.J_total, s2.m_total
                    if J1 == J2 and m1 == m2 - 1:
                        matrix[i1, i2] = np.sqrt((J2 + m2) * (J2 - m2 + 1))
        elif axis == "x" or axis == "X" or axis == "y" or axis == "Y":
            Jp = AngularMomentumOperator(basis, 1)
            Jm = AngularMomentumOperator(basis, -1)
            if axis == "x" or axis == "X":
                matrix = ((Jp + Jm)/2).matrix
            else:
                matrix = ((Jp - Jm)/(2j)).matrix
        else:
            raise NotImplementedError("axis must be 0,-1,1, or x, y, z!")

        super().__init__(basis, matrix)


class ElectronicSpinState(AngularMomentumState):
    def __init__(self, S, ms):
        super().__init__(S,ms, "S", "ms")

class NuclearSpinState(AngularMomentumState):
    def __init__(self, I, mI):
        super().__init__(I, mI, "I", "mI")

class ElectronicSpinBasis(AngularMomentumBasis):
    def __init__(self, S_range, ms_range=(-100,100)):
        vectors = []
        S = S_range[0]
        while S <= S_range[1]:
            ms = -S
            while ms <= S:
                if ms_range[0] <= ms <= ms_range[1]:
                    vectors.append(ElectronicSpinState(S,ms))
                ms += 1
            S += 1
        super().__init__(vectors,"ES basis")

class NuclearSpinBasis(AngularMomentumBasis):
    def __init__(self, I_range, mI_range=(-100,100)):
        vectors = []
        I = I_range[0]
        while I <= I_range[1]:
            mI = -I
            while mI <= I:
                if mI_range[0] <= mI <= mI_range[1]:
                    vectors.append(NuclearSpinState(I, mI))
                mI += 1
            I += 1
        super().__init__(vectors,"NS basis")