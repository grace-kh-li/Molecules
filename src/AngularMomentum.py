from src.quantum_mechanics.Basis import *
from RotationalStates import *

class AngularMomentumState(BasisVector):
    def __init__(self, J, m, J_symbol="J_total", m_symbol="m_total", other_quantum_numbers = None):
        self.J_total = J
        self.m_total = m
        self.J_symbol = J_symbol
        self.m_symbol = m_symbol
        if other_quantum_numbers is None:
            self.other_quantum_numbers = {}
        else:
            self.other_quantum_numbers = other_quantum_numbers # dictionary {quantum_number_name: value}


        s = ""
        for qn in self.other_quantum_numbers:
            s += f"{qn}={other_quantum_numbers[qn]}, "
        s += f"{J_symbol}={J}, {m_symbol}={m}"
        super().__init__(s)

    def rename_symbols(self, J, m):
        self.J_symbol = J
        self.m_symbol = m

    def __str__(self):
        s = "|"
        for qn in self.other_quantum_numbers:
            if qn == "other":
                s += f"{self.other_quantum_numbers[qn]}, "
            else:
                s += f"{qn}={self.other_quantum_numbers[qn]}, "
        s += f"{self.J_symbol}={self.J_total}, {self.m_symbol}={self.m_total}>"
        return s


class AngularMomentumBasis(OrthogonalBasis):
    def __init__(self, basis_vectors, name = "basis"):
        """ Note: you are responsible for inputting correct and compete basis vectors.
        Each J space must have m = -J, ..., J."""
        super().__init__(basis_vectors, name)
        self.states_sorted = {} # dictionary: {other quantum numbers : {J: [J,m states]}}
        for b in self.basis_vectors:
            if not str(b.other_quantum_numbers) in self.states_sorted:
                self.states_sorted[str(b.other_quantum_numbers)] = {b.J_total: [b]}
            else:
                if b.J_total not in self.states_sorted[str(b.other_quantum_numbers)]:
                    self.states_sorted[str(b.other_quantum_numbers)][b.J_total] = [b]
                else:
                    self.states_sorted[str(b.other_quantum_numbers)][b.J_total].append(b)

    def __mul__(self, other):
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
                                    vectors.append(v)
                                    for qn in new_qn:
                                        setattr(v, qn, new_qn[qn])
                                    m += 1
                                F += 1
            return AngularMomentumBasis(vectors, f"{self.name} x {other.name}")
        else:
            tensor_basis = []
            for b1 in self:
                for b2 in other:
                    new_qn = {**b1.other_quantum_numbers, "other": b2.label}
                    v = AngularMomentumState(b1.J_total, b1.m_total, b1.J_symbol, b1.m_symbol, other_quantum_numbers=new_qn)
                    tensor_basis.append(v)
                    for qn in new_qn:
                        setattr(v, qn, new_qn[qn])
                    setattr(v, b1.J_symbol, b1.J_total)
                    setattr(v, b1.m_symbol, b1.m_total)
            return AngularMomentumBasis(tensor_basis, name=other.name + " x " + self.name)

    def rename_symbols(self, J, m):
        if len(self.basis_vectors) > 0 and isinstance(self.basis_vectors[0], AngularMomentumState):
            for b in self.basis_vectors:
                b.rename_symbols(J, m)



class ElectronicSpinState(AngularMomentumState):
    def __init__(self, S, ms):
        self.S = S
        self.ms = ms
        super().__init__(S,ms, "S", "ms")

class NuclearSpinState(AngularMomentumState):
    def __init__(self, I, mI):
        self.I = I
        self.mI = mI
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