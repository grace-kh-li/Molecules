from src.quantum_mechanics.Basis import *
from ElectronicStates import *
from VibrationalStates import *
from RotationalStates import *
from AngularMomentum import *

class HundsCaseA_State(AngularMomentumState):
    def __init__(self, S, Sigma,J,Omega,m):
        self.S = S
        self.Sigma = Sigma
        self.J = J
        self.Omega = Omega
        self.m = m
        self.k = self.Omega - self.Sigma

        super().__init__(J, m, J_symbol="J", m_symbol="m", other_quantum_numbers={"S":S,"Σ":Sigma,"Ω":Omega})

class HundsCaseA_Basis(AngularMomentumBasis):
    def __init__(self, S_range, J_range, Sigma_range=(-100,100),Omega_range=(-100,100),m_range=(-100,100)):
        vectors = []
        S = S_range[0]
        while S <= S_range[1]:
            Sigma = -S
            while Sigma <= S:
                if Sigma_range[0] <= abs(Sigma) <= Sigma_range[1]:
                    J = J_range[0]
                    while J <= J_range[1]:
                        P = -J
                        while P <= J:
                            if Omega_range[0] <= abs(P) <= Omega_range[1]:
                                m = -J
                                while m <= J:
                                    if m_range[0] <= m <= m_range[1]:
                                        vectors.append(HundsCaseA_State(S, Sigma, J, P, m))
                                    m += 1
                            P += 1
                        J += 1
                Sigma += 1
            S += 1
        super().__init__(vectors, "Hund's case A basis, without nuclear spin")
        for v in vectors:
            v.set_basis(self)

    def get_S_states(self, S):
        out = []
        for b in self.basis_vectors:
            if b.S == S:
                out.append(b)
        return out

    def get_Sigma_states(self, Sigma):
        out = []
        for b in self.basis_vectors:
            if abs(b.Sigma) == Sigma:
                out.append(b)
        return out

    def get_J_states(self, J):
        out = []
        for b in self.basis_vectors:
            if b.J_total == J:
                out.append(b)
        return out

    def get_Omega_states(self, Omega):
        out = []
        for b in self.basis_vectors:
            if abs(b.Omega) == Omega:
                out.append(b)
        return out

    def get_m_states(self, m):
        out = []
        for b in self.basis_vectors:
            if b.m_total == m:
                out.append(b)
        return out


class HundsCaseA_Basis_with_NS(AngularMomentumBasis):
    def __init__(self, S_range, J_range, I_range, Sigma_range=(-100,100), Omega_range=(-100,100),F_range=(-100,100),m_range=(-100,100)):
        b_ns = NuclearSpinBasis(I_range, [-I_range[1], I_range[1]])
        b_B = HundsCaseA_Basis(S_range=S_range, Sigma_range=Sigma_range,J_range=J_range,Omega_range=Omega_range,m_range=[-J_range[1],J_range[1]])
        product = b_ns * b_B
        vectors = []
        for b in product.basis_vectors:
            if m_range[0] <= b.m_total <= m_range[1] and F_range[0] <= b.J_total <= F_range[1]:
                b.F = b.J_total
                # b.J = b.other_quantum_numbers["J"]
                b.Omega = b.other_quantum_numbers["Ω"]
                b.Sigma = b.other_quantum_numbers["Σ"]
                # b.S = b.other_quantum_numbers["S"]
                # b.I = b.other_quantum_numbers["I"]
                b.m = b.m_total
                b.rename_symbols("F","mF")
                vectors.append(b)
        super().__init__(vectors, "Hund's case A Basis with nuclear spin")

    def get_I_states(self, I):
        out = []
        for b in self.basis_vectors:
            if b.I == I:
                out.append(b)
        return out

    def get_F_states(self, F):
        out = []
        for b in self.basis_vectors:
            if b.J_total == F:
                out.append(b)
        return out

    def get_S_states(self, S):
        out = []
        for b in self.basis_vectors:
            if b.S == S:
                out.append(b)
        return out

    def get_Sigma_states(self, Sigma):
        out = []
        for b in self.basis_vectors:
            if abs(b.Sigma) == Sigma:
                out.append(b)
        return out

    def get_J_states(self, J):
        out = []
        for b in self.basis_vectors:
            if b.J_total == J:
                out.append(b)
        return out

    def get_Omega_states(self, Omega):
        out = []
        for b in self.basis_vectors:
            if abs(b.Omega) == Omega:
                out.append(b)
        return out

    def get_m_states(self, m):
        out = []
        for b in self.basis_vectors:
            if b.m_total == m:
                out.append(b)
        return out