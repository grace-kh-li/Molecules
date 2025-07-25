from src.quantum_mechanics.Basis import *
from src.AngularMomentum import *


class HundsCaseB_State(AngularMomentumState):
    def __init__(self, N,k,S,J,m):
        self.N = N
        self.k = k
        self.S = S
        self.J = J
        self.m = m
        super().__init__(J, m, J_symbol="J", m_symbol="m", other_quantum_numbers={"N":N,"k":k, "S":S})

class HundsCaseB_Basis(AngularMomentumBasis):
    def __init__(self, N_range, S_range, k_range=(-100,100), J_range=(-100,100), m_range=(-100,100)):
        vectors = []
        for N in range(N_range[0], N_range[1] + 1):
            for k in range(-N, N + 1):
                if k_range[0] <= abs(k) <= k_range[1]:
                    S = S_range[0]
                    while S <= S_range[1]:
                        J = max(N, S) - min(N, S)
                        while J <= N + S:
                            if J_range[0] <= J <= J_range[1]:
                                m = -J
                                while m <= J:
                                    if m_range[0] <= m <= m_range[1]:
                                        vectors.append(HundsCaseB_State(N, k, S, J, m))
                                    m += 1
                            J += 1
                        S += 1
        super().__init__(vectors, "Hund's case B Basis")

    def get_N_states(self, N):
        out = []
        for b in self.basis_vectors:
            if b.N == N:
                out.append(b)
        return out

    def get_k_states(self, k):
        out = []
        for b in self.basis_vectors:
            if b.k == k:
                out.append(b)
        return out

    def get_S_states(self, S):
        out = []
        for b in self.basis_vectors:
            if b.S == S:
                out.append(b)
        return out

    def get_J_states(self, J):
        out = []
        for b in self.basis_vectors:
            if b.J_total == J:
                out.append(b)
        return out

    def get_m_states(self, m):
        out = []
        for b in self.basis_vectors:
            if b.m_total == m:
                out.append(b)
        return out

class HundsCaseB_Basis_with_NS(AngularMomentumBasis):
    def __init__(self, N_range, S_range,I_range, k_range=(-100,100), J_range=(-100,100), F_range=(-100,100), m_range=(-100,100)):
        b_ns = NuclearSpinBasis(I_range, [-I_range[1], I_range[1]])
        b_B = HundsCaseB_Basis(N_range=N_range, k_range=k_range, S_range=S_range, J_range=J_range, m_range=[-J_range[1],J_range[1]])
        product = b_ns * b_B
        vectors = []
        for b in product.basis_vectors:
            if m_range[0] <= b.m_total <= m_range[1] and F_range[0] <= b.J_total <= F_range[1]:
                b.F = b.J_total
                # b.J = b.other_quantum_numbers["J"]
                # b.N = b.other_quantum_numbers["N"]
                # b.k = b.other_quantum_numbers["k"]
                # b.S = b.other_quantum_numbers["S"]
                # b.I = b.other_quantum_numbers["I"]
                b.m = b.m_total
                b.rename_symbols("F","mF")
                vectors.append(b)
        super().__init__(vectors, "Hund's case B Basis with nuclear spin")

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

    def get_N_states(self, N):
        out = []
        for b in self.basis_vectors:
            if b.N == N:
                out.append(b)
        return out

    def get_k_states(self, k):
        out = []
        for b in self.basis_vectors:
            if b.k == k:
                out.append(b)
        return out

    def get_S_states(self, S):
        out = []
        for b in self.basis_vectors:
            if b.S == S:
                out.append(b)
        return out

    def get_J_states(self, J):
        out = []
        for b in self.basis_vectors:
            if b.J_total == J:
                out.append(b)
        return out

    def get_m_states(self, m):
        out = []
        for b in self.basis_vectors:
            if b.m_total == m:
                out.append(b)
        return out
