from src.quantum_mechanics.Basis import *
from ElectronicStates import *
from VibrationalStates import *
from RotationalStates import *
from SpinStates import *


class HundsCaseB_State(BasisVector):
    def __init__(self, N,k,S,J,m):
        self.N = N
        self.k = k
        self.S = S
        self.J = J
        self.m = m
        super().__init__(f"S={S}, N={N}, k={k}, J={J}, mJ={m}")

class HundsCaseB_Basis(OrthogonalBasis):
    def __init__(self, N_range, k_range, S_range, J_range, m_range):
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
            if b.J == J:
                out.append(b)
        return out

    def get_m_states(self, m):
        out = []
        for b in self.basis_vectors:
            if b.m == m:
                out.append(b)
        return out

class HundsCaseB_State_with_NS(BasisVector):
    def __init__(self, N,k,S,J,I,F,m):
        self.N = N
        self.k = k
        self.S = S
        self.J = J
        self.m = m
        self.I = I
        self.F = F
        super().__init__(f"S={S}, I={I}, N={N}, k={k}, J={J}, F={F}, mF={m}")

class HundsCaseB_Basis_with_NS(OrthogonalBasis):
    def __init__(self, N_range, k_range, S_range, J_range, I_range, F_range, m_range):
        vectors = []
        for N in range(N_range[0], N_range[1]+1):
            for k in range(-N,N+1):
                if k_range[0] <= abs(k) <= k_range[1]:
                    S = S_range[0]
                    while S <= S_range[1]:
                        J = max(N,S) - min(N,S)
                        while J <= N + S:
                            if J_range[0] <= J <= J_range[1]:
                                I = I_range[0]
                                while I <= I_range[1]:
                                    F = max(J,I) - min(J,I)
                                    while F <= I + J:
                                        if F_range[0] <= F <= F_range[1]:
                                            m = -F
                                            while m <= F:
                                                if m_range[0] <= m <= m_range[1]:
                                                    vectors.append(HundsCaseB_State_with_NS(N, k, S, J, I,F,m))
                                                m += 1
                                        F += 1
                                    I += 1
                            J += 1
                        S += 1
        super().__init__(vectors,"Hund's case B Basis with nuclear spin")

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
            if b.J == J:
                out.append(b)
        return out

    def get_m_states(self, m):
        out = []
        for b in self.basis_vectors:
            if b.m == m:
                out.append(b)
        return out

    def get_I_states(self, I):
        out = []
        for b in self.basis_vectors:
            if b.I == I:
                out.append(b)
        return out

    def get_F_states(self, F):
        out = []
        for b in self.basis_vectors:
            if b.F == F:
                out.append(b)
        return out