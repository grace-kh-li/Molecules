from src.quantum_mechanics.Operator import *

class STM_RotationalState(BasisVector):
    def __init__(self, R, k, m):
        assert R % 1 == 0 # make sure R, k, m are integers
        assert k % 1 == 0
        assert m % 1 == 0
        assert R >= 0
        assert -R <= k <= R
        assert -R <= m <= R
        super().__init__(f"R={R}, k={k}, mR={m}")
        self.R = R
        self.k = k
        self.m_R = m

class STM_RotationalBasis(OrthogonalBasis):
    def __init__(self, R_range, m_range):
        basis_vectors = []
        for R in range(R_range[0], R_range[1] + 1):
            for k in range(-R, R+1):
                for m in range(-R, R+1):
                    if not m_range[0] <= m <= m_range[1]:
                        continue
                    basis_vectors.append(STM_RotationalState(R, k, m))
        super().__init__(basis_vectors)

    def get_R_subspace(self, R):
        out = []
        for b in self.basis_vectors:
            if b.R == R:
                out.append(b)
        return out

    def get_k_subspace(self, k):
        out = []
        for b in self.basis_vectors:
            if b.k == k or b.k == -k:
                out.append(b)
        return out

class STM_RaisingOperator(Operator):
    def __init__(self, basis:STM_RotationalBasis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)
        for i, b1 in enumerate(basis):
            for j, b2 in enumerate(basis):
                if b1.R == b2.R and b1.m_R == b2.m_R and b1.k == b2.k + 1:
                    R = b2.R
                    k = b2.k
                    matrix[i, j] = np.sqrt(R*(R+1) - k * (k+1))
        super().__init__(basis, matrix)

class STM_LoweringOperator(Operator):
    def __init__(self, basis:STM_RotationalBasis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)
        for i, b1 in enumerate(basis):
            for j, b2 in enumerate(basis):
                if b1.R == b2.R and b1.m_R == b2.m_R and b1.k == b2.k - 1:
                    R = b2.R
                    k = b2.k
                    matrix[i, j] = np.sqrt(R*(R+1) - k * (k-1))
        super().__init__(basis, matrix)

class STM_R2_Operator(Operator):
    def __init__(self, basis:STM_RotationalBasis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)
        for i, b1 in enumerate(basis):
            matrix[i,i] = b1.R * (b1.R + 1)
        super().__init__(basis, matrix)

class STM_Ra_Operator(Operator):
    def __init__(self, basis:STM_RotationalBasis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)
        for i, b1 in enumerate(basis):
            matrix[i,i] = b1.k
        super().__init__(basis, matrix)

class Linear_RotationalState(STM_RotationalState):
    def __init__(self, R, m):
        super().__init__(R, 0, m)

class ATM_RotationalState(BasisVector):
    def __init__(self, R, ka, kc, m, STM_decomp=None, E=0.0, extra_label=""):
        assert R % 1 == 0  # make sure R, k, m are integers
        assert ka % 1 == 0
        assert kc % 1 == 0
        assert m % 1 == 0
        assert R >= 0
        assert -R <= ka <= R
        assert -R <= kc <= R
        assert abs(ka) + abs(kc) == R or abs(ka) + abs(kc) == R + 1
        assert -R <= m <= R
        super().__init__(f"{extra_label}, R={R}, ka={ka}, kc={kc}, m={m}")
        self.R = R
        self.ka = ka
        self.kc = kc
        self.m_R = m
        self.STM_decomp = STM_decomp # a quantum state in STM basis, representing the physical composition of this ATM state.
        self.E = E
        self.extra_label = extra_label + ", "

    def get_STM_decomp(self):
        return self.STM_decomp

    def __str__(self):
        if self.STM_decomp is None:
            return super().__str__()
        else:
            s = "|{label}{R}_{ka}{kc}> = ".format(label=self.extra_label,R=self.R, ka=self.ka, kc=self.kc)
            s += str(self.STM_decomp)[len(self.STM_decomp.name)+2:]
            return s


class ATM_RotationalBasis(OrthogonalBasis):
    def __init__(self, A, BC_avg2, BC_diff4, R_range, m_range, extra_label=""):
        basis = STM_RotationalBasis(R_range=R_range, m_range=m_range)
        J_p = STM_RaisingOperator(basis)
        J_m = STM_LoweringOperator(basis)
        R2 = STM_R2_Operator(basis)
        Ra = STM_Ra_Operator(basis)
        H = Ra * Ra * (A - BC_avg2) + R2 * BC_avg2 + (J_p * J_p + J_m * J_m) * BC_diff4
        Es, states = H.diagonalize()
        for s in states:
            s.sort()
        self.extra_label = extra_label


        R_states = {}
        R_energies = {}

        for R in range(R_range[0], R_range[1] + 1):
            if R not in R_states:
                R_states[R] = []
                R_energies[R] = []
        for i, s in enumerate(states):
            R = s[0].R
            R_states[R].append(s)
            R_energies[R].append(np.real(Es[i]))

        basis_vectors = []
        # assume prolate
        for R in R_states:
            ka = 0
            for i, s in enumerate(R_states[R]):
                if ka == 0:
                    basis_vectors.append(ATM_RotationalState(R, ka=ka, kc=R, m=s[0].m_R, STM_decomp=s, E=R_energies[R][i], extra_label=self.extra_label))
                    ka += 1
                else:
                    if basis_vectors[-1].ka == ka and basis_vectors[-1].R == R:
                        basis_vectors.append(ATM_RotationalState(R, ka=ka, kc=R-ka,m = s[0].m_R, STM_decomp=s, E=R_energies[R][i], extra_label=self.extra_label))
                        ka += 1
                    else:
                        basis_vectors.append(ATM_RotationalState(R, ka=ka, kc=R+1-ka,m = s[0].m_R, STM_decomp=s, E=R_energies[R][i], extra_label=self.extra_label))
        # else:
        #     for R in R_states:
        #         kc = 0
        #         for s in R_states[R]:
        #             if kc == 0:
        #                 basis_vectors.append(ATM_RotationalState(R, ka=R, kc=kc, m=s[0].m, STM_decomp=s, E=R_energies[R][i]))
        #                 kc += 1
        #             else:
        #                 if basis_vectors[-1].kc == kc and basis_vectors[-1].R == R:
        #                     basis_vectors.append(ATM_RotationalState(R, ka=R-kc, kc=kc,m = s[0].m, STM_decomp=s, E=R_energies[R][i]))
        #                     kc += 1
        #                 else:
        #                     basis_vectors.append(ATM_RotationalState(R, ka=R+1-kc, kc=kc,m = s[0].m, STM_decomp=s, E=R_energies[R][i]))

        super().__init__(basis_vectors)

    def get_ka_subspace(self, ka):
        out = []
        for s in self.basis_vectors:
            if s.ka == ka:
                out.append(s)
        return out

    def get_kc_subspace(self, kc):
        out = []
        for s in self.basis_vectors:
            if s.kc == kc:
                out.append(s)
        return out

    def get_state(self, R, ka, kc):
        for s in self.basis_vectors:
            if s.R == R and s.ka == ka and s.kc == kc:
                return s
        print("State not found.")
        return None