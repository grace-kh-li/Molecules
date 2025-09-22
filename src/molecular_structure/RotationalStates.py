from src.molecular_structure.RotationOperators import STM_R2_Operator, MShiftOperator, STM_Ra_Operator, \
    STM_LoweringOperator, STM_RaisingOperator
from src.quantum_mechanics.AngularMomentum import AngularMomentumState, AngularMomentumBasis
from src.quantum_mechanics.Operator import *

class RotationalState(AngularMomentumState):
    def __init__(self, R, m, other_qns, use_N=True):
        self.info = "RotationalState"
        if not use_N:
            super().__init__(R, m, "R", "mR", other_qns)
        else:
            super().__init__(R, m, "N", "mN", other_qns)

class RotationalBasis(AngularMomentumBasis):
    def __init__(self, vectors, name):
        self.info = "RotationalBasis"
        super().__init__(vectors, name)

    def get_R_subspace(self, R):
        out = []
        for b in self.basis_vectors:
            if b.R == R:
                out.append(b)
        return out


class STM_RotationalState(RotationalState):
    def __init__(self, R, k, m, use_N = True):
        assert R % 1 == 0 # make sure R, k, m are integers
        assert k % 1 == 0
        assert m % 1 == 0
        assert R >= 0
        assert -R <= k <= R
        assert -R <= m <= R
        super().__init__(R,m,{"k":k})
        self.R = R
        if use_N:
            self.N = R
        self.k = k
        self.mR = m
        self.label=f"{self.J_symbol}={R}, k={k}, {self.m_symbol}={m}"

    def __str__(self):
        return "|" + self.label + ">"


class STM_RotationalBasis(RotationalBasis):
    def __init__(self, R_range, k_range=(-100,100), m_range=(-100,100)):
        basis_vectors = []
        for R in range(R_range[0], R_range[1] + 1):
            for k in range(-R, R+1):
                if not k_range[0] <= abs(k) <= k_range[1]:
                    continue
                for m in range(-R, R+1):
                    if not m_range[0] <= m <= m_range[1]:
                        continue
                    basis_vectors.append(STM_RotationalState(R, k, m))
        super().__init__(basis_vectors, "STM Rotational Basis")

    def get_k_subspace(self, k):
        out = []
        for b in self.basis_vectors:
            if b.k == k or b.k == -k:
                out.append(b)
        return out


class Linear_RotationalState(STM_RotationalState):
    def __init__(self, R, m):
        super().__init__(R, 0, m)

class ATM_RotationalState(RotationalState):
    def __init__(self, R, ka, kc, m, STM_basis=None, STM_coeff=None, E=0.0):
        assert R % 1 == 0  # make sure R, k, m are integers
        assert ka % 1 == 0
        assert kc % 1 == 0
        assert m % 1 == 0
        assert R >= 0
        assert -R <= ka <= R
        assert -R <= kc <= R
        assert abs(ka) + abs(kc) == R or abs(ka) + abs(kc) == R + 1
        assert -R <= m <= R
        super().__init__(R, m, {"k_a": ka, "k_c":kc})
        self.R = R
        self.ka = ka
        self.kc = kc
        self.mR = m
        # self.STM_decomp = STM_decomp # a quantum state in STM basis, representing the physical composition of this ATM state.
        self.E = E
        if STM_basis is not None and STM_coeff is not None:
            self.set_defining_basis(STM_basis, STM_coeff)


class ATM_RotationalBasis(RotationalBasis):
    def __init__(self, A, BC_avg2, BC_diff4, R_range, m_range=(-100,100)):
        basis = STM_RotationalBasis(R_range=R_range, m_range=m_range)
        J_p = STM_RaisingOperator(basis)
        J_m = STM_LoweringOperator(basis)
        R2 = STM_R2_Operator(basis)
        Ra = STM_Ra_Operator(basis)
        Z = MShiftOperator(basis) * 1e-6
        H = Ra * Ra * (A - BC_avg2) + R2 * BC_avg2 + (J_p * J_p + J_m * J_m) * BC_diff4 + Z
        self.H = H
        self.STM_basis = basis
        Es, states= H.diagonalize()

        for s in states:
            s.sort() # sort the composition of the state in order of descending coefficients

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
            m_degeneracy = min(m_range[1],R) + 1 - max(m_range[0],-R)
            ka = 0
            i = 0
            while i < len(R_states[R]):
                if ka == 0:
                    for j in range(m_degeneracy):
                        s = R_states[R][i + j]
                        basis_vectors.append(ATM_RotationalState(R, ka=ka, kc=R, m=s[0].mR, STM_basis=self.STM_basis, STM_coeff=s.coeff, E=R_energies[R][i]))
                    i += m_degeneracy
                    ka += 1
                else:
                    if basis_vectors[-1].ka == ka and basis_vectors[-1].R == R:
                        for j in range(m_degeneracy):
                            s = R_states[R][i + j]
                            basis_vectors.append(ATM_RotationalState(R, ka=ka, kc=R-ka, m = s[0].mR, STM_basis=self.STM_basis, STM_coeff=s.coeff, E=R_energies[R][i]))
                        i += m_degeneracy
                        ka += 1
                    else:
                        for j in range(m_degeneracy):
                            s = R_states[R][i + j]
                            basis_vectors.append(ATM_RotationalState(R, ka=ka, kc=R+1-ka, m = s[0].mR, STM_basis=self.STM_basis, STM_coeff=s.coeff, E=R_energies[R][i]))
                        i += m_degeneracy

        vector_decomps = [b.coeff for b in basis_vectors]
        self.STM_basis_change_matrix = np.column_stack(vector_decomps) # STM basis coeff = M @ ATM basis coeff

        super().__init__(basis_vectors, "ATM rotational basis")

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

def dict_to_tuple(dict, exclude_list):
    a = []
    for key in dict:
        if key not in exclude_list:
            a.append(dict[key])
    return tuple(a)

def rename_ATM_states(states):
    qn_dict = {} # dictionary {quantum number tuple : states}. # This is to separate states with different quantum numbers so that each of these subspaces can be sorted into ATM states.

    for s in states:
        qn_tuple = dict_to_tuple(s[0].quantum_numbers, ("N","k","R"))
        if qn_tuple not in qn_dict:
            qn_dict[qn_tuple] = [s]
        else:
            qn_dict[qn_tuple].append(s)

    for qn in qn_dict:
        N_dict = {}
        for s in qn_dict[qn]: # sort the states with this set of qns into N stacks
            N = s[0].N
            if N not in N_dict:
                N_dict[N] = [s]
            else:
                N_dict[N].append(s)

        for N in N_dict:
            ka = 0
            i = 0
            last_state = None
            while i < len(N_dict[N]):
                s = N_dict[N][i]
                s.ka = ka
                if ka == 0:
                    s.ka = ka
                    s.kc = N
                    i += 1
                    ka += 1
                else:
                    if last_state.ka == ka:
                        s.ka = ka
                        s.kc = N - ka
                        i += 1
                        ka += 1
                    else:
                        s.ka = ka
                        s.kc = N + 1 - ka
                        i += 1
                last_state = s
                s.label = ""
                qns = s[0].quantum_numbers
                for qn in qns:
                    if qn not in ("N","k","R","J","m","mJ"):
                        s.label += f"{qn} = {qns[qn]}, "
                s.label += f"N={N}, ka={s.ka}, kc={s.kc}, "
                for qn in qns:
                    if qn in ("J","mJ","m","F"):
                        s.label += f"{qn} = {qns[qn]}, "
                s.label = s.label[:-2]
