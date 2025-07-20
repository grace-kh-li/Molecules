from Basis import *
from Operator import *

class STM_RotationalState(BasisVector):
    def __init__(self, R, k, m):
        assert R % 1 == 0 # make sure R, k, m are integers
        assert k % 1 == 0
        assert m % 1 == 0
        assert R >= 0
        assert -R <= k <= R
        assert -R <= m <= R
        super().__init__(f"R={R}, k={k}, m={m}")
        self.R = R
        self.k = k
        self.m_R = m

class STM_RotationalBasis(OrthogonalBasis):
    def __init__(self, R_range, k_range, m_range):
        basis_vectors = []
        for R in range(R_range[0], R_range[1] + 1):
            for k in range(-R, R+1):
                if not k_range[0] <= k <= k_range[1]:
                    continue
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
    def __init__(self, R, ka, kc, m, STM_decomp=None):
        assert R % 1 == 0  # make sure R, k, m are integers
        assert ka % 1 == 0
        assert kc % 1 == 0
        assert m % 1 == 0
        assert R >= 0
        assert -R <= ka <= R
        assert -R <= kc <= R
        assert abs(ka) + abs(kc) == R or abs(ka) + abs(kc) == R + 1
        assert -R <= m <= R
        super().__init__(f"R={R}, ka={ka}, kc={kc}, m={m}")
        self.R = R
        self.ka = ka
        self.kc = kc
        self.m_R = m
        self.STM_decomp = STM_decomp # a quantum state in STM basis, representing the physical composition of this ATM state.

    def get_STM_decomp(self):
        return self.STM_decomp