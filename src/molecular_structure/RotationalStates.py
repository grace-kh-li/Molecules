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


class Linear_RotationalState(RotationalState):
    def __init__(self, R, m, STM_basis=None, STM_coeff=None, E=0.0, use_N=True):
        assert R % 1 == 0
        assert m % 1 == 0
        assert R >= 0
        assert -R <= m <= R
        super().__init__(R, m, {"k": 0}, use_N=use_N)
        self.R = R
        if use_N:
            self.N = R
        self.k = 0
        self.mR = m
        self.E = E
        self.label = f"{self.J_symbol}={R}, k=0, {self.m_symbol}={m}"
        if STM_basis is not None and STM_coeff is not None:
            self.set_defining_basis(STM_basis, STM_coeff)

    def __str__(self):
        return "|" + self.label + ">"


class Linear_RotationalBasis(RotationalBasis):
    def __init__(self, R_range, m_range=(-100,100), use_N=True):
        self.STM_basis = STM_RotationalBasis(R_range=R_range, k_range=(0, 0), m_range=m_range)
        basis_vectors = []
        for i, stm_state in enumerate(self.STM_basis):
            coeff = np.zeros(self.STM_basis.dimension)
            coeff[i] = 1
            basis_vectors.append(
                Linear_RotationalState(
                    stm_state.R,
                    stm_state.mR,
                    STM_basis=self.STM_basis,
                    STM_coeff=coeff,
                    use_N=use_N,
                )
            )
        self.STM_basis_change_matrix = np.column_stack([b.coeff for b in basis_vectors])
        super().__init__(basis_vectors, "Linear rotational basis")

    def get_state(self, R, m):
        for s in self.basis_vectors:
            if s.R == R and s.mR == m:
                return s
        print("State not found.")
        return None

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

def _atm_wang_basis(basis):
    """Return real Wang vectors and their prolate (N, ka, kc) labels.

    With a=z, b=x, c=y, the relative sign of the +/-k components is
    (-1)**(N + ka + kc). Thus kc parity follows the wavefunction symmetry,
    even when spin-rotation reverses the energy order of a doublet.
    Convention: https://pgopher.chemistry.bristol.ac.uk/Help/asymsym.htm
    """
    lookup = {tuple(sorted(b.quantum_numbers.items())): i for i, b in enumerate(basis)}
    vectors, labels = [], []
    for i, b in enumerate(basis):
        qns = b.quantum_numbers
        k = qns["k"]
        if k < 0:
            continue
        N = qns.get("N", qns.get("R"))
        if k == 0:
            vector = np.zeros(len(basis))
            vector[i] = 1
            vectors.append(vector)
            labels.append({key: value for key, value in qns.items() if key != "k"}
                          | {"ka": 0, "kc": N})
            continue
        partner = dict(qns, k=-k)
        j = lookup[tuple(sorted(partner.items()))]
        for kc in (N - k, N - k + 1):
            vector = np.zeros(len(basis))
            vector[i] = 1 / np.sqrt(2)
            vector[j] = (-1)**(N + k + kc) / np.sqrt(2)
            vectors.append(vector)
            labels.append({key: value for key, value in qns.items() if key != "k"}
                          | {"ka": k, "kc": kc})
    return np.column_stack(vectors), labels


def diagonalize_ATM_Hamiltonian(H):
    """Solve a Hamiltonian preserving ka/kc parity in symmetry blocks.

    Keep each conserved nonrotational quantum number separate (including J
    and m at zero field). Detect conservation from the actual matrix, so
    Zeeman terms may mix J and, for transverse fields, m. Subtract the block
    origin before diagonalizing to avoid losing small splittings to large
    electronic offsets. The returned coefficients remain in H.basis.
    """
    basis = H.basis
    wang, labels = _atm_wang_basis(basis)
    rows, cols = np.nonzero(H.matrix)
    conserved = []
    for qn in basis[0].quantum_numbers:
        if qn in ("N", "R", "k"):
            continue
        values = np.array([b.quantum_numbers[qn] for b in basis])
        if np.all(values[rows] == values[cols]):
            conserved.append(qn)

    groups = {}
    for i, b in enumerate(basis):
        key = tuple(b.quantum_numbers[qn] for qn in conserved)
        groups.setdefault(key, []).append(i)
    wang_groups = {}
    for i, qns in enumerate(labels):
        key = tuple(qns[qn] for qn in conserved)
        wang_groups.setdefault(key, []).append(i)

    energies, vectors = [], []
    for key, indices in groups.items():
        columns = wang_groups[key]
        transform = wang[np.ix_(indices, columns)]
        matrix = np.array(H.matrix[np.ix_(indices, indices)], dtype=complex, copy=True)
        origin = np.trace(matrix).real / len(indices)
        matrix -= origin * np.eye(len(indices))
        matrix = transform.T @ matrix @ transform
        symmetries = [(labels[i]["ka"] % 2, labels[i]["kc"] % 2) for i in columns]
        blocks = {}
        for i, symmetry in enumerate(symmetries):
            blocks.setdefault(symmetry, []).append(i)
        # Do not silently discard a physical interaction that breaks these
        # rotational symmetries. Roundoff from the Wang transform is allowed.
        forbidden = np.array([[a != b for b in symmetries] for a in symmetries])
        tolerance = 64 * np.finfo(float).eps * max(1, np.max(np.abs(matrix)))
        if np.any(np.abs(matrix[forbidden]) > tolerance):
            raise ValueError("Hamiltonian does not conserve ka/kc parity")
        for block in blocks.values():
            Es, coeffs = np.linalg.eigh(matrix[np.ix_(block, block)])
            original_coeffs = transform[:, block] @ coeffs
            for E, coeff in zip(Es, original_coeffs.T):
                vector = np.zeros(len(basis), dtype=complex)
                vector[indices] = coeff
                energies.append(E + origin)
                vectors.append(vector)

    order = np.argsort(energies, kind="stable")
    states = [QuantumState(f"φ_{i}", vectors[j], basis, sorted=True)
              for i, j in enumerate(order)]
    return np.asarray(energies)[order], states


def rename_ATM_states(states):
    """Assign near-prolate labels from Wang-component probabilities.

    N and ka are approximate when spin-rotation mixes rotational states.
    Sum the probabilities for each (N, ka, kc), then use the most populated
    assignment. This is independent of energy order, input order, eigenvector
    phase, and the display ordering/threshold of QuantumState components.
    """
    if not states:
        return
    basis = states[0].defining_basis
    if any(s.defining_basis is not basis for s in states):
        raise ValueError("ATM states must share a defining basis")
    wang, labels = _atm_wang_basis(basis)
    probabilities = np.abs(wang.T @ np.column_stack([s.coeff for s in states]))**2
    for s, weights in zip(states, probabilities.T):
        populations = {}
        for qns, weight in zip(labels, weights):
            key = (qns.get("N", qns.get("R")), qns["ka"], qns["kc"])
            populations[key] = populations.get(key, 0.0) + weight
        N, ka, kc = max(populations, key=populations.get)
        candidates = [i for i, qns in enumerate(labels)
                      if (qns.get("N", qns.get("R")), qns["ka"], qns["kc"]) == (N, ka, kc)]
        dominant = max(candidates, key=lambda i: weights[i])
        s.quantum_numbers = dict(labels[dominant])
        s.N, s.ka, s.kc = N, ka, kc
        s.sort()  # Preserve the notebook convention that s[0] is dominant.
        s.label = ", ".join(f"{qn}={value}" for qn, value in s.quantum_numbers.items())
