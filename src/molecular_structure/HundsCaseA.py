import numpy as np

from src.molecular_structure.RotationalStates import Linear_RotationalBasis
from src.quantum_mechanics.AngularMomentum import *

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

    def get_caseB_basis_change_matrix(self, caseB_basis):
        """Return a matrix mapping coefficients in this case A basis to case B."""
        from src.molecular_structure.HundsCaseTransforms import caseA_to_caseB_matrix

        return caseA_to_caseB_matrix(self, caseB_basis)

    def get_caseA_basis_change_matrix(self, caseB_basis):
        """Return a matrix mapping coefficients from case B into this case A basis."""
        from src.molecular_structure.HundsCaseTransforms import caseB_to_caseA_matrix

        return caseB_to_caseA_matrix(caseB_basis, self)

    def change_basis_to_caseB(self, caseB_basis):
        return self.get_caseB_basis_change_matrix(caseB_basis)

    def change_basis_from_caseB(self, caseB_basis):
        return self.get_caseA_basis_change_matrix(caseB_basis)

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


class STM_HundsCaseA_Basis(HundsCaseA_Basis):
    def __init__(self, S_range, J_range, Sigma_range=(-100,100), Omega_range=(-100,100), k_range=(-100,100), m_range=(-100,100)):
        vectors = []
        S = S_range[0]
        while S <= S_range[1]:
            Sigma = -S
            while Sigma <= S:
                if Sigma_range[0] <= abs(Sigma) <= Sigma_range[1]:
                    J = J_range[0]
                    while J <= J_range[1]:
                        Omega = -J
                        while Omega <= J:
                            k = Omega - Sigma
                            if Omega_range[0] <= abs(Omega) <= Omega_range[1] and k_range[0] <= abs(k) <= k_range[1]:
                                m = -J
                                while m <= J:
                                    if m_range[0] <= m <= m_range[1]:
                                        vectors.append(HundsCaseA_State(S, Sigma, J, Omega, m))
                                    m += 1
                            Omega += 1
                        J += 1
                Sigma += 1
            S += 1
        AngularMomentumBasis.__init__(self, vectors, "STM Hund's case A basis, without nuclear spin")


class Linear_HundsCaseA_SpinProjectionState(BasisVector):
    def __init__(self, S, Sigma):
        super().__init__(f"S={S}, Sigma={Sigma}")
        self.S = S
        self.Sigma = Sigma
        self.quantum_numbers = {"S": S, "Σ": Sigma}


class Linear_HundsCaseA_State(HundsCaseA_State):
    def __init__(self, S, Sigma, J, m, uncoupled_basis=None, coeff=None):
        super().__init__(S, Sigma, J, Sigma, m)
        self.Lambda = 0
        self.k = 0
        self.quantum_numbers["k"] = 0
        self.reorder_quantum_numbers()
        if uncoupled_basis is not None and coeff is not None:
            self.set_defining_basis(uncoupled_basis, coeff)


class Linear_HundsCaseA_Basis(HundsCaseA_Basis):
    def __init__(self, S_range, J_range, Sigma_range=(-100,100), m_range=(-100,100)):
        self.rot_basis = Linear_RotationalBasis(J_range, m_range=m_range)
        spin_vectors = []
        S = S_range[0]
        while S <= S_range[1]:
            Sigma = -S
            while Sigma <= S:
                if Sigma_range[0] <= abs(Sigma) <= Sigma_range[1]:
                    spin_vectors.append(Linear_HundsCaseA_SpinProjectionState(S, Sigma))
                Sigma += 1
            S += 1
        self.spin_projection_basis = OrthogonalBasis(spin_vectors, "Hund's case A spin projection basis")
        self.uncoupled_basis = self.rot_basis.STM_basis * self.spin_projection_basis

        vectors = []
        basis_change_rows = []
        for rot_index, rot_state in enumerate(self.rot_basis.STM_basis):
            J = rot_state.R
            m = rot_state.mR
            for spin_index, spin_state in enumerate(self.spin_projection_basis):
                if abs(spin_state.Sigma) > J:
                    continue
                coeff = np.zeros(self.uncoupled_basis.dimension)
                coeff[rot_index * self.spin_projection_basis.dimension + spin_index] = 1
                state = Linear_HundsCaseA_State(
                    spin_state.S,
                    spin_state.Sigma,
                    J,
                    m,
                    uncoupled_basis=self.uncoupled_basis,
                    coeff=coeff,
                )
                vectors.append(state)
                basis_change_rows.append(coeff)

        AngularMomentumBasis.__init__(self, vectors, "Linear Hund's case A basis, without nuclear spin")
        self.tensor_basis = self.uncoupled_basis
        self.uncoupled_bases = [self.rot_basis.STM_basis, self.spin_projection_basis]
        self.basis_change_matrix = np.array(basis_change_rows)
        self.coupled = False

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
