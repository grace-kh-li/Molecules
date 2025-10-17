from src.group_theory.Group import *
from src.molecular_structure.VibronicStates import VibronicState, OffsetOperator
from src.molecular_structure.RotationalStates import STM_RotationalBasis
from src.molecular_structure.molecular_Hamiltonians.SpinRotation import Spin_Rotation_Hamiltonian_evCaseB
from src.quantum_mechanics.AngularMomentum import ElectronicSpinBasis
from src.quantum_mechanics.Basis import OrthogonalBasis
from src.molecular_structure.molecular_Hamiltonians.Rotational_Hamiltonian import *
from src.quantum_mechanics.Operator import ZeroOperator
from src.tools.SphericalTensors import Irreducible_SphericalTensor


def wavenumber_to_Hz(k):
    if isinstance(k, list):
        return [wavenumber_to_Hz(item) for item in k]
    if isinstance(k, dict):
        return {key: wavenumber_to_Hz(k[key]) for key in k}
    return k * 299792458 * 100

def wavenumber_to_GHz(k):
    if isinstance(k, list):
        return [wavenumber_to_GHz(item) for item in k]
    if isinstance(k, dict):
        return {key: wavenumber_to_GHz(k[key]) for key in k}
    return k * 299792458 * 100 / 1e9

class ATM_molecule:
    def __init__(self, name, group):
        self.evr_basis = None
        self.caseB_basis = None
        self.es_basis = None
        self.rot_basis = None
        self.vibronic_d = None
        self.vibronic_basis = None
        self.vibronic_states = []
        self.name = name
        self.group = group

    def include_vibronic_states(self, labels, irreps, TDM_matrices):
        from src.tools.SphericalTensors import SphericalTensor_prolate, Irreducible_SphericalTensor
        self.vibronic_states = []
        for i, label in enumerate(labels):
            state = VibronicState(label, symmetry_group=self.group, irrep=irreps[i])
            self.vibronic_states.append(state)

        self.vibronic_basis = OrthogonalBasis(self.vibronic_states,"Vibronic")

        vibronic_d_a = Operator(self.vibronic_basis, np.array(TDM_matrices[0]))
        vibronic_d_b = Operator(self.vibronic_basis, np.array(TDM_matrices[1]))
        vibronic_d_c = Operator(self.vibronic_basis, np.array(TDM_matrices[2]))

        # the transition dipole moment in molecule frame as a spherical tensor
        self.vibronic_d = SphericalTensor_prolate(np.array([vibronic_d_a, vibronic_d_b, vibronic_d_c]), is_operator=True,
                                             operator_basis=self.vibronic_basis)


    def include_rotational_states(self, N_range):
        self.rot_basis = STM_RotationalBasis(R_range=N_range)


    def include_es_states(self, S=1/2):
        self.es_basis = ElectronicSpinBasis(S_range=(S,S))


    def setup_caseB_basis(self):
        self.evr_basis = self.rot_basis * self.vibronic_basis
        basis = self.evr_basis.couple(self.es_basis)
        basis.rename_symbols("J", "m")  # rename angular momentum symbols.
        self.caseB_basis = basis

    def set_rotational_params(self, A_dict, BC_avg2_dict, BC_diff4_dict, units="GHz"):
        if units == "wavenumber" or units == "cm-1":
            for key in A_dict:
                A_dict[key] = wavenumber_to_GHz(A_dict[key])
                BC_avg2_dict[key] = wavenumber_to_GHz(BC_avg2_dict[key])
                BC_diff4_dict[key] = wavenumber_to_GHz(BC_diff4_dict[key])
        self.A_dict = A_dict
        self.BC_avg2_dict = BC_avg2_dict
        self.BC_diff4_dict = BC_diff4_dict
        self.H_rot = Rotational_Hamiltonian_evr(self.caseB_basis, self.A_dict, self.BC_avg2_dict, self.BC_diff4_dict)

    def set_spin_rotation_params(self, e_aa_dict, e_bb_dict, e_cc_dict, units="GHz"):
        if units == "wavenumber" or units == "cm-1":
            for key in e_aa_dict:
                e_aa_dict[key] = wavenumber_to_GHz(e_aa_dict[key])
                e_bb_dict[key] = wavenumber_to_GHz(e_bb_dict[key])
                e_cc_dict[key] = wavenumber_to_GHz(e_cc_dict[key])
        self.e_aa_dict = e_aa_dict
        self.e_bb_dict = e_bb_dict
        self.e_cc_dict = e_cc_dict
        self.H_SR = Spin_Rotation_Hamiltonian_evCaseB(self.caseB_basis, self.e_aa_dict, self.e_bb_dict, self.e_cc_dict)


    def set_offset_params(self, offset_dict, units="GHz"):
        if units == "wavenumber" or units == "cm-1":
            for key in offset_dict:
                offset_dict[key] = wavenumber_to_GHz(offset_dict[key])
        self.offset_dict = offset_dict
        self.H_offset = OffsetOperator(self.caseB_basis, self.offset_dict)

    def solve_Hamiltonian(self, spin_rotation=True, m_shift=True):
        from src.molecular_structure.RotationOperators import MShiftOperator

        H = ZeroOperator(self.caseB_basis)
        H += self.H_rot

        if spin_rotation:
            H += self.H_SR
        if m_shift:
            Z = MShiftOperator(self.caseB_basis) * 1e-5  # + JShiftOperator(basis) * 1e-5
            H += Z

        H += self.H_offset
        self.H = H


        from src.molecular_structure.RotationalStates import rename_ATM_states

        Es, states = H.diagonalize()
        rename_ATM_states(states)
        self.Es = Es
        self.eigenstates = states

    def calculate_TDMs(self):
        from src.molecular_structure.TDMs import DipoleOperator_evr, DipoleOperator_spin

        TDM = Irreducible_SphericalTensor(1, is_operator=True, operator_basis=self.caseB_basis)

        for sigma_d in (-1, 0, 1):
            d_evr = DipoleOperator_evr(self.evr_basis, self.vibronic_d, sigma_d)
            d_es = DipoleOperator_spin(self.es_basis, sigma_d)
            d_uncoupled = d_evr.tensor(d_es)
            d = d_uncoupled.change_basis(self.caseB_basis, self.caseB_basis.get_basis_change_matrix())
            TDM[sigma_d] = d

        self.TDMs = TDM

class CaNH2_molecule(ATM_molecule):
    def __init__(self, vibronic_states_to_include=("X","A"), N_range=(0,2)):
        group = C2vGroup()
        super().__init__("CaNH2", group)
        A1 = C2v_A1_representation(group)
        A2 = C2v_A2_representation(group)
        B1 = C2v_B1_representation(group)
        B2 = C2v_B2_representation(group)

        TDM_matrices_full = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [1, 0, 0], [0, 0, 0]],[[0, 0, 1], [0, 0, 0], [1, 0, 0]]]
        irreps_full = {"X": A1, "A": B2, "B": B1}

        mapping = {"X":0, "A":1, "B":2}

        irreps = []
        for s in vibronic_states_to_include:
            irreps.append(irreps_full[s])

        l = len(vibronic_states_to_include)
        self.TDM_matrices = [np.zeros((l,l),dtype=np.complex128) for _ in range(3)]
        for axis, m in enumerate(self.TDM_matrices):
            for i in range(l):
                for j in range(l):
                    ii = mapping[vibronic_states_to_include[i]]
                    jj= mapping[vibronic_states_to_include[j]]
                    m[i,j] = TDM_matrices_full[axis][ii][jj]

        self.include_vibronic_states(vibronic_states_to_include,irreps, self.TDM_matrices)
        self.include_rotational_states(N_range)
        self.include_es_states(1/2)
        self.setup_caseB_basis()
        self.setup_params()

        self.solve_Hamiltonian()
        self.calculate_TDMs()

    def setup_params(self):
        A_dict = {}
        BC_avg2_dict = {}
        BC_diff4_dict = {}

        e_aa_dict = {}
        e_bb_dict = {}
        e_cc_dict = {}

        offset_dict = {}

        # X state constants
        A_dict["X"] = wavenumber_to_GHz(13.05744)
        BC_avg2_dict["X"] = wavenumber_to_GHz(0.296652)
        BC_diff4_dict["X"] = wavenumber_to_GHz(1.8894e-3)
        e_aa_dict["X"] = 45.7e-3
        e_bb_dict["X"] = 32.063e-3
        e_cc_dict["X"] = 41.110e-3
        offset_dict["X"] = 0.0

        # A state constants
        A_dict["A"] = wavenumber_to_GHz(11.44854)
        BC_avg2_dict["A"] = wavenumber_to_GHz(0.303107)
        BC_diff4_dict["A"] = wavenumber_to_GHz(1.958e-3)
        e_aa_dict["A"] = wavenumber_to_GHz(8.2369)
        e_bb_dict["A"] = wavenumber_to_GHz(3.0534e-2 - 1.2617e-2 * 2)
        e_cc_dict["A"] = wavenumber_to_GHz(3.0534e-2 + 1.2617e-2 * 2)
        offset_dict["A"] = (wavenumber_to_Hz(15464.36739) - 192e6) / 1e9

        # B state constants
        A_dict["B"] = wavenumber_to_GHz(14.3664)
        BC_avg2_dict["B"] = wavenumber_to_GHz(0.301693)
        BC_diff4_dict["B"] = wavenumber_to_GHz(4.68e-3)
        e_aa_dict["B"] = wavenumber_to_GHz(-7.5472)
        e_bb_dict["B"] = wavenumber_to_GHz(2.083e-2 - 8.66e-3 * 2)
        e_cc_dict["B"] = wavenumber_to_GHz(2.083e-2 + 8.66e-3 * 2)
        offset_dict["B"] = (wavenumber_to_Hz(15885.28188) - 129e6) / 1e9

        self.set_rotational_params(A_dict,BC_avg2_dict,BC_diff4_dict,units="wavenumbers")
        self.set_spin_rotation_params(e_aa_dict,e_bb_dict,e_cc_dict,units="wavenumbers")
        self.set_offset_params(offset_dict,units="GHz")

