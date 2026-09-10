import numpy as np
from sympy import Rational

from src.group_theory.Group import LieAlgebra, LieAlgebraElement
from src.molecular_structure.CaF.CaF_package import CAF_BLUE_MOT_CONSTANTS, CaF_molecule, wavenumber_to_MHz
from src.molecular_structure.CaNH2.CaNH2_package import CaNH2_molecule
from src.molecular_structure.CaOH.CaOH_package import CAOH_BLUE_MOT_CONSTANTS, CaOH_molecule
from src.molecular_structure.HundsCaseA import HundsCaseA_Basis, Linear_HundsCaseA_Basis, STM_HundsCaseA_Basis
from src.molecular_structure.HundsCaseB import HundsCaseB_Basis, HundsCaseB_Basis_with_NS, Linear_HundsCaseB_Basis, \
    STM_HundsCaseB_Basis
from src.molecular_structure.molecular_Hamiltonians import LinearMolecule as LinearHamiltonians
from src.molecular_structure.RotationalStates import Linear_RotationalBasis
from src.quantum_mechanics.Basis import BasisVector, OrthogonalBasis, QuantumState
from src.tools.SphericalTensors import SphericalTensor_prolate
from src.tools.WignerSymbols import clear_wigner_cache, wigner_3j, wigner_6j, wigner_9j, wigner_cache_info


def test_quantum_state_arithmetic_preserves_basis_and_coefficients():
    basis = OrthogonalBasis([BasisVector("a"), BasisVector("b")])
    q1 = QuantumState("q1", np.array([1.0, 0.0]), basis)
    q2 = QuantumState("q2", np.array([0.0, 2.0]), basis)

    assert np.allclose((q1 + q2).coeff, [1.0, 2.0])
    assert np.allclose((q2 - q1).coeff, [-1.0, 2.0])
    assert np.allclose((3 * q1).coeff, [3.0, 0.0])
    assert np.allclose((q2 / 2).coeff, [0.0, 1.0])
    assert (q1 + q2).defining_basis == basis


def test_quantum_state_sorted_flag_is_not_replaced_by_builtin():
    basis = OrthogonalBasis([BasisVector("a"), BasisVector("b")])
    q = QuantumState("q", np.array([0.1, 0.9]), basis, sorted=False)

    assert q.sorted is False


def test_basis_vector_dot_and_direct_sum_work():
    b1 = OrthogonalBasis([BasisVector("a"), BasisVector("b")], name="left")
    b2 = OrthogonalBasis([BasisVector("c")], name="right")

    assert b1[0].dot(b1[0]) == 1
    assert b1[0].dot(b1[1]) == 0

    summed = b1 + b2
    assert len(summed) == 3
    assert [b.label for b in summed] == ["a", "b", "c"]
    assert summed[0].basis == summed


def test_spherical_tensor_prolate_matrix_conversion_is_silent(capsys):
    tensor = SphericalTensor_prolate(np.eye(3))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert tensor[0] is not None
    assert tensor[1] is not None
    assert tensor[2] is not None


def test_lie_bracket_returns_matrix_commutator():
    x = LieAlgebraElement("x", np.array([[0, 1], [1, 0]], dtype=complex))
    z = LieAlgebraElement("z", np.array([[1, 0], [0, -1]], dtype=complex))
    algebra = LieAlgebra("test", [x, z])

    bracket = algebra.Lie_bracket(x, z)

    assert bracket.Lie_algebra == algebra
    assert np.allclose(bracket.matrix, x.matrix @ z.matrix - z.matrix @ x.matrix)


def test_hunds_case_b_filters_j_and_m():
    basis = HundsCaseB_Basis(N_range=(0, 1), S_range=(0.5, 0.5), J_range=(1.5, 1.5), m_range=(-0.5, 0.5))

    assert len(basis) > 0
    assert all(b.J_total == 1.5 for b in basis)
    assert all(-0.5 <= b.m_total <= 0.5 for b in basis)
    assert basis.basis_change_matrix.shape[0] == len(basis)


def test_hunds_case_b_with_nuclear_spin_builds_f_basis():
    basis = HundsCaseB_Basis_with_NS(
        N_range=(0, 1),
        S_range=(0.5, 0.5),
        I_range=(0.5, 0.5),
        F_range=(0, 1),
        m_range=(-1, 1),
    )

    assert len(basis) > 0
    assert all(hasattr(b, "F") for b in basis)
    assert all("F" in b.quantum_numbers for b in basis)
    assert basis.basis_change_matrix.shape[0] == len(basis)


def test_linear_rotational_basis_expands_in_k_zero_stm_basis():
    basis = Linear_RotationalBasis(R_range=(0, 2))

    assert len(basis) == len(basis.STM_basis)
    assert all(stm_state.k == 0 for stm_state in basis.STM_basis)
    assert all(state.defining_basis == basis.STM_basis for state in basis)
    assert np.allclose(basis.STM_basis_change_matrix, np.eye(len(basis)))


def test_linear_hunds_case_b_uses_stm_k_zero_uncoupled_basis():
    basis = Linear_HundsCaseB_Basis(N_range=(0, 2), S_range=(0.5, 0.5))
    stm_basis, spin_basis = basis.uncoupled_bases

    assert all(state.k == 0 for state in basis)
    assert all(stm_state.k == 0 for stm_state in stm_basis)
    assert basis.tensor_basis.tensor_components == [stm_basis, spin_basis]
    assert basis.basis_change_matrix.shape == (len(basis), basis.tensor_basis.dimension)
    assert all(state.defining_basis == basis.tensor_basis for state in basis)
    assert isinstance(basis, HundsCaseB_Basis)


def test_linear_hunds_case_a_is_omega_equals_sigma_k_zero_basis():
    basis = Linear_HundsCaseA_Basis(S_range=(0.5, 0.5), J_range=(0, 2))
    stm_basis, sigma_basis = basis.uncoupled_bases

    assert all(state.Omega == state.Sigma for state in basis)
    assert all(state.k == 0 for state in basis)
    assert all(state.quantum_numbers["k"] == 0 for state in basis)
    assert all(abs(state.Sigma) <= state.J for state in basis)
    assert all(stm_state.k == 0 for stm_state in stm_basis)
    assert basis.tensor_basis.tensor_components == [stm_basis, sigma_basis]
    assert basis.basis_change_matrix.shape == (len(basis), basis.tensor_basis.dimension)
    assert isinstance(basis, HundsCaseA_Basis)


def test_stm_hunds_case_classes_exist_and_extend_case_bases():
    case_a = STM_HundsCaseA_Basis(S_range=(0.5, 0.5), J_range=(0.5, 1.5), k_range=(0, 0))
    case_b = STM_HundsCaseB_Basis(N_range=(0, 2), S_range=(0.5, 0.5), k_range=(0, 0))

    assert isinstance(case_a, HundsCaseA_Basis)
    assert isinstance(case_b, HundsCaseB_Basis)
    assert all(state.k == 0 for state in case_a)
    assert all(state.k == 0 for state in case_b)


def test_hunds_case_a_b_basis_change_round_trips_matched_stm_blocks():
    case_a = STM_HundsCaseA_Basis(S_range=(0.5, 0.5), J_range=(0.5, 0.5), k_range=(0, 0))
    case_b = STM_HundsCaseB_Basis(N_range=(0, 1), S_range=(0.5, 0.5), k_range=(0, 0), J_range=(0.5, 0.5))

    a_to_b = case_a.get_caseB_basis_change_matrix(case_b)
    b_to_a = case_b.get_caseA_basis_change_matrix(case_a)

    assert a_to_b.shape == (len(case_b), len(case_a))
    assert b_to_a.shape == (len(case_a), len(case_b))
    assert np.allclose(a_to_b, case_b.get_caseB_basis_change_matrix(case_a))
    assert np.allclose(b_to_a, case_a.get_caseA_basis_change_matrix(case_b))
    assert np.allclose(b_to_a @ a_to_b, np.eye(len(case_a)), atol=1e-12)
    assert np.allclose(a_to_b @ b_to_a, np.eye(len(case_b)), atol=1e-12)


def test_can_build_small_canh2_model_after_hamiltonian_cleanup():
    molecule = CaNH2_molecule(vibronic_states_to_include=("X", "A"), N_range=(0, 1))

    assert len(molecule.caseB_basis) == len(molecule.Es)
    assert molecule.H.matrix.shape == (len(molecule.caseB_basis), len(molecule.caseB_basis))
    assert np.all(np.diff(molecule.Es) >= 0)


def test_wigner_symbols_are_numeric_by_default_and_exact_on_request():
    assert isinstance(wigner_3j(1, 1, 1, 1, -1, 0), float)
    assert isinstance(wigner_6j(1, 1, 1, 1, 1, 1), float)
    assert isinstance(wigner_9j(1, 1, 1, 1, 1, 1, 1, 1, 1), float)

    assert wigner_3j(1, 1, 1, 1, -1, 0, numeric=False) ** 2 == Rational(1, 6)
    assert wigner_6j(1, 1, 1, 1, 1, 1, numeric=False) == Rational(1, 6)


def test_wigner_symbol_cache_reuses_repeated_calculations():
    clear_wigner_cache()

    first = wigner_6j(1, 1, 1, 1, 1, 1)
    second = wigner_6j(1, 1, 1, 1, 1, 1)
    cache_info = wigner_cache_info()["wigner_6j_float"]

    assert first == second
    assert cache_info.hits == 1
    assert cache_info.misses == 1


def test_linear_case_b_hamiltonian_terms_match_simple_values():
    basis = Linear_HundsCaseB_Basis(N_range=(0, 1), S_range=(0.5, 0.5))
    state = next(s for s in basis if s.N == 1 and s.J == 0.5 and s.m == -0.5)

    assert LinearHamiltonians.CaseB_Rotation(state, state) == 2
    assert LinearHamiltonians.CaseB_RotationDistortion(state, state) == -4
    assert np.isclose(LinearHamiltonians.CaseB_SpinRotation_Lambda0(state, state), -1.0)

    H_rot = LinearHamiltonians.CaseB_RotationHamiltonian(basis)
    H_sr = LinearHamiltonians.CaseB_SpinRotationLambda0Hamiltonian(basis)
    index = basis.get_index(state)
    assert H_rot[index, index] == 2
    assert np.isclose(H_sr[index, index], -1.0)


def test_linear_case_a_hamiltonian_terms_match_simple_values():
    basis = STM_HundsCaseA_Basis(
        S_range=(0.5, 0.5),
        J_range=(0.5, 0.5),
        Omega_range=(0.5, 0.5),
        m_range=(0.5, 0.5),
    )
    state = next(s for s in basis if s.Sigma == -0.5 and s.Omega == 0.5)

    assert np.isclose(LinearHamiltonians.CaseA_SpinOrbit(state, state), -0.5)
    assert np.isclose(LinearHamiltonians.CaseA_Rotation(state, state), 1.0)

    H_rot = LinearHamiltonians.CaseA_RotationHamiltonian(basis)
    H_so = LinearHamiltonians.CaseA_SpinOrbitHamiltonian(basis)
    index = basis.get_index(state)
    assert np.isclose(H_rot[index, index], 1.0)
    assert np.isclose(H_so[index, index], -0.5)


def test_all_linear_case_hamiltonian_operator_wrappers_construct():
    case_b = Linear_HundsCaseB_Basis(N_range=(0, 1), S_range=(0.5, 0.5))
    case_a = STM_HundsCaseA_Basis(
        S_range=(0.5, 0.5),
        J_range=(0.5, 0.5),
        Omega_range=(0.5, 0.5),
        m_range=(0.5, 0.5),
    )

    case_b_classes = [
        "CaseB_RotationHamiltonian",
        "CaseB_RotationDistortionHamiltonian",
        "CaseB_SpinRotationLambda0Hamiltonian",
        "CaseB_SpinRotationHamiltonian",
        "CaseB_HyperfineISHamiltonian",
        "CaseB_HyperfineDipolarHamiltonian",
        "CaseB_NuclearQuadrupoleHamiltonian",
        "CaseB_MagneticQuadrupoleHamiltonian",
        "CaseB_lDoublingHamiltonian",
        "CaseB_StarkOperator",
        "CaseB_ZeemanOperator",
        "CaseB_ZeemanNuclearOperator",
        "CaseB_ZeemanRotationOperator",
        "CaseB_TDMOperator",
        "CaseB_TDMVibrationalOperator",
        "CaseB_TDMMagneticOperator",
        "CaseB_PolarizabilityOperator",
        "CaseB_PolarizabilityParityOperator",
    ]
    case_a_classes = [
        "CaseA_RotationHamiltonian",
        "CaseA_RotationSigmaHamiltonian",
        "CaseA_RotationDeltaHamiltonian",
        "CaseA_SpinOrbitHamiltonian",
        "CaseA_SpinUncouplingHamiltonian",
        "CaseA_LambdaDoublingQHamiltonian",
        "CaseA_LambdaDoublingP2QHamiltonian",
        "CaseA_lDoublingHamiltonian",
        "CaseA_gKNonadiabaticHamiltonian",
        "CaseA_HyperfineILHamiltonian",
        "CaseA_HyperfineIFHamiltonian",
        "CaseA_HyperfineDipolarCHamiltonian",
        "CaseA_HyperfineDipolarDHamiltonian",
        "CaseA_RennerTellerHamiltonian",
        "CaseA_TDMOperator",
        "CaseA_ZeemanLOperator",
        "CaseA_ZeemanSOperator",
        "CaseA_ZeemanGLPrimeOperator",
    ]

    for name in case_b_classes:
        op = getattr(LinearHamiltonians, name)(case_b)
        assert op.matrix.shape == (len(case_b), len(case_b))
    for name in case_a_classes:
        op = getattr(LinearHamiltonians, name)(case_a)
        assert op.matrix.shape == (len(case_a), len(case_a))


def test_caf_constants_are_available_from_linked_package_values():
    constants = CAF_BLUE_MOT_CONSTANTS

    assert constants["lambda_m"] == 606e-9
    assert constants["X"]["B"] == 10023.0841
    assert constants["X"]["D"] == 4.8078e-7
    assert constants["X"]["gamma"] == 39.65895
    assert np.isclose(constants["A"]["T"], wavenumber_to_MHz(16526.750))
    assert np.isclose(constants["A"]["Aso"], wavenumber_to_MHz(71.429))


def test_caf_x_state_example_builds_linear_hunds_case_b_hamiltonian():
    caf = CaF_molecule(N_range=(0, 1), A_J_range=(0.5, 0.5))
    constants = CAF_BLUE_MOT_CONSTANTS["X"]

    assert len(caf.X_basis) == len(caf.Es_X)
    assert all(state.k == 0 for state in caf.X_basis)
    assert set(caf.H_terms_X) == {"rotation", "rotation_distortion", "spin_rotation"}
    assert np.allclose(
        caf.H_rot_X.matrix,
        constants["B"] * LinearHamiltonians.CaseB_RotationHamiltonian(caf.X_basis).matrix,
    )

    N0_energy = 0.0
    N1_rot = constants["B"] * 2 - constants["D"] * 4
    N1_J12_energy = N1_rot - constants["gamma"]
    N1_J32_energy = N1_rot + constants["gamma"] / 2

    assert np.isclose(caf.Es_X[0], N0_energy, atol=1e-8)
    assert np.isclose(caf.Es_X[1], N0_energy, atol=1e-8)
    assert np.count_nonzero(np.isclose(caf.Es_X, N1_J12_energy, atol=1e-8)) == 2
    assert np.count_nonzero(np.isclose(caf.Es_X, N1_J32_energy, atol=1e-8)) == 4
    assert caf.H_A.matrix.shape == (len(caf.A_basis), len(caf.A_basis))
    assert set(caf.H_terms_A) == {
        "origin",
        "rotation",
        "spin_orbit",
        "lambda_doubling_q",
        "lambda_doubling_p2q",
    }
    assert all(abs(state.k) == 1 for state in caf.A_basis)
    assert np.allclose(caf.H_origin_A.matrix, CAF_BLUE_MOT_CONSTANTS["A"]["T"] * np.eye(len(caf.A_basis)))


def test_caoh_constants_are_available_from_linked_package_values():
    constants = CAOH_BLUE_MOT_CONSTANTS

    assert constants["lambda_m"] == 626e-9
    assert constants["X"]["B"] == 10023.0841
    assert constants["X"]["D"] == 1.154e-2
    assert constants["X"]["gamma"] == 34.7593
    assert constants["X"]["bF"] == 2.602
    assert constants["X"]["c"] == 2.053
    assert np.isclose(constants["A"]["T"], wavenumber_to_MHz(15998.122))
    assert np.isclose(constants["A"]["Aso"], wavenumber_to_MHz(66.8181))


def test_caoh_x_state_example_builds_linear_hunds_case_b_hamiltonian():
    caoh = CaOH_molecule(N_range=(0, 1), A_J_range=(0.5, 0.5))
    constants = CAOH_BLUE_MOT_CONSTANTS["X"]

    assert len(caoh.X_basis) == len(caoh.Es_X)
    assert all(state.k == 0 for state in caoh.X_basis)
    assert set(caoh.H_terms_X) == {"rotation", "rotation_distortion", "spin_rotation"}
    assert np.allclose(
        caoh.H_SR_X.matrix,
        constants["gamma"] * LinearHamiltonians.CaseB_SpinRotationLambda0Hamiltonian(caoh.X_basis).matrix,
    )

    N0_energy = 0.0
    N1_rot = constants["B"] * 2 - constants["D"] * 4
    N1_J12_energy = N1_rot - constants["gamma"]
    N1_J32_energy = N1_rot + constants["gamma"] / 2

    assert np.isclose(caoh.Es_X[0], N0_energy, atol=1e-8)
    assert np.isclose(caoh.Es_X[1], N0_energy, atol=1e-8)
    assert np.count_nonzero(np.isclose(caoh.Es_X, N1_J12_energy, atol=1e-8)) == 2
    assert np.count_nonzero(np.isclose(caoh.Es_X, N1_J32_energy, atol=1e-8)) == 4
    assert caoh.H_A.matrix.shape == (len(caoh.A_basis), len(caoh.A_basis))
    assert set(caoh.H_terms_A) == {
        "origin",
        "rotation",
        "spin_orbit",
        "lambda_doubling_q",
        "lambda_doubling_p2q",
    }
    assert all(abs(state.k) == 1 for state in caoh.A_basis)
    assert np.allclose(caoh.H_origin_A.matrix, CAOH_BLUE_MOT_CONSTANTS["A"]["T"] * np.eye(len(caoh.A_basis)))
