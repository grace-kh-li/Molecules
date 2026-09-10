import numpy as np

from src.molecular_structure.HundsCaseA import STM_HundsCaseA_Basis
from src.molecular_structure.HundsCaseB import Linear_HundsCaseB_Basis
from src.molecular_structure.molecular_Hamiltonians import LinearMolecule as LinearHamiltonians
from src.quantum_mechanics.Operator import Operator


def wavenumber_to_MHz(k):
    if isinstance(k, list):
        return [wavenumber_to_MHz(item) for item in k]
    if isinstance(k, dict):
        return {key: wavenumber_to_MHz(k[key]) for key in k}
    return k * 299792458 * 100 / 1e6


CAOH_BLUE_MOT_CONSTANTS = {
    "lambda_m": 626e-9,
    "Gamma_rad_per_s": 2 * np.pi * 6.4e6,
    "mass_u": 57,
    "X": {
        # Constants from CaOH_package.jl, in MHz.
        "B": 10023.0841,
        "D": 1.154e-2,
        "gamma": 34.7593,
        "bF": 2.602,
        "c": 2.053,
        "B_z_default": 1e-5,
    },
    "A": {
        # Constants from CaOH_package.jl, converted from cm^-1 to MHz.
        "T": wavenumber_to_MHz(15998.122),
        "B": wavenumber_to_MHz(0.3412200),
        "Aso": wavenumber_to_MHz(66.8181),
        "p": wavenumber_to_MHz(-0.04287),
        "q": wavenumber_to_MHz(-0.3257e-3),
    },
}


CAOH_12_A_STATE_CONSTANTS = {
    **CAOH_BLUE_MOT_CONSTANTS,
    "X": {
        **CAOH_BLUE_MOT_CONSTANTS["X"],
        "B_z_default": 0.0,
    },
}


def _scaled(operator, coefficient):
    return operator * coefficient


def _identity_operator(basis, coefficient=1.0):
    return Operator(basis, coefficient * np.eye(basis.dimension, dtype=np.complex128))


def case_b_x_hamiltonian_terms(basis, constants):
    return {
        "rotation": _scaled(LinearHamiltonians.CaseB_RotationHamiltonian(basis), constants["B"]),
        "rotation_distortion": _scaled(
            LinearHamiltonians.CaseB_RotationDistortionHamiltonian(basis),
            constants.get("D", 0.0),
        ),
        "spin_rotation": _scaled(
            LinearHamiltonians.CaseB_SpinRotationLambda0Hamiltonian(basis),
            constants.get("gamma", 0.0),
        ),
    }


def case_a_hamiltonian_terms(basis, constants):
    q = constants.get("q", 0.0)
    return {
        "origin": _identity_operator(basis, constants.get("T", 0.0)),
        "rotation": _scaled(LinearHamiltonians.CaseA_RotationHamiltonian(basis), constants["B"]),
        "spin_orbit": _scaled(LinearHamiltonians.CaseA_SpinOrbitHamiltonian(basis), constants["Aso"]),
        "lambda_doubling_q": _scaled(LinearHamiltonians.CaseA_LambdaDoublingQHamiltonian(basis), q),
        "lambda_doubling_p2q": _scaled(
            LinearHamiltonians.CaseA_LambdaDoublingP2QHamiltonian(basis),
            constants.get("p", 0.0) + 2 * q,
        ),
    }


def sum_hamiltonian_terms(terms):
    values = list(terms.values())
    if not values:
        raise ValueError("At least one Hamiltonian term is required.")
    total = values[0]
    for term in values[1:]:
        total = total + term
    return total


class CaOH_molecule:
    """CaOH example with linear Hund's case B X-state and optional case A A-state terms.

    The X-state hyperfine constants are kept in the constants table, but are not added by default
    because the current runnable example basis does not include nuclear spin.
    """

    def __init__(self, N_range=(0, 3), m_range=(-100, 100), constants=None, include_A=True, A_J_range=None):
        self.name = "CaOH"
        self.constants = constants or CAOH_BLUE_MOT_CONSTANTS
        self.X_constants = self.constants["X"]
        self.A_constants = self.constants["A"]

        self.X_basis = Linear_HundsCaseB_Basis(
            N_range=N_range,
            S_range=(1 / 2, 1 / 2),
            m_range=m_range,
        )
        self.H_terms_X = case_b_x_hamiltonian_terms(self.X_basis, self.X_constants)
        self.H_rot_X = self.H_terms_X["rotation"]
        self.H_cd_X = self.H_terms_X["rotation_distortion"]
        self.H_SR_X = self.H_terms_X["spin_rotation"]
        self.H_X = sum_hamiltonian_terms(self.H_terms_X)
        self.Es_X, self.eigenstates_X = self.H_X.diagonalize()

        if include_A:
            if A_J_range is None:
                A_J_range = (1 / 2, max(1 / 2, N_range[1] + 1 / 2))
            self.A_basis = STM_HundsCaseA_Basis(
                S_range=(1 / 2, 1 / 2),
                J_range=A_J_range,
                k_range=(1, 1),
                m_range=m_range,
            )
            self.H_terms_A = case_a_hamiltonian_terms(self.A_basis, self.A_constants)
            self.H_origin_A = self.H_terms_A["origin"]
            self.H_rot_A = self.H_terms_A["rotation"]
            self.H_SO_A = self.H_terms_A["spin_orbit"]
            self.H_LD_q_A = self.H_terms_A["lambda_doubling_q"]
            self.H_LD_p2q_A = self.H_terms_A["lambda_doubling_p2q"]
            self.H_A = sum_hamiltonian_terms(self.H_terms_A)
            self.Es_A, self.eigenstates_A = self.H_A.diagonalize()
        else:
            self.A_basis = None
            self.H_terms_A = {}
            self.H_A = None
            self.Es_A = None
            self.eigenstates_A = None
