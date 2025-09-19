from src.quantum_mechanics.Operator import Operator
from RotationalStates import STM_RotationalBasis, STM_RotationalState, RotationalBasis, ATM_RotationalBasis
import numpy as np
from src.tools.WignerSymbols import wigner_3j


class DipoleOperator(Operator):
    """ Abstract class for dipole operators. """
    def __init__(self, sigma_space, basis, matrix, symmetry_group=None, irrep=None):
        self.sigma_space = sigma_space
        super().__init__(basis, matrix, symmetry_group, irrep)





# class DipoleOperator_STM(DipoleOperator):
#     """ Electric dipole moment operator, for STM basis """
#     def __init__(self, basis, dipole_mol, sigma_space, symmetry_group=None, irrep=None):
#         """ STM basis, rotational transitions """
#         matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
#         for sigma_mol in (-1, 0, 1):
#             D = D_matrix_conj(basis, sigma_space, sigma_mol)
#             for i, b in enumerate(basis):
#                 for i1, b1 in enumerate(basis):
#                     matrix[i,i1] += dipole_mol[1][sigma_mol] * D[i,i1]
#         super().__init__(sigma_space, basis, matrix, symmetry_group, irrep)


class DipoleOperator_evr(DipoleOperator):
    """ Electric dipole moment operator, for Vibronic x (STM or ATM) basis """
    def __init__(self, basis, dipole_mol, sigma_space, symmetry_group=None, irrep=None):
        matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
        rot_basis = basis.tensor_components[0]
        if isinstance(rot_basis, STM_RotationalBasis):
            STM_basis = rot_basis
        elif isinstance(rot_basis, ATM_RotationalBasis):
            STM_basis = rot_basis.STM_basis
        else:
            raise TypeError("The rotational state must be either STM or ATM!")

        for sigma_mol in (-1, 0, 1):
            D = D_matrix_conj(STM_basis, sigma_space, sigma_mol)
            d_ev = dipole_mol[1][sigma_mol]
            if isinstance(rot_basis, ATM_RotationalBasis):
                D = D.change_basis(rot_basis,rot_basis.STM_basis_change_matrix.conj().T)
            matrix += np.kron(D.matrix, d_ev.matrix)
        super().__init__(sigma_space, basis, matrix, symmetry_group, irrep)

class DipoleOperator_spin(DipoleOperator):
    """ Electric dipole moment operator, for electronic spin basis """
    def __init__(self, basis, sigma_space, symmetry_group=None, irrep=None):
        matrix = np.eye(len(basis), dtype=np.complex128)
        super().__init__(sigma_space, basis, matrix, symmetry_group, irrep)


class D_matrix_conj(Operator):
    """ Wigner matrix (D_sigma_space sigma_mol)^* """
    def __init__(self, basis, sigma_space, sigma_mol, symmetry_group=None, irrep=None):
        matrix = np.zeros((len(basis), len(basis)))
        for i, b in enumerate(basis):
            for i1, b1 in enumerate(basis):
                j, k, m = b.R, b.k, b.mR
                j1, k1, m1 = b1.R, b1.k, b1.mR
                matrix[i, i1] = np.sqrt((2 * j + 1) * (2 * j1 + 1)) * wigner_3j(j, 1, j1, -m, sigma_space, m1) * wigner_3j(j, 1, j1, -k, sigma_mol, k1) * (-1) ** (m + k)
        super().__init__(basis, matrix, symmetry_group, irrep)

