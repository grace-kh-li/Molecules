import numpy as np

from src.molecular_structure.HundsCaseB import HundsCaseB_Basis
from src.molecular_structure.RotationalStates import STM_RotationalBasis
from src.molecular_structure.RotationOperators import STM_RaisingOperator, STM_LoweringOperator, STM_R2_Operator, \
    STM_Ra_Operator
from src.molecular_structure.VibronicStates import VibronicState
from src.quantum_mechanics.Operator import Operator

class Rotational_Hamiltonian(Operator):
    def __init__(self, basis, A, BC_avg2, BC_diff4):
        """ The basis must be an STM basis. """
        # assert isinstance(basis, STM_RotationalBasis) or isinstance(basis, HundsCaseB_Basis)
        J_p = STM_RaisingOperator(basis)
        J_m = STM_LoweringOperator(basis)
        R2 = STM_R2_Operator(basis)
        Ra = STM_Ra_Operator(basis)
        H_rot = Ra * Ra * (A - BC_avg2) + R2 * BC_avg2 + (J_p * J_p + J_m * J_m) * BC_diff4
        super().__init__(basis, H_rot.matrix)

class Rotational_Hamiltonian_evr(Operator):
    def __init__(self, basis, A_dict, BC_avg2_dict, BC_diff4_dict):
        """
        The basis must be vibronic x STM. The parameters for each electronic state is given in the input
        dictionaries {electronic state name: parameter value}.
        """

        J_p = STM_RaisingOperator(basis)
        J_m = STM_LoweringOperator(basis)
        R2 = STM_R2_Operator(basis)
        Ra = STM_Ra_Operator(basis)
        Ra2_matrix = (Ra * Ra).matrix
        asym_matrix = (J_p * J_p + J_m * J_m).matrix

        vibronic_states = []
        for s in basis:
            elec = s.quantum_numbers["elec"]
            if elec not in vibronic_states:
                vibronic_states.append(elec)

        matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
        self.rot_Hamiltonians = {}
        for elec in vibronic_states:
            H_matrix = (
                Ra2_matrix * (A_dict[elec] - BC_avg2_dict[elec])
                + R2.matrix * BC_avg2_dict[elec]
                + asym_matrix * BC_diff4_dict[elec]
            )
            self.rot_Hamiltonians[elec] = Operator(basis, H_matrix)
            indices = [i for i, b in enumerate(basis) if b.quantum_numbers["elec"] == elec]
            matrix[np.ix_(indices, indices)] = H_matrix[np.ix_(indices, indices)]

        super().__init__(basis, matrix)
