from src.RotationalStates import STM_RaisingOperator, STM_LoweringOperator, STM_R2_Operator,STM_Ra_Operator,STM_RotationalBasis
from src.quantum_mechanics.Operator import Operator

class Rotational_Hamiltonian(Operator):
    def __init__(self, basis, A, BC_avg2, BC_diff4):
        J_p = STM_RaisingOperator(basis)
        J_m = STM_LoweringOperator(basis)
        R2 = STM_R2_Operator(basis)
        Ra = STM_Ra_Operator(basis)
        H_rot = Ra * Ra * (A - BC_avg2) + R2 * BC_avg2 + (J_p * J_p + J_m * J_m) * BC_diff4
        super().__init__(basis, H_rot.matrix)