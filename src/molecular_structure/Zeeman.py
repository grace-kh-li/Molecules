from src.quantum_mechanics.Operator import Operator
import numpy as np
from src.quantum_mechanics.AngularMomentum import ElectronicSpinBasis, AngularMomentumOperator
from scipy.constants import physical_constants, h

class ZeemanHamiltonian_es(Operator):
    """ Zeeman component from electronic spin. Actual Hamiltonian = this operator * B_i.
    The unit will be in GHz / Gauss. """
    def __init__(self, basis, axis):
        gS = 2.00232 # electron g-factor
        mu_B = physical_constants["Bohr magneton"][0] / h * 1e-6 / 1e4 # GHz/Gauss

        if axis not in ("x", "y", "z", "X", "Y", "Z", 0, 1,-1):
            raise NotImplementedError("axis must be 0,-1,1, or x, y, z!")
        if isinstance(basis, ElectronicSpinBasis):
            matrix = AngularMomentumOperator(basis, axis).matrix * gS * mu_B
        else:
            matrix = np.eye(len(basis),dtype=complex)
        super().__init__(basis, matrix )