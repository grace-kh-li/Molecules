from mpmath import isint

from quantum_mechanics.Operator import Operator
from RotationalStates import STM_RotationalBasis, STM_RotationalState
import numpy as np
from tools.WignerSymbols import wigner_3j
from tools.SphericalTensors import SphericalTensor_prolate

class DipoleOperator_separable(Operator):
    """ Electric dipole moment operator, for separable basis. It can contain vibronic, rotational, electronic spin and nuclear spin."""
    def __init__(self, basis, dipole_mol, sigma):
        tensor_components = []
        for comp in basis[0].tensor_components:
            tensor_components.append(type(comp))
        """ STM basis, rotational transitions """
        self.dipole_mol = dipole_mol # this should be a spherical tensor for the dipole moment in the molecule frame, aka <vibronic'|mu_mol|vibronic''>
        self.sigma = sigma # sigma = 0, 1, -1. This is the index in space-fixed frame.
        matrix = np.zeros((len(basis), len(basis)))
        for i, b in enumerate(basis):
            for i1, b1 in enumerate(basis):
                j,k,m = b.R, b.k, b.mR
                j1,k1,m1 = b1.R, b1.k, b1.mR
                for sigma1 in (-1,0,1):
                    matrix[i,i1] += dipole_mol[1][sigma1] * np.sqrt((2*j + 1) * (2*j1 + 1)) * wigner_3j(j,1,j1,-m,sigma,m1) * wigner_3j(j,1,j1,-k,sigma1,k1) * (-1)**(m + k)

        super().__init__(basis, matrix)


