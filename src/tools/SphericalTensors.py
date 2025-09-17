import numpy as np
from src.quantum_mechanics.Operator import *

class Irreducible_SphericalTensor:
    def __init__(self, k, coeffs=None, is_operator=False, operator_basis=None):
        self.rank = k
        if coeffs is None:
            if is_operator:
                self.coeffs = np.array([ZeroOperator(operator_basis) for _ in range(2*k+1)])
            else:
                self.coeffs = np.zeros(2*k + 1, dtype=np.complex128)
        else:
            self.coeffs = coeffs

    def __getitem__(self, i):
        return self.coeffs[i + self.rank]

    def __setitem__(self, i, val):
        self.coeffs[i + self.rank] = val


    def __mul__(self, other):
        if isinstance(other, Irreducible_SphericalTensor) and self.rank == other.rank:
            s = 0
            for p in range(-self.rank, self.rank + 1):
                s += (-1)**p * self[p] * other[-p]
            return s
        elif self.rank != other.rank:
            raise TypeError("Can not multiply Irreducible_SphericalTensor with different rank")
        elif isinstance(other, np.complex128) or isinstance(other, np.float64):
            return Irreducible_SphericalTensor(self.rank, self.coeffs * other)
        else:
            raise TypeError("Undefined multiplication.")

class SphericalTensor:
    def __init__(self, T, is_operator=False, operator_basis=None):
        if T.shape == (3, 3):
            T0 = Irreducible_SphericalTensor(0, is_operator=is_operator, operator_basis=operator_basis)
            T1 = Irreducible_SphericalTensor(1, is_operator=is_operator, operator_basis=operator_basis)
            T2 = Irreducible_SphericalTensor(2, is_operator=is_operator, operator_basis=operator_basis)
            self.tensors = [T0, T1, T2]
            T0[0] = (T[0,0] + T[1,1] + T[2,2]) * (-1/np.sqrt(3))
            T1[0] = (T[0,1] - T[1,0]) * (1j / np.sqrt(2))
            T1[-1] = ((T[1,2] - T[2,1]) -  (T[2,0] - T[0,2])*1j) * 1j/2
            T1[1] =  ((T[1,2] - T[2,1]) +  (T[2,0] - T[0,2])*1j)*(-1j/2)
            T2[0] =  (T[2,2]*2  - T[0,0] - T[1,1])*1/np.sqrt(6)
            T2[-1] = ((T[0,2] + T[2,0]) - (T[1,2] + T[2,1])*1j )*1/2
            T2[1] =  ((T[0,2] + T[2,0]) + (T[1,2] + T[2,1])*1j )*(-1/2)
            T2[-2] = ((T[0,0] - T[1,1]) - (T[0,1] + T[1,0])*1j)*(1/2)
            T2[2] =((T[0,0] - T[1,1]) +(T[0,1] + T[1,0])* 1j )*(1/2)

        if T.shape == (3,):
            T1 = Irreducible_SphericalTensor(1,is_operator=is_operator, operator_basis=operator_basis)
            self.tensors = [T1]
            T1[0] = T[2]
            T1[-1] = (T[0] - T[1]*1j) * 1/np.sqrt(2)
            T1[1] = (T[0] + T[1]*1j) *  (-1/np.sqrt(2) )

    def __getitem__(self, i):
        for t in self.tensors:
            if t.rank == i:
                return t
        return None

class SphericalTensor_prolate(SphericalTensor):
    """ converts Cartesian tensor in order abc into xyz, before turning it into a spherical tensor."""
    def __init__(self, T, is_operator=False, operator_basis=None):
        if T.shape == (3,): #a, b, c
            xyz = np.array([T[1],T[2],T[0]])
            super().__init__(xyz, is_operator=is_operator, operator_basis=operator_basis)
        if T.shape == (3,3):
            xyz = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    i1 = (i-1) % 3
                    j1 = (j-1) % 3
                    xyz[i1,j1] = T[i,j]
            super().__init__(xyz, is_operator=is_operator, operator_basis=operator_basis)