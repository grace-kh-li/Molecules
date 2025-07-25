import numpy as np

class Irreducible_SphericalTensor:
    def __init__(self, k, coeffs=None):
        self.rank = k
        if coeffs is None:
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
    def __init__(self, T):
        if T.shape == (3, 3):
            T0 = Irreducible_SphericalTensor(0)
            T1 = Irreducible_SphericalTensor(1)
            T2 = Irreducible_SphericalTensor(2)
            self.tensors = [T0, T1, T2]
            T0[0] = -1/np.sqrt(3) *(T[0,0] + T[1,1] + T[2,2])
            T1[0] = 1j / np.sqrt(2) * (T[0,1] - T[1,0])
            T1[-1] = 1j/2 * ((T[1,2] - T[2,1]) - 1j * (T[2,0] - T[0,2]))
            T1[1] = -1j/2 * ((T[1,2] - T[2,1]) + 1j * (T[2,0] - T[0,2]))
            T2[0] = 1/np.sqrt(6) * (2 * T[2,2] - T[0,0] - T[1,1])
            T2[-1] = 1/2 * ((T[0,2] + T[2,0]) - 1j * (T[1,2] + T[2,1]))
            T2[1] = -1/2 * ((T[0,2] + T[2,0]) + 1j * (T[1,2] + T[2,1]))
            T2[-2] = 1/2*((T[0,0] - T[1,1]) - 1j * (T[0,1] + T[1,0]))
            T2[2] = 1/2*((T[0,0] - T[1,1]) + 1j * (T[0,1] + T[1,0]))

        if T.shape == (3,):
            T1 = Irreducible_SphericalTensor(1)
            self.tensors = [T1]
            T1[0] = T[2,2]
            T1[-1] = 1/np.sqrt(2) * (T[0] - 1j*T[1])
            T1[1] = -1/np.sqrt(2) * (T[0] + 1j*T[1])

    def __getitem__(self, i):
        return self.tensors[i]