from src.quantum_mechanics.Operator import Operator
import numpy as np
from src.tools.SphericalTensors import SphericalTensor
from src.tools.WignerSymbols import wigner_3j, wigner_6j

class SpinRotationHamiltonian(Operator):
    def __init__(self, basis, epsilon_cartesian):
        self.basis = basis # Hund's case B
        self.epsilon_cartesian = epsilon_cartesian
        T = SphericalTensor(epsilon_cartesian)
        matrix = np.zeros((len(self.basis), len(self.basis)),dtype=complex)

        for i, b1 in enumerate(self.basis):
            for j, b2 in enumerate(self.basis):
                if b1.J != b2.J:
                    continue
                if b1.m != b2.m:
                    continue
                if b1.S != b2.S:
                    continue

                S = b1.S
                J = b1.J
                N1 = b1.N
                N = b2.N
                K1 = b1.k
                K = b2.k

                s = 0
                for k in range(0,3):
                    q_sum = 0
                    for q in range(-k,k+1):
                        q_sum += (-1)**(N1-K1) * wigner_3j(N1, k, N, -K1, q, K) * T[k][q]
                    part = 1/2*((-1)**k * np.sqrt(N*(N+1)*(2*N+1)) * wigner_6j(1,1,k,N1,N,N)
                            + np.sqrt(N1*(N1+1)*(2*N1+1)) * wigner_6j(1,1,k,N,N1,N1))
                    s += (np.sqrt(2*k+1) * np.sqrt(S * (S+1) * (2*S+1) * (2*N+1) * (2*N1+1))
                              * (-1)**(J+S+N1)) * wigner_6j(N,S,J,S,N1,1) * part * q_sum
                matrix[i,j] = s
        super().__init__(basis, matrix)