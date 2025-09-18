from src.quantum_mechanics.Operator import Operator
import numpy as np
from src.tools.SphericalTensors import SphericalTensor
from src.tools.WignerSymbols import wigner_3j, wigner_6j

# class SpinRotationHamiltonian_PGopher(Operator):
#     def __init__(self, basis, epsilon_cartesian):
#         self.basis = basis # Hund's case B
#         self.epsilon_cartesian = epsilon_cartesian
#         T = SphericalTensor(epsilon_cartesian)
#         matrix = np.zeros((len(self.basis), len(self.basis)),dtype=complex)
#
#         for i, b1 in enumerate(self.basis):
#             for j, b2 in enumerate(self.basis):
#                 if b1.J != b2.J:
#                     continue
#                 if b1.m != b2.m:
#                     continue
#                 if b1.S != b2.S:
#                     continue
#
#                 S = b1.S
#                 J = b1.J
#                 N = b1.N
#                 N1 = b2.N
#                 K1 = b1.k
#                 K = b2.k
#
#                 eaa = epsilon_cartesian[0,0]
#                 ebb = epsilon_cartesian[1,1]
#                 ecc = epsilon_cartesian[2,2]
#
#                 # s = 0
#                 # for k in range(0,3):
#                 #     q_sum = 0
#                 #     for q in range(-k,k+1):
#                 #         q_sum += (-1)**(N1-K1) * wigner_3j(N1, k, N, -K1, q, K) * T[k][q]
#                 #     part = 1/2*((-1)**k * np.sqrt(N*(N+1)*(2*N+1)) * wigner_6j(1,1,k,N1,N,N)
#                 #             + np.sqrt(N1*(N1+1)*(2*N1+1)) * wigner_6j(1,1,k,N,N1,N1))
#                 #     s += (np.sqrt(2*k+1) * np.sqrt(S * (S+1) * (2*S+1) * (2*N+1) * (2*N1+1))
#                 #               * (-1)**(J+S+N1)) * wigner_6j(N,S,J,S,N1,1) * part * q_sum
#                 # matrix[i,j] = s
#
#                 # matrix element <N1,K1 | eaa*Eaa + ebb*Ebb + ecc*Ecc | N,K>
#                 def _s(x):
#                     # guard tiny negatives from roundoff inside sqrt
#                     return np.sqrt(np.maximum(x, 0.0))
#
#                 # ----- Eaa -----
#                 val_eaa = 0.0
#                 if K1 == K:
#                     if N1 == N-1 and N != 0:
#                         val_eaa = ( -_s(((N-K)*(N+K)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                         ((2*N-1)*(2*N+1)*(N+J+S+1))) * (N+J+S+1) * K ) / (2*N)
#                     elif N1 == N and N != 0:
#                         val_eaa = ( ((-N+J-S)*(N+J+S+2) + (-N+J+S)*(N+J-S)) * (K**2) ) / (4*N*(N+1))
#                     elif N1 == N+1:
#                         val_eaa = ( -_s(((N-K+1)*(N+K+1)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                         ((2*N+1)*(2*N+3)*(N+J+S+2))) * (N+J+S+2) * K ) / (2*(N+1))
#
#                 # ----- Ebb -----
#                 val_ebb = 0.0
#                 # K' = K-2
#                 if K1 == K-2:
#                     if N1 == N-1 and N != 0:
#                         val_ebb = ( (_s(((N*(N-1)+(-K+1)*(K-2))*(N+K)*(N+K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    + _s(((N*(N+1)-K*(K-1))*(N+K-1)*(N+K-2)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1))))*(N+J+S+1) ) / (16*N)
#                     elif N1 == N and N != 0:
#                         val_ebb = ( (_s((N*(N+1)+(-K+1)*(K-2))*(N-K+1)*(N+K))
#                                    + _s((N*(N+1)-K*(K-1))*(N-K+2)*(N+K-1)))
#                                    *((-N+J-S)*(N+J+S+2)+(-N+J+S)*(N+J-S)) ) / (32*N*(N+1))
#                     elif N1 == N+1:
#                         val_ebb = ( (-_s((((N+1)*(N+2)+(-K+1)*(K-2))*(N-K+1)*(N-K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                           ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     -_s(((N*(N+1)-K*(K-1))*(N-K+2)*(N-K+3)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                           ((2*N+1)*(2*N+3)*(N+J+S+2))))*(N+J+S+2) ) / (16*(N+1))
#                 # K' = K
#                 if K1 == K:
#                     if N1 == N-1 and N != 0:
#                         val_ebb = ( (_s(((N*(N-1)-K*(K-1))*(N+K)*(N+K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    - _s(((N*(N-1)-K*(K+1))*(N-K)*(N-K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    - _s(((N*(N+1)-K*(K-1))*(N-K)*(N-K+1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    + _s(((N*(N+1)-K*(K+1))*(N+K)*(N+K+1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1))))*(N+J+S+1) ) / (16*N)
#                     elif N1 == N and N != 0:
#                         val_ebb = ( (_s((N*(N+1)-K*(K-1))*(N-K+1)*(N+K))
#                                    + _s((N*(N+1)-K*(K+1))*(N-K)*(N+K+1)))
#                                    *((-N+J-S)*(N+J+S+2)+(-N+J+S)*(N+J-S)) ) / (16*N*(N+1))
#                     elif N1 == N+1:
#                         val_ebb = ( (-_s((((N+1)*(N+2)-K*(K-1))*(N-K+1)*(N-K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                           ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     + _s((((N+1)*(N+2)-K*(K+1))*(N+K+1)*(N+K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                           ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     + _s(((N*(N+1)-K*(K-1))*(N+K)*(N+K+1)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                           ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     - _s(((N*(N+1)-K*(K+1))*(N-K)*(N-K+1)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                           ((2*N+1)*(2*N+3)*(N+J+S+2))))*(N+J+S+2) ) / (16*(N+1))
#                 # K' = K+2
#                 if K1 == K+2:
#                     if N1 == N-1 and N != 0:
#                         val_ebb = ( (-_s(((N*(N-1)+(-K-1)*(K+2))*(N-K)*(N-K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                            ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                     -_s(((N*(N+1)-K*(K+1))*(N-K-1)*(N-K-2)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                            ((2*N-1)*(2*N+1)*(N+J+S+1))))*(N+J+S+1) ) / (16*N)
#                     elif N1 == N and N != 0:
#                         val_ebb = ( (_s((N*(N+1)+(-K-1)*(K+2))*(N-K)*(N+K+1))
#                                    + _s((N*(N+1)-K*(K+1))*(N-K-1)*(N+K+2)))
#                                    *((-N+J-S)*(N+J+S+2)+(-N+J+S)*(N+J-S)) ) / (32*N*(N+1))
#                     elif N1 == N+1:
#                         val_ebb = ( (_s((((N+1)*(N+2)+(-K-1)*(K+2))*(N+K+1)*(N+K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                          ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                    + _s(((N*(N+1)-K*(K+1))*(N+K+2)*(N+K+3)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                          ((2*N+1)*(2*N+3)*(N+J+S+2))))*(N+J+S+2) ) / (16*(N+1))
#
#                 # ----- Ecc -----
#                 val_ecc = 0.0
#                 # K' = K-2
#                 if K1 == K-2:
#                     if N1 == N-1 and N != 0:
#                         val_ecc = ( (-_s(((N*(N-1)+(-K+1)*(K-2))*(N+K)*(N+K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                            ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                     -_s(((N*(N+1)-K*(K-1))*(N+K-1)*(N+K-2)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                            ((2*N-1)*(2*N+1)*(N+J+S+1))))*(N+J+S+1) ) / (16*N)
#                     elif N1 == N and N != 0:
#                         val_ecc = ( (-_s((N*(N+1)+(-K+1)*(K-2))*(N-K+1)*(N+K))
#                                     -_s((N*(N+1)-K*(K-1))*(N-K+2)*(N+K-1)))
#                                    *((-N+J-S)*(N+J+S+2)+(-N+J+S)*(N+J-S)) ) / (32*N*(N+1))
#                     elif N1 == N+1:
#                         val_ecc = ( (_s((((N+1)*(N+2)+(-K+1)*(K-2))*(N-K+1)*(N-K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                          ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                    + _s(((N*(N+1)-K*(K-1))*(N-K+2)*(N-K+3)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                          ((2*N+1)*(2*N+3)*(N+J+S+2))))*(N+J+S+2) ) / (16*(N+1))
#                 # K' = K
#                 if K1 == K:
#                     if N1 == N-1 and N != 0:
#                         val_ecc = ( (_s(((N*(N-1)-K*(K-1))*(N+K)*(N+K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    - _s(((N*(N-1)-K*(K+1))*(N-K)*(N-K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    - _s(((N*(N+1)-K*(K-1))*(N-K)*(N-K+1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    + _s(((N*(N+1)-K*(K+1))*(N+K)*(N+K+1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1))))*(N+J+S+1) ) / (16*N)
#                     elif N1 == N and N != 0:
#                         val_ecc = ( (_s((N*(N+1)-K*(K-1))*(N-K+1)*(N+K))
#                                    + _s((N*(N+1)-K*(K+1))*(N-K)*(N+K+1)))
#                                    *((-N+J-S)*(N+J+S+2)+(-N+J+S)*(N+J-S)) ) / (16*N*(N+1))
#                     elif N1 == N+1:
#                         val_ecc = ( (-_s((((N+1)*(N+2)-K*(K-1))*(N-K+1)*(N-K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                            ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     + _s((((N+1)*(N+2)-K*(K+1))*(N+K+1)*(N+K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                            ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     + _s(((N*(N+1)-K*(K-1))*(N+K)*(N+K+1)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                            ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     - _s(((N*(N+1)-K*(K+1))*(N-K)*(N-K+1)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                            ((2*N+1)*(2*N+3)*(N+J+S+2))))*(N+J+S+2) ) / (16*(N+1))
#                 # K' = K+2
#                 if K1 == K+2:
#                     if N1 == N-1 and N != 0:
#                         val_ecc = ( (_s(((N*(N-1)+(-K-1)*(K+2))*(N-K)*(N-K-1)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1)))
#                                    + _s(((N*(N+1)-K*(K+1))*(N-K-1)*(N-K-2)*(-N+J+S+1)*(N-J+S)*(N+J-S)) /
#                                           ((2*N-1)*(2*N+1)*(N+J+S+1))))*(N+J+S+1) ) / (16*N)
#                     elif N1 == N and N != 0:
#                         val_ecc = ( (-_s((N*(N+1)+(-K-1)*(K+2))*(N-K)*(N+K+1))
#                                     -_s((N*(N+1)-K*(K+1))*(N-K-1)*(N+K+2)))
#                                    *((-N+J-S)*(N+J+S+2)+(-N+J+S)*(N+J-S)) ) / (32*N*(N+1))
#                     elif N1 == N+1:
#                         val_ecc = ( (-_s((((N+1)*(N+2)+(-K-1)*(K+2))*(N+K+1)*(N+K+2)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                            ((2*N+1)*(2*N+3)*(N+J+S+2)))
#                                     - _s(((N*(N+1)-K*(K+1))*(N+K+2)*(N+K+3)*(-N+J+S)*(N-J+S+1)*(N+J-S+1)) /
#                                            ((2*N+1)*(2*N+3)*(N+J+S+2))))*(N+J+S+2) ) / (16*(N+1))
#
#                 matrix[i, j] = eaa*val_eaa + ebb*val_ebb + ecc*val_ecc
#
#
#         super().__init__(basis, matrix)

from src.quantum_mechanics.Operator import Operator
import numpy as np
from src.tools.SphericalTensors import SphericalTensor
from src.tools.WignerSymbols import wigner_3j, wigner_6j

class SpinRotationHamiltonian(Operator):
    def __init__(self, basis, e_aa, e_bb, e_cc):
        self.basis = basis # Hund's case B
        self.e_aa = e_aa
        self.e_bb = e_bb
        self.e_cc = e_cc
        epsilon_cartesian = np.array([[e_bb,0,0],[0,e_cc,0],[0,0,e_aa]]) # yeah that's right, b is x, c is y, a is z. You're welcome. I just spent 4 hours debugging my spin-rotation expression that THIS was the problem.
        T = SphericalTensor(epsilon_cartesian)
        matrix = np.zeros((len(self.basis), len(self.basis)), dtype=complex)

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
                for k in range(0, 3):
                    q_sum = 0
                    for q in range(-k, k + 1):
                        q_sum += (-1) ** (N1 - K1) * wigner_3j(N1, k, N, -K1, q, K) * T[k][q]
                    part = 1 / 2 * ((-1) ** k * np.sqrt(N * (N + 1) * (2 * N + 1)) * wigner_6j(1, 1, k, N1, N, N)
                                    + np.sqrt(N1 * (N1 + 1) * (2 * N1 + 1)) * wigner_6j(1, 1, k, N, N1, N1))
                    s += (np.sqrt(2 * k + 1) * np.sqrt(S * (S + 1) * (2 * S + 1) * (2 * N + 1) * (2 * N1 + 1))
                          * (-1) ** (J + S + N1)) * wigner_6j(N, S, J, S, N1, 1) * part * q_sum
                matrix[i, j] = s

        super().__init__(basis, matrix)