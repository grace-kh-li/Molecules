import numpy as np

from src.quantum_mechanics.Operator import Operator

def check_other_qns(b1,b2,white_list=()):
    # if other quantum numbers like vibronic states are different, the matrix element is zero.
    is_zero = False
    for qn in b1.quantum_numbers:
        if b1.quantum_numbers[qn] != b2.quantum_numbers[qn] and qn not in white_list:
            is_zero = True
            break
    return is_zero

class STM_RaisingOperator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((basis.dimension, basis.dimension), dtype=complex)
        for i, b1 in enumerate(basis):
            for j, b2 in enumerate(basis):
                # if other quantum numbers like vibronic states are different, the matrix element is zero.
                is_zero = check_other_qns(b1, b2,("k",))
                if is_zero:
                    continue

                if hasattr(b1, "N"):
                    R = b1.N
                else:
                    R = b1.R

                if b1.k == b2.k + 1:
                    k = b2.k
                    matrix[i, j] = np.sqrt(R * (R + 1) - k * (k + 1))

        super().__init__(basis, matrix)

class STM_LoweringOperator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)

        for i, b1 in enumerate(basis):
            for j, b2 in enumerate(basis):
                # if other quantum numbers like vibronic states are different, the matrix element is zero.
                is_zero = check_other_qns(b1, b2,("k",))
                if is_zero:
                    continue

                if hasattr(b1, "N"):
                    R = b1.N
                else:
                    R = b1.R

                if b1.k == b2.k - 1:
                    k = b2.k
                    matrix[i, j] = np.sqrt(R * (R + 1) - k * (k - 1))

        super().__init__(basis, matrix)

class STM_R2_Operator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)

        for i, b1 in enumerate(basis):
            if hasattr(b1, "N"):
                R = b1.N
            else:
                R = b1.R

            matrix[i,i] = R * (R + 1)

        super().__init__(basis, matrix)

class STM_Ra_Operator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)
        for i, b1 in enumerate(basis):
            matrix[i,i] = b1.k
        super().__init__(basis, matrix)

class MShiftOperator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)

        for i, b1 in enumerate(basis):
            matrix[i,i] = b1.m_total

        super().__init__(basis, matrix)

class JShiftOperator(Operator):
    def __init__(self, basis):
        matrix = np.zeros((basis.dimension, basis.dimension),dtype=complex)
        if hasattr(basis[0], "J"):
            for i, b1 in enumerate(basis):
                matrix[i,i] = b1.J
        super().__init__(basis, matrix)