from src.quantum_mechanics.Basis import *
from RotationalStates import *

class AngularMomentumState(BasisVector):
    def __init__(self, J, m, J_symbol="J", m_symbol="m"):
        self.J = J
        self.m = m
        self.J_symbol = J_symbol
        self.m_symbol = m_symbol
        super().__init__(f"{J_symbol}={J}, {m_symbol}={m}")

class ElectronicSpinState(AngularMomentumState):
    def __init__(self, S, ms):
        self.S = S
        self.ms = ms
        super().__init__(S,ms, "S", "ms")

class NuclearSpinState(AngularMomentumState):
    def __init__(self, I, mI):
        self.I = I
        self.mI = mI
        super().__init__(I, mI, "I", "mI")

class ElectronicSpinBasis(OrthogonalBasis):
    def __init__(self, S_range, ms_range):
        vectors = []
        S = S_range[0]
        while S <= S_range[1]:
            ms = -S
            while ms <= S:
                if ms_range[0] <= ms <= ms_range[1]:
                    vectors.append(ElectronicSpinState(S,ms))
                ms += 1
            S += 1
        super().__init__(vectors,"ES basis")

class NuclearSpinBasis(OrthogonalBasis):
    def __init__(self, I_range, mI_range):
        vectors = []
        I = I_range[0]
        while I <= I_range[1]:
            mI = -I
            while mI <= I:
                if mI_range[0] <= mI <= mI_range[1]:
                    vectors.append(NuclearSpinState(I, mI))
                mI += 1
            I += 1
        super().__init__(vectors,"NS basis")