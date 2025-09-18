from src.quantum_mechanics.Basis import *


class ElectronicState(BasisVector):
    def __init__(self, label, Lambda = 0, symmetry_group=None, irrep=None):
        self.Lambda = Lambda
        super().__init__(label,symmetry_group=symmetry_group, irrep=irrep)
        self.quantum_numbers = {"elec": label}