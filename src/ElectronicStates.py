from src.quantum_mechanics.Basis import *


class ElectronicState(BasisVector):
    def __init__(self, label, Lambda = 0, group=None, irrep=None):
        self.irrep = irrep
        self.group = group
        self.Lambda = Lambda
        super().__init__(label)