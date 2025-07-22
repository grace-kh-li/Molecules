from Basis import *


class ElectronicState(BasisVector):
    def __init__(self, label, group=None, irrep=None):
        self.irrep = irrep
        self.group = group
        super().__init__(label)