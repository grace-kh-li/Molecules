from Basis import *
from src import Basis


class ElectronicState(BasisVector):
    def __init__(self, label, group=None, irrep=None):
        self.label = label
        self.irrep = irrep