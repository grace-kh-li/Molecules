import numpy as np

from src.tools.WignerSymbols import wigner_3j


def _same(value1, value2, tol=1e-9):
    return abs(value1 - value2) < tol


def case_a_b_overlap(case_a_state, case_b_state):
    """Return <case A | case B> for matching symmetric-top Hund's case labels."""
    if not _same(case_a_state.S, case_b_state.S):
        return 0.0
    if not _same(case_a_state.J, case_b_state.J):
        return 0.0
    if not _same(case_a_state.m, case_b_state.m):
        return 0.0
    if not _same(case_a_state.k, case_b_state.k):
        return 0.0

    S = case_a_state.S
    Sigma = case_a_state.Sigma
    J = case_a_state.J
    Omega = case_a_state.Omega
    N = case_b_state.N
    k = case_b_state.k

    phase = (-1) ** int(round(S - Sigma))
    return phase * np.sqrt(2 * N + 1) * wigner_3j(J, S, N, Omega, -Sigma, -k)


def caseA_to_caseB_matrix(caseA_basis, caseB_basis):
    """Matrix mapping coefficients in a case A basis to a case B basis."""
    overlap = np.zeros((caseA_basis.dimension, caseB_basis.dimension), dtype=np.complex128)
    for i, case_a_state in enumerate(caseA_basis):
        for j, case_b_state in enumerate(caseB_basis):
            overlap[i, j] = case_a_b_overlap(case_a_state, case_b_state)
    return overlap.conj().T


def caseB_to_caseA_matrix(caseB_basis, caseA_basis):
    """Matrix mapping coefficients in a case B basis to a case A basis."""
    return caseA_to_caseB_matrix(caseA_basis, caseB_basis).conj().T

