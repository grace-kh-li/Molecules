import numpy as np

from src.quantum_mechanics.Operator import Operator
from src.tools.WignerSymbols import wigner_3j, wigner_6j, wigner_9j


T_KQ = {
    (0, 0): -2 / np.sqrt(3),
    (2, 0): -2 / np.sqrt(6),
}


def delta(a, b, tol=1e-9):
    return abs(a - b) < tol


def phase(x):
    return (-1) ** int(round(float(x)))


def safe_sqrt(x):
    if x < 0 and abs(x) < 1e-12:
        x = 0.0
    return np.sqrt(x)


def qn(state, *names, default=0.0):
    aliases = {
        "Lambda": ("Lambda", "Λ"),
        "ell": ("ell", "ℓ"),
        "Sigma": ("Sigma", "Σ"),
    }
    for name in names:
        for candidate in aliases.get(name, (name,)):
            if hasattr(state, candidate):
                return getattr(state, candidate)
            if hasattr(state, "quantum_numbers") and candidate in state.quantum_numbers:
                return state.quantum_numbers[candidate]
    return default


def caseB_values(state):
    N = qn(state, "N", "R")
    K = qn(state, "K", "k")
    Lambda = qn(state, "Lambda", default=K)
    ell = qn(state, "ell", default=K - Lambda)
    J = qn(state, "J", "J_total")
    S = qn(state, "S")
    I = qn(state, "I")
    F = qn(state, "F", default=J)
    M = qn(state, "M", "mF", "m", "m_total")
    return {
        "v1": qn(state, "v_1"),
        "v2": qn(state, "v_2"),
        "v3": qn(state, "v_3"),
        "S": S,
        "I": I,
        "Lambda": Lambda,
        "ell": ell,
        "K": K,
        "N": N,
        "J": J,
        "F": F,
        "M": M,
    }


def caseA_values(state):
    S = qn(state, "S")
    Sigma = qn(state, "Sigma")
    J = qn(state, "J", "J_total")
    P = qn(state, "P", "Omega", default=Sigma)
    K = qn(state, "K", "k", default=P - Sigma)
    Lambda = qn(state, "Lambda", default=K)
    ell = qn(state, "ell", default=K - Lambda)
    I = qn(state, "I")
    F = qn(state, "F", default=J)
    M = qn(state, "M", "mF", "m", "m_total")
    return {
        "v1": qn(state, "v_1"),
        "v2": qn(state, "v_2"),
        "v3": qn(state, "v_3"),
        "ell": ell,
        "Lambda": Lambda,
        "K": K,
        "I": I,
        "S": S,
        "Sigma": Sigma,
        "J": J,
        "P": P,
        "F": F,
        "M": M,
    }


def same(values, values_p, *keys):
    return all(delta(values[key], values_p[key]) for key in keys)


class MatrixElementOperator(Operator):
    def __init__(self, basis, matrix_element, *args):
        matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
        for i, state in enumerate(basis):
            for j, state_p in enumerate(basis):
                matrix[i, j] = matrix_element(state, state_p, *args)
        super().__init__(basis, matrix)


def CaseB_Rotation(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "K", "Lambda", "ell", "S", "I", "N", "J", "F", "M"):
        return 0.0
    return v["N"] * (v["N"] + 1) - v["Lambda"] ** 2


def CaseB_RotationDistortion(state, state_p):
    rot = CaseB_Rotation(state, state_p)
    if rot == 0.0:
        return 0.0
    return -(caseB_values(state)["N"] * (caseB_values(state)["N"] + 1) - caseB_values(state)["Lambda"] ** 2) ** 2


def CaseB_SpinRotation_Lambda0(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "K", "Lambda", "ell", "S", "I", "N", "J", "F", "M"):
        return 0.0
    S, N, J = v["S"], v["N"], v["J"]
    return (
        phase(N + S + J)
        * safe_sqrt(S * (S + 1) * (2 * S + 1) * N * (N + 1) * (2 * N + 1))
        * wigner_6j(S, N, J, N, S, 1)
    )


def CaseB_SpinRotation(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "J", "F", "M"):
        return 0.0
    S, N, Np, Jp = v["S"], v["N"], vp["N"], vp["J"]
    K, Kp = v["K"], vp["K"]
    total = 0.0
    for k in range(3):
        for q in (-1, 0, 1):
            tkq = T_KQ.get((k, q), 0.0)
            if tkq == 0.0:
                continue
            part = (
                phase(k) * safe_sqrt(Np * (Np + 1) * (2 * Np + 1)) * wigner_6j(1, 1, k, N, Np, Np)
                + safe_sqrt(N * (N + 1) * (2 * N + 1)) * wigner_6j(1, 1, k, Np, N, N)
            )
            total += safe_sqrt(2 * k + 1) * part * wigner_3j(N, k, Np, -K, q, Kp) * tkq
    return (
        0.5
        * phase(Jp + S + N)
        * phase(N - K)
        * safe_sqrt(S * (S + 1) * (2 * S + 1) * (2 * N + 1) * (2 * Np + 1))
        * wigner_6j(Np, S, Jp, S, N, 1)
        * total
    )


def CaseB_Hyperfine_IS(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "K", "Lambda", "ell", "S", "I", "N", "F", "M"):
        return 0.0
    S, I, Np, J, Jp, Fp = v["S"], v["I"], vp["N"], v["J"], vp["J"], vp["F"]
    return (
        phase(Np + S + J)
        * phase(Jp + I + Fp + 1)
        * safe_sqrt((2 * Jp + 1) * (2 * J + 1) * S * (S + 1) * (2 * S + 1) * I * (I + 1) * (2 * I + 1))
        * wigner_6j(I, J, Fp, Jp, I, 1)
        * wigner_6j(S, J, Np, Jp, S, 1)
    )


def CaseB_Hyperfine_Dipolar(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "Lambda", "ell", "F", "M"):
        return 0.0
    S, I = v["S"], v["I"]
    N, Np, J, Jp, F, Fp, K, Kp = v["N"], vp["N"], v["J"], vp["J"], v["F"], vp["F"], v["K"], vp["K"]
    return (
        np.sqrt(30)
        * phase(N - K)
        * phase(Jp + I + F + 1)
        * wigner_6j(I, J, Fp, Jp, I, 1)
        * wigner_9j(N, Np, 2, S, S, 1, J, Jp, 1)
        * wigner_3j(N, 2, Np, -K, 0, Kp)
        * safe_sqrt(S * (S + 1) * (2 * S + 1) * I * (I + 1) * (2 * I + 1) * (2 * J + 1) * (2 * Jp + 1) * (2 * N + 1) * (2 * Np + 1))
    )


def CaseB_NuclearQuadrupole(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "F", "M"):
        return 0.0
    I = v["I"]
    if I == 0:
        return 0.0
    N, Np, J, Jp, S, F, K, Kp = v["N"], vp["N"], v["J"], vp["J"], v["S"], v["F"], v["K"], vp["K"]
    return (
        0.25
        * phase(J + I + F)
        * phase(Np + S + J)
        * safe_sqrt((I + 1) * (2 * I + 1) * (2 * I + 3) / (I * (I + 1)))
        * safe_sqrt((2 * Jp + 1) * (2 * J + 1) * (2 * Np + 1) * (2 * N + 1))
        * wigner_6j(I, Jp, F, J, I, 2)
        * wigner_6j(Np, Jp, S, J, N, 2)
        * phase(Np - Kp)
        * wigner_3j(Np, 2, N, -Kp, 0, K)
    )


def CaseB_MagneticQuadrupole(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "F", "M"):
        return 0.0
    F, J, I = v["F"], v["J"], v["I"]
    return 0.5 * (F * (F + 1) - J * (J + 1) - I * (I + 1))


def CaseB_lDoubling(state, state_p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "Lambda", "S", "I", "N", "J", "F", "M"):
        return 0.0
    if not delta(abs(vp["ell"] - v["ell"]), 2):
        return 0.0
    N, Np, K, Kp = v["N"], vp["N"], v["K"], vp["K"]
    return (
        phase(N - K)
        / (2 * np.sqrt(6))
        * safe_sqrt((2 * N - 1) * (2 * N) * (2 * N + 1) * (2 * N + 2) * (2 * N + 3))
        * sum(wigner_3j(N, 2, Np, -K, 2 * q, Kp) for q in (-1, 1))
    )


def CaseB_Stark(state, state_p, p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "ell", "v1", "v2", "v3"):
        return 0.0
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, N, Np, S, I, K, Kp = v["J"], vp["J"], v["N"], vp["N"], v["S"], v["I"], v["K"], vp["K"]
    return (
        -phase(p)
        * phase(F - M)
        * wigner_3j(F, 1, Fp, -M, p, Mp)
        * phase(J + I + Fp + 1)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_6j(Jp, Fp, I, F, J, 1)
        * phase(N + S + Jp + 1)
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
        * wigner_6j(Np, Jp, S, J, N, 1)
        * phase(N - K)
        * safe_sqrt((2 * N + 1) * (2 * Np + 1))
        * wigner_3j(N, 1, Np, -K, 0, Kp)
    )


def CaseB_Zeeman(state, state_p, p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "ell", "Lambda", "K", "N"):
        return 0.0
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, N, S, I = v["J"], vp["J"], v["N"], v["S"], v["I"]
    return (
        phase(p)
        * phase(F - M)
        * wigner_3j(F, 1, Fp, -M, p, Mp)
        * phase(Jp + I + F + 1)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_6j(J, F, I, Fp, Jp, 1)
        * phase(S + N + Jp + 1)
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1) * S * (S + 1) * (2 * S + 1))
        * wigner_6j(S, J, N, Jp, S, 1)
    )


def CaseB_ZeemanNuclear(state, state_p, p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "ell", "Lambda", "K", "N"):
        return 0.0
    I = v["I"]
    F, Fp, M, Mp, J = v["F"], vp["F"], v["M"], vp["M"], v["J"]
    return (
        phase(p)
        * phase(F - M)
        * wigner_3j(F, 1, Fp, -M, p, Mp)
        * phase(J + I + F + 1)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_6j(I, Fp, J, F, I, 1)
        * safe_sqrt(I * (I + 1) * (2 * I + 1))
    )


def CaseB_ZeemanRotation(state, state_p, p=0):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "Lambda", "ell", "K", "N", "J", "F", "M"):
        return 0.0
    return v["M"]


def CaseB_SigmaExpectation(state):
    v = caseB_values(state)
    total = 0.0
    S, Lambda, N, J = v["S"], v["Lambda"], v["N"], v["J"]
    sigma = -S
    while sigma <= S:
        Omega = Lambda + sigma
        total += sigma * (2 * N + 1) * wigner_3j(J, S, N, Omega, -sigma, -Lambda) ** 2
        sigma += 1
    return total


def CaseB_TDM(state, state_p, p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "ell"):
        return 0.0
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, N, Np, S, I, K, Kp = v["J"], vp["J"], v["N"], vp["N"], v["S"], v["I"], v["K"], vp["K"]
    return (
        -phase(p)
        * phase(F - M)
        * wigner_3j(F, 1, Fp, -M, -p, Mp)
        * phase(Jp + I + F + 1)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_6j(J, F, I, Fp, Jp, 1)
        * phase(Np + S + J + 1)
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
        * wigner_6j(N, J, S, Jp, Np, 1)
        * phase(N - K)
        * safe_sqrt((2 * N + 1) * (2 * Np + 1))
        * sum(wigner_3j(N, 1, Np, -K, q, Kp) for q in (-1, 0, 1))
    )


def CaseB_TDMVibrational(state, state_p, p):
    v, vp = caseB_values(state), caseB_values(state_p)
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, N, Np, S, I, K, Kp = v["J"], vp["J"], v["N"], vp["N"], v["S"], v["I"], v["K"], vp["K"]
    return (
        -phase(p)
        * phase(F - M)
        * wigner_3j(F, 1, Fp, -M, p, Mp)
        * phase(J + I + Fp + 1)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_6j(Jp, Fp, I, F, J, 1)
        * phase(N + S + Jp + 1)
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
        * wigner_6j(Np, Jp, S, J, N, 1)
        * phase(N - K)
        * safe_sqrt((2 * N + 1) * (2 * Np + 1))
        * sum(wigner_3j(N, 1, Np, -K, q, Kp) for q in (-1, 0, 1))
    )


def CaseB_TDMMagnetic(state, state_p, p):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "Lambda", "N"):
        return 0.0
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, N, S, I = v["J"], vp["J"], v["N"], v["S"], v["I"]
    return (
        phase(p)
        * phase(Fp - Mp)
        * wigner_3j(Fp, 1, F, -Mp, -p, M)
        * phase(Jp + I + F + 1)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_6j(Jp, Fp, I, F, J, 1)
        * phase(N + S + Jp + 1)
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1) * S * (S + 1) * (2 * S + 1))
        * wigner_6j(S, Jp, N, J, S, 1)
    )


def polarization_tensor(K, P, eps):
    eps_m1, eps_0, eps_p1 = eps[0], eps[1], eps[2]
    if P == 0:
        if K == 0:
            return 1.0
        if K == 1:
            return eps_p1 * np.conj(eps_p1) - eps_m1 * np.conj(eps_m1)
        if K == 2:
            return -0.5 * (1 - 3 * eps_0 * np.conj(eps_0))
    if P == 1:
        if K == 1:
            return -(eps_0 * np.conj(eps_m1) + np.conj(eps_0) * eps_p1)
        if K == 2:
            return np.sqrt(3 / 2) * (-eps_0 * np.conj(eps_m1) + np.conj(eps_0) * eps_p1)
    if P == -1:
        if K == 1:
            return eps_0 * np.conj(eps_p1) + np.conj(eps_0) * eps_m1
        if K == 2:
            return np.sqrt(3 / 2) * (-eps_0 * np.conj(eps_p1) + np.conj(eps_0) * eps_m1)
    if P == 2 and K == 2:
        return -np.sqrt(3 / 2) * np.conj(eps_m1) * eps_p1
    if P == -2 and K == 2:
        return -np.sqrt(3 / 2) * np.conj(eps_p1) * eps_m1
    return 0.0


def CaseB_Polarizability(state, state_p, alpha, eps):
    v, vp = caseB_values(state), caseB_values(state_p)
    val = 0.0
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, N, Np, S, I, K, Kp = v["J"], vp["J"], v["N"], vp["N"], v["S"], v["I"], v["K"], vp["K"]
    for L in range(3):
        for P in range(-L, L + 1):
            val += (
                -phase(P)
                * phase(F - M)
                * wigner_3j(F, L, Fp, -M, P, Mp)
                * phase(J + I + Fp + L)
                * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
                * wigner_6j(J, F, I, Fp, Jp, L)
                * phase(N + S + Jp + L)
                * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
                * wigner_6j(N, J, S, Jp, Np, L)
                * phase(N - K)
                * safe_sqrt((2 * N + 1) * (2 * Np + 1))
                * wigner_3j(N, L, Np, -K, 0, Kp)
                * alpha[L]
                * polarization_tensor(L, -P, eps)
            )
    return val


def CaseB_PolarizabilityParity(state, state_p, alpha, eps):
    v, vp = caseB_values(state), caseB_values(state_p)
    if not same(v, vp, "S", "I", "ell"):
        return 0.0
    val = 0.0
    S, Sp, I = v["S"], vp["S"], v["I"]
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, N, Np, K, Kp = v["J"], vp["J"], v["N"], vp["N"], v["K"], vp["K"]
    for k in range(3):
        for p in range(-k, k + 1):
            sigma_sum = 0.0
            sigma = -S
            while sigma <= S:
                sigma_sum += (
                    wigner_3j(J, N, S, K + sigma, -K, -sigma)
                    * wigner_3j(Jp, Np, Sp, Kp + sigma, -Kp, -sigma)
                    * phase(J - sigma)
                    * wigner_3j(J, k, Jp, -sigma, 0, sigma)
                )
                sigma += 1
            val += -(
                phase(F - M)
                * wigner_3j(F, k, Fp, -M, p, Mp)
                * phase(Fp + J + I + k)
                * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
                * wigner_6j(J, F, I, Fp, Jp, k)
                * phase(N + Np)
                * safe_sqrt((2 * N + 1) * (2 * Np + 1))
                * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
                * sigma_sum
            ) * alpha[k] * polarization_tensor(k, -p, eps)
    return val


def CaseA_Rotation(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    J, S, P, Sigma, K = v["J"], v["S"], v["P"], v["Sigma"], v["K"]
    term1 = (J * (J + 1) + S * (S + 1) - 2 * P * Sigma - K**2) if delta(Sigma, vp["Sigma"]) and delta(P, vp["P"]) else 0.0
    term2 = (
        -2
        * phase(J - P + S - Sigma)
        * safe_sqrt(J * (J + 1) * (2 * J + 1) * S * (S + 1) * (2 * S + 1))
        * sum(wigner_3j(J, 1, J, -P, q, vp["P"]) * wigner_3j(S, 1, S, -Sigma, q, vp["Sigma"]) for q in (-1, 1))
    )
    if not same(v, vp, "ell", "Lambda", "J", "F", "M"):
        return 0.0
    return term1 + term2


def CaseA_RotationSigma(state, state_p):
    return CaseA_Rotation(state, state_p) if delta(abs(caseA_values(state)["Lambda"] + caseA_values(state)["ell"]), 0) else 0.0


def CaseA_RotationDelta(state, state_p):
    return CaseA_Rotation(state, state_p) if delta(abs(caseA_values(state)["Lambda"] + caseA_values(state)["ell"]), 2) else 0.0


def CaseA_SpinOrbit(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "Lambda", "Sigma", "J", "F", "M"):
        return 0.0
    return v["Lambda"] * v["Sigma"]


def CaseA_SpinUncoupling(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "Lambda", "ell", "J", "F", "M"):
        return 0.0
    J, P, S, Sigma = v["J"], v["P"], v["S"], v["Sigma"]
    return (
        -2
        * phase(J - P + S - Sigma)
        * safe_sqrt(J * (J + 1) * (2 * J + 1) * S * (S + 1) * (2 * S + 1))
        * sum(wigner_3j(J, 1, J, -P, q, vp["P"]) * wigner_3j(S, 1, S, -Sigma, q, vp["Sigma"]) for q in (-1, 1))
    )


def CaseA_LambdaDoubling_q(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "Sigma", "J", "F", "M"):
        return 0.0
    J, P = v["J"], v["P"]
    return (
        phase(J - P)
        / (2 * np.sqrt(6))
        * safe_sqrt((2 * J - 1) * (2 * J) * (2 * J + 1) * (2 * J + 2) * (2 * J + 3))
        * sum((1.0 if delta(vp["Lambda"], v["Lambda"] + 2 * q) else 0.0) * wigner_3j(J, 2, vp["J"], -P, -2 * q, vp["P"]) for q in (-1, 1))
    )


def CaseA_LambdaDoubling_p2q(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "J", "F", "M"):
        return 0.0
    J, P, S, Sigma = v["J"], v["P"], v["S"], v["Sigma"]
    return (
        phase(J - P)
        * phase(S - Sigma)
        * safe_sqrt(J * (J + 1) * (2 * J + 1) * S * (S + 1) * (2 * S + 1))
        * sum(
            (1.0 if delta(vp["Lambda"], v["Lambda"] + 2 * q) else 0.0)
            * wigner_3j(J, 1, vp["J"], -P, -q, vp["P"])
            * wigner_3j(S, 1, S, -Sigma, q, vp["Sigma"])
            for q in (-1, 1)
        )
    )


def CaseA_lDoubling(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "Lambda", "J", "F", "M"):
        return 0.0
    J, P, S, Sigma = v["J"], v["P"], v["S"], v["Sigma"]
    term1 = (
        phase(J - P)
        / (2 * np.sqrt(6))
        * safe_sqrt((2 * J - 1) * (2 * J) * (2 * J + 1) * (2 * J + 2) * (2 * J + 3))
        * sum((1.0 if delta(vp["ell"], v["ell"] + 2 * q) else 0.0) * wigner_3j(J, 2, vp["J"], -P, -2 * q, vp["P"]) for q in (-1, 1))
    ) if delta(Sigma, vp["Sigma"]) else 0.0
    term2 = (
        phase(J - P + S - Sigma)
        * safe_sqrt(J * (J + 1) * (2 * J + 1) * S * (S + 1) * (2 * S + 1))
        * sum(
            (1.0 if delta(vp["ell"], v["ell"] + 2 * q) else 0.0)
            * wigner_3j(J, 1, vp["J"], -P, -q, vp["P"])
            * wigner_3j(S, 1, vp["S"], -Sigma, q, vp["Sigma"])
            for q in (-1, 1)
        )
    )
    return term1 - term2


def CaseA_gK_nonadiabatic(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "Lambda", "K", "Sigma", "J", "P", "F", "M"):
        return 0.0
    return v["K"] * v["Lambda"]


def CaseA_Hyperfine_IL(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "Sigma", "Lambda", "K", "P", "F", "M"):
        return 0.0
    I, J, Jp, Fp, P, Lambda_p = v["I"], v["J"], vp["J"], vp["F"], v["P"], vp["Lambda"]
    return (
        Lambda_p
        * phase(I + J + Fp)
        * phase(J - P)
        * safe_sqrt(I * (I + 1) * (2 * I + 1) * (2 * J + 1) * (2 * Jp + 1))
        * wigner_6j(I, J, Fp, Jp, I, 1)
        * wigner_3j(J, 1, Jp, -P, 0, vp["P"])
    )


def CaseA_Hyperfine_IF(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "F", "M"):
        return 0.0
    I, J, Jp, Fp, S, Sigma, P = v["I"], v["J"], vp["J"], vp["F"], v["S"], v["Sigma"], v["P"]
    return (
        phase(I + J + Fp)
        * phase(S - Sigma)
        * phase(J - P)
        * safe_sqrt(I * (I + 1) * (2 * I + 1) * (2 * J + 1) * (2 * Jp + 1) * S * (S + 1) * (2 * S + 1))
        * wigner_6j(I, J, Fp, Jp, I, 1)
        * sum(wigner_3j(J, 1, Jp, -P, q, vp["P"]) * wigner_3j(S, 1, S, -Sigma, q, vp["Sigma"]) for q in (-1, 0, 1))
    )


def CaseA_Hyperfine_Dipolar_c(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    I, J, Jp, F, S, Sigma, P = v["I"], v["J"], vp["J"], v["F"], v["S"], v["Sigma"], v["P"]
    return (
        np.sqrt(30)
        / 3
        * phase(I + Jp + F)
        * phase(J - P)
        * phase(S - Sigma)
        * wigner_6j(I, Jp, F, J, I, 1)
        * safe_sqrt(I * (I + 1) * (2 * I + 1))
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
        * safe_sqrt(S * (S + 1) * (2 * S + 1))
        * sum(
            phase(q)
            * wigner_3j(J, 1, Jp, -P, q, vp["P"])
            * sum(wigner_3j(1, 2, 1, qp, 0, -q) * wigner_3j(S, 1, S, -Sigma, qp, vp["Sigma"]) for qp in (-1, 0, 1))
            for q in (-1, 0, 1)
        )
        * (1.0 if same(v, vp, "F", "M") else 0.0)
    )


def CaseA_Hyperfine_Dipolar_d(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    I, J, Jp, F, S, Sigma, P = v["I"], v["J"], vp["J"], v["F"], v["S"], v["Sigma"], v["P"]
    return (
        np.sqrt(30)
        * 0.5
        * np.sqrt(3 / 2)
        * (2 / 3)
        * phase(I + Jp + F)
        * phase(J - P)
        * phase(S - Sigma)
        * wigner_6j(I, Jp, F, J, I, 1)
        * safe_sqrt(I * (I + 1) * (2 * I + 1))
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
        * safe_sqrt(S * (S + 1) * (2 * S + 1))
        * sum(
            phase(q)
            * wigner_3j(J, 1, Jp, -P, q, P)
            * sum(
                (wigner_3j(1, 2, 1, qp, 2, -q) + wigner_3j(1, 2, 1, qp, -2, -q))
                * wigner_3j(S, 1, S, -Sigma, qp, vp["Sigma"])
                for qp in (-1, 0, 1)
            )
            for q in (-1, 0, 1)
        )
        * (1.0 if same(v, vp, "F", "M") else 0.0)
    )


def CaseA_RennerTeller(state, state_p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "Sigma", "J", "F", "M"):
        return 0.0
    return 0.5 * sum(
        safe_sqrt((v["v2"] + 1) ** 2 - v["K"] ** 2)
        * (1.0 if delta(v["Lambda"], vp["Lambda"] + 2 * q) and delta(v["ell"], vp["ell"] - 2 * q) else 0.0)
        for q in (-1, 1)
    )


def CaseA_TDM(state, state_p, p):
    v, vp = caseA_values(state), caseA_values(state_p)
    F, Fp, M, Mp = v["F"], vp["F"], v["M"], vp["M"]
    J, Jp, I, Pp = v["J"], vp["J"], v["I"], vp["P"]
    if not delta(v["Sigma"], vp["Sigma"]):
        return 0.0
    return (
        phase(Fp - Mp)
        * wigner_3j(Fp, 1, F, -Mp, p, M)
        * phase(F + Jp + I + 1)
        * safe_sqrt((2 * Fp + 1) * (2 * F + 1))
        * phase(Jp - Pp)
        * wigner_6j(J, F, I, Fp, Jp, 1)
        * safe_sqrt((2 * Jp + 1) * (2 * J + 1))
        * sum(wigner_3j(Jp, 1, J, -Pp, q, v["P"]) for q in (-1, 0, 1))
    )


def CaseA_Zeeman_L(state, state_p, p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "Lambda", "K", "I", "S", "Sigma", "P"):
        return 0.0
    F, Fp, M, Mp, J, Jp, I, P, Lambda = v["F"], vp["F"], v["M"], vp["M"], v["J"], vp["J"], v["I"], v["P"], v["Lambda"]
    return (
        phase(p)
        * Lambda
        * phase(F - M + Fp + J + I + 1 + J - P)
        * wigner_6j(J, F, I, Fp, Jp, 1)
        * wigner_3j(F, 1, Fp, -M, p, Mp)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1) * (2 * J + 1) * (2 * Jp + 1))
        * wigner_3j(J, 1, Jp, -P, 0, vp["P"])
    )


def CaseA_Zeeman_S(state, state_p, p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if not same(v, vp, "ell", "I", "S"):
        return 0.0
    F, Fp, M, Mp, J, Jp, I, P, S, Sigma = v["F"], vp["F"], v["M"], vp["M"], v["J"], vp["J"], v["I"], v["P"], v["S"], v["Sigma"]
    return sum(
        phase(p)
        * phase(F - M + J + I + Fp + 1 + J - P + S - Sigma)
        * wigner_6j(J, F, I, Fp, Jp, 1)
        * wigner_3j(F, 1, Fp, -M, p, Mp)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_3j(J, 1, Jp, -P, q, vp["P"])
        * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
        * wigner_3j(S, 1, S, -Sigma, q, vp["Sigma"])
        * safe_sqrt(S * (S + 1) * (2 * S + 1))
        for q in (-1, 0, 1)
    )


def CaseA_Zeeman_glprime(state, state_p, p):
    v, vp = caseA_values(state), caseA_values(state_p)
    if all(delta(v[key], vp[key]) for key in v):
        return 0.0
    F, Fp, M, Mp, J, Jp, I, P, S, Sigma, K = v["F"], vp["F"], v["M"], vp["M"], v["J"], vp["J"], v["I"], v["P"], v["S"], v["Sigma"], v["K"]
    return (
        phase(p)
        * phase(F - M)
        * wigner_3j(F, 1, Fp, -M, p, Mp)
        * phase(Fp + J + I + 1)
        * safe_sqrt((2 * F + 1) * (2 * Fp + 1))
        * wigner_6j(Jp, Fp, I, F, J, 1)
        * sum(
            (1.0 if delta(vp["K"], K - 2 * q) else 0.0)
            * phase(J - P + S - Sigma)
            * phase(J - P)
            * wigner_3j(J, 1, Jp, -P, q, vp["P"])
            * safe_sqrt((2 * J + 1) * (2 * Jp + 1))
            * phase(S - Sigma)
            * wigner_3j(S, 1, S, -Sigma, -q, vp["Sigma"])
            * safe_sqrt(S * (S + 1) * (2 * S + 1))
            for q in (-1, 1)
        )
    )


def make_operator_class(name, matrix_element, *default_args):
    return type(name, (MatrixElementOperator,), {"__init__": lambda self, basis, *args: MatrixElementOperator.__init__(self, basis, matrix_element, *(args or default_args))})


CaseB_RotationHamiltonian = make_operator_class("CaseB_RotationHamiltonian", CaseB_Rotation)
CaseB_RotationDistortionHamiltonian = make_operator_class("CaseB_RotationDistortionHamiltonian", CaseB_RotationDistortion)
CaseB_SpinRotationLambda0Hamiltonian = make_operator_class("CaseB_SpinRotationLambda0Hamiltonian", CaseB_SpinRotation_Lambda0)
CaseB_SpinRotationHamiltonian = make_operator_class("CaseB_SpinRotationHamiltonian", CaseB_SpinRotation)
CaseB_HyperfineISHamiltonian = make_operator_class("CaseB_HyperfineISHamiltonian", CaseB_Hyperfine_IS)
CaseB_HyperfineDipolarHamiltonian = make_operator_class("CaseB_HyperfineDipolarHamiltonian", CaseB_Hyperfine_Dipolar)
CaseB_NuclearQuadrupoleHamiltonian = make_operator_class("CaseB_NuclearQuadrupoleHamiltonian", CaseB_NuclearQuadrupole)
CaseB_MagneticQuadrupoleHamiltonian = make_operator_class("CaseB_MagneticQuadrupoleHamiltonian", CaseB_MagneticQuadrupole)
CaseB_lDoublingHamiltonian = make_operator_class("CaseB_lDoublingHamiltonian", CaseB_lDoubling)
CaseB_StarkOperator = make_operator_class("CaseB_StarkOperator", CaseB_Stark, 0)
CaseB_ZeemanOperator = make_operator_class("CaseB_ZeemanOperator", CaseB_Zeeman, 0)
CaseB_ZeemanNuclearOperator = make_operator_class("CaseB_ZeemanNuclearOperator", CaseB_ZeemanNuclear, 0)
CaseB_ZeemanRotationOperator = make_operator_class("CaseB_ZeemanRotationOperator", CaseB_ZeemanRotation, 0)
CaseB_TDMOperator = make_operator_class("CaseB_TDMOperator", CaseB_TDM, 0)
CaseB_TDMVibrationalOperator = make_operator_class("CaseB_TDMVibrationalOperator", CaseB_TDMVibrational, 0)
CaseB_TDMMagneticOperator = make_operator_class("CaseB_TDMMagneticOperator", CaseB_TDMMagnetic, 0)
CaseB_PolarizabilityOperator = make_operator_class("CaseB_PolarizabilityOperator", CaseB_Polarizability, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
CaseB_PolarizabilityParityOperator = make_operator_class("CaseB_PolarizabilityParityOperator", CaseB_PolarizabilityParity, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))

CaseA_RotationHamiltonian = make_operator_class("CaseA_RotationHamiltonian", CaseA_Rotation)
CaseA_RotationSigmaHamiltonian = make_operator_class("CaseA_RotationSigmaHamiltonian", CaseA_RotationSigma)
CaseA_RotationDeltaHamiltonian = make_operator_class("CaseA_RotationDeltaHamiltonian", CaseA_RotationDelta)
CaseA_SpinOrbitHamiltonian = make_operator_class("CaseA_SpinOrbitHamiltonian", CaseA_SpinOrbit)
CaseA_SpinUncouplingHamiltonian = make_operator_class("CaseA_SpinUncouplingHamiltonian", CaseA_SpinUncoupling)
CaseA_LambdaDoublingQHamiltonian = make_operator_class("CaseA_LambdaDoublingQHamiltonian", CaseA_LambdaDoubling_q)
CaseA_LambdaDoublingP2QHamiltonian = make_operator_class("CaseA_LambdaDoublingP2QHamiltonian", CaseA_LambdaDoubling_p2q)
CaseA_lDoublingHamiltonian = make_operator_class("CaseA_lDoublingHamiltonian", CaseA_lDoubling)
CaseA_gKNonadiabaticHamiltonian = make_operator_class("CaseA_gKNonadiabaticHamiltonian", CaseA_gK_nonadiabatic)
CaseA_HyperfineILHamiltonian = make_operator_class("CaseA_HyperfineILHamiltonian", CaseA_Hyperfine_IL)
CaseA_HyperfineIFHamiltonian = make_operator_class("CaseA_HyperfineIFHamiltonian", CaseA_Hyperfine_IF)
CaseA_HyperfineDipolarCHamiltonian = make_operator_class("CaseA_HyperfineDipolarCHamiltonian", CaseA_Hyperfine_Dipolar_c)
CaseA_HyperfineDipolarDHamiltonian = make_operator_class("CaseA_HyperfineDipolarDHamiltonian", CaseA_Hyperfine_Dipolar_d)
CaseA_RennerTellerHamiltonian = make_operator_class("CaseA_RennerTellerHamiltonian", CaseA_RennerTeller)
CaseA_TDMOperator = make_operator_class("CaseA_TDMOperator", CaseA_TDM, 0)
CaseA_ZeemanLOperator = make_operator_class("CaseA_ZeemanLOperator", CaseA_Zeeman_L, 0)
CaseA_ZeemanSOperator = make_operator_class("CaseA_ZeemanSOperator", CaseA_Zeeman_S, 0)
CaseA_ZeemanGLPrimeOperator = make_operator_class("CaseA_ZeemanGLPrimeOperator", CaseA_Zeeman_glprime, 0)


__all__ = [
    name
    for name in globals()
    if name.startswith(("CaseA_", "CaseB_")) or name in {"MatrixElementOperator", "polarization_tensor"}
]
