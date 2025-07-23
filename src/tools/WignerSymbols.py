from sympy import S
import sympy.physics.wigner

def wigner_3j(j1, j2, j3, m1, m2, m3, numeric=True, precision=15):
    """
    Evaluate the Wigner 3j symbol ⟨j1 j2 j3 | m1 m2 m3⟩.

    Parameters:
        j1, j2, j3 : int or float — total angular momenta (can be half-integers)
        m1, m2, m3 : int or float — magnetic quantum numbers
        numeric : bool — if True, return floating point result
        precision : int — decimal digits of precision for numeric result

    Returns:
        SymPy Rational or Float
    """
    # Convert all inputs to SymPy Rationals for exact arithmetic
    j1, j2, j3 = S(j1), S(j2), S(j3)
    m1, m2, m3 = S(m1), S(m2), S(m3)

    result = sympy.physics.wigner.wigner_3j(j1, j2, j3, m1, m2, m3)

    if numeric:
        return result.evalf(precision)
    return result

def wigner_6j(j1, j2, j3, j4, j5, j6, numeric=False, precision=15):
    """
    Evaluate the Wigner 6j symbol {j1 j2 j3; j4 j5 j6}.

    Parameters:
        j1 to j6 : int or float — angular momentum values (can be half-integers)
        numeric : bool — if True, return floating point result
        precision : int — number of decimal digits for numeric result

    Returns:
        SymPy Rational or Float
    """
    # Convert inputs to SymPy Rational for exact computation
    j1, j2, j3 = S(j1), S(j2), S(j3)
    j4, j5, j6 = S(j4), S(j5), S(j6)

    result = sympy.physics.wigner.wigner_6j(j1, j2, j3, j4, j5, j6)

    if numeric:
        return result.evalf(precision)
    return result
