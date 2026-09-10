from functools import lru_cache

from sympy import Rational, S
import sympy.physics.wigner


def _to_twice_integer(value):
    """Represent integer/half-integer angular momenta as 2*j integer cache keys."""
    twice_value = 2 * float(value)
    rounded = round(twice_value)
    if abs(twice_value - rounded) > 1e-9:
        raise ValueError(f"Wigner symbol arguments must be integer or half-integer, got {value!r}")
    return int(rounded)


def _from_twice_integer(value):
    return Rational(value, 2)


def _invalid_3j(j1, j2, j3, m1, m2, m3):
    if m1 + m2 + m3 != 0:
        return True
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return True
    if j1 + j2 < j3 or j1 + j3 < j2 or j2 + j3 < j1:
        return True
    return (j1 - m1) % 2 or (j2 - m2) % 2 or (j3 - m3) % 2


def _invalid_6j(j1, j2, j3, j4, j5, j6):
    triples = ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    for a, b, c in triples:
        if a + b < c or a + c < b or b + c < a:
            return True
        if (a + b + c) % 2:
            return True
    return False


@lru_cache(maxsize=None)
def _wigner_3j_exact(j1, j2, j3, m1, m2, m3):
    if _invalid_3j(j1, j2, j3, m1, m2, m3):
        return S.Zero
    return sympy.physics.wigner.wigner_3j(
        _from_twice_integer(j1),
        _from_twice_integer(j2),
        _from_twice_integer(j3),
        _from_twice_integer(m1),
        _from_twice_integer(m2),
        _from_twice_integer(m3),
    )


@lru_cache(maxsize=None)
def _wigner_6j_exact(j1, j2, j3, j4, j5, j6):
    if _invalid_6j(j1, j2, j3, j4, j5, j6):
        return S.Zero
    return sympy.physics.wigner.wigner_6j(
        _from_twice_integer(j1),
        _from_twice_integer(j2),
        _from_twice_integer(j3),
        _from_twice_integer(j4),
        _from_twice_integer(j5),
        _from_twice_integer(j6),
    )


@lru_cache(maxsize=None)
def _wigner_9j_exact(j1, j2, j3, j4, j5, j6, j7, j8, j9):
    return sympy.physics.wigner.wigner_9j(
        _from_twice_integer(j1),
        _from_twice_integer(j2),
        _from_twice_integer(j3),
        _from_twice_integer(j4),
        _from_twice_integer(j5),
        _from_twice_integer(j6),
        _from_twice_integer(j7),
        _from_twice_integer(j8),
        _from_twice_integer(j9),
    )


@lru_cache(maxsize=None)
def _wigner_3j_float(j1, j2, j3, m1, m2, m3, precision):
    return float(_wigner_3j_exact(j1, j2, j3, m1, m2, m3).evalf(precision))


@lru_cache(maxsize=None)
def _wigner_6j_float(j1, j2, j3, j4, j5, j6, precision):
    return float(_wigner_6j_exact(j1, j2, j3, j4, j5, j6).evalf(precision))


@lru_cache(maxsize=None)
def _wigner_9j_float(j1, j2, j3, j4, j5, j6, j7, j8, j9, precision):
    return float(_wigner_9j_exact(j1, j2, j3, j4, j5, j6, j7, j8, j9).evalf(precision))


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
    key = tuple(_to_twice_integer(x) for x in (j1, j2, j3, m1, m2, m3))

    if numeric:
        return _wigner_3j_float(*key, precision)
    return _wigner_3j_exact(*key)

def wigner_6j(j1, j2, j3, j4, j5, j6, numeric=True, precision=15):
    """
    Evaluate the Wigner 6j symbol {j1 j2 j3; j4 j5 j6}.

    Parameters:
        j1 to j6 : int or float — angular momentum values (can be half-integers)
        numeric : bool — if True, return floating point result
        precision : int — number of decimal digits for numeric result

    Returns:
        SymPy Rational or Float
    """
    key = tuple(_to_twice_integer(x) for x in (j1, j2, j3, j4, j5, j6))

    if numeric:
        return _wigner_6j_float(*key, precision)
    return _wigner_6j_exact(*key)


def wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9, numeric=True, precision=15):
    """
    Evaluate the Wigner 9j symbol {{j1 j2 j3}, {j4 j5 j6}, {j7 j8 j9}}.
    """
    key = tuple(_to_twice_integer(x) for x in (j1, j2, j3, j4, j5, j6, j7, j8, j9))

    if numeric:
        return _wigner_9j_float(*key, precision)
    return _wigner_9j_exact(*key)


def clear_wigner_cache():
    _wigner_3j_exact.cache_clear()
    _wigner_6j_exact.cache_clear()
    _wigner_9j_exact.cache_clear()
    _wigner_3j_float.cache_clear()
    _wigner_6j_float.cache_clear()
    _wigner_9j_float.cache_clear()


def wigner_cache_info():
    return {
        "wigner_3j_exact": _wigner_3j_exact.cache_info(),
        "wigner_6j_exact": _wigner_6j_exact.cache_info(),
        "wigner_9j_exact": _wigner_9j_exact.cache_info(),
        "wigner_3j_float": _wigner_3j_float.cache_info(),
        "wigner_6j_float": _wigner_6j_float.cache_info(),
        "wigner_9j_float": _wigner_9j_float.cache_info(),
    }
