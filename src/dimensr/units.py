"""Exact rational SI exponents, in the order L, M, T, I, Theta, N, J."""
from fractions import Fraction
import math

ZERO = (Fraction(0),) * 7
BASE_UNITS = ("L", "M", "T", "I", "Theta", "N", "J")


def unit(values):
    if len(values) != 7:
        raise ValueError("Units must contain 7 exponents: L, M, T, I, Theta, N, J.")
    result = []
    for value in values:
        exponent = Fraction(str(value))
        if not math.isfinite(float(exponent)):
            raise ValueError("Unit exponents must be finite.")
        result.append(exponent)
    return tuple(result)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, factor):
    return tuple(x * factor for x in a)


def encode(a):
    return [str(x) for x in a]
