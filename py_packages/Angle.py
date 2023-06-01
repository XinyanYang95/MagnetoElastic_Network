"""Angle package for angle manipulation
NOTE: shorter than the one for dihedral angles
"""
import numpy as np

WRAP_LO = 0
WRAP_HI = 2*np.pi


def xy2ang(rx, ry):
    """circular statistic helper function
    for converting unit circle position to angle"""
    am = np.arctan(ry / rx) + np.vectorize(ang_shift)(rx)
    return am


def ang_shift(rx):
    """circular statistic helper function
    convert unit circle position to angle shift to correct arctan
    depending on which half of the circle point is from"""
    if rx >= 0:
        shift = 0.0  # necessary!
    else:
        shift = np.pi
    return shift


def wrap(a, low=WRAP_LO, high=WRAP_HI):
    """wrap angles between low and high
    np.vectorize(wrap)(angle, lo, hi) for matrices"""
    while a < low or a >= high:
        if a < low:
            a += 2*np.pi
        if a >= high:
            a -= 2*np.pi
    return a
