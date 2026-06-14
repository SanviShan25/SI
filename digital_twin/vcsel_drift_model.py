import numpy as np

def wavelength_shift(temp):

    base = 794.98

    shift = 0.06 * (
        temp - 45
    )

    return base + shift