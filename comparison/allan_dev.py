import pandas as pd
import numpy as np

rb = pd.read_csv("rb_clock_data.csv")
cs = pd.read_csv("cs_clock_data.csv")

def allan_deviation(freq):

    return np.sqrt(
        0.5*np.mean(
            np.diff(freq)**2
        )
    )

rb_allan = allan_deviation(
    rb["frequency_offset"].values
)

cs_allan = allan_deviation(
    cs["frequency_offset"].values
)

print("\n====================")
print("ALLAN DEVIATION")
print("====================")

print("Rb Allan Dev :", rb_allan)
print("Cs Allan Dev :", cs_allan)