import pandas as pd
import matplotlib.pyplot as plt

from allan_curve import allan_curve

df = pd.read_csv(
    "../advanced_rb_clock.csv"
)

taus, devs = allan_curve(
    df["frequency_offset"]
)

plt.loglog(
    taus,
    devs,
    marker="o"
)

plt.xlabel(
    "Tau (s)"
)

plt.ylabel(
    "Allan Deviation"
)

plt.title(
    "Atomic Clock Stability"
)

plt.grid(True)

plt.show()