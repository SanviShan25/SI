import pandas as pd
import matplotlib.pyplot as plt

from allan_curve import allan_curve

rb = pd.read_csv("rb_clock_data.csv")
cs = pd.read_csv("cs_clock_data.csv")

rb_tau, rb_dev = allan_curve(
    rb["frequency_offset"]
)

cs_tau, cs_dev = allan_curve(
    cs["frequency_offset"]
)

plt.figure(figsize=(8,5))

plt.loglog(
    rb_tau,
    rb_dev,
    marker="o",
    label="Rubidium"
)

plt.loglog(
    cs_tau,
    cs_dev,
    marker="s",
    label="Cesium"
)

plt.xlabel("Tau (s)")
plt.ylabel("Allan Deviation")
plt.title("Rb vs Cs Stability Comparison")
plt.legend()
plt.grid(True)

plt.show()