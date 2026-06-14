import pandas as pd
import matplotlib.pyplot as plt

rb = pd.read_csv("rb_clock_data.csv")
cs = pd.read_csv("cs_clock_data.csv")

plt.figure(figsize=(12,6))

plt.plot(
    rb["frequency_offset"][:1000],
    label="Rubidium"
)

plt.plot(
    cs["frequency_offset"][:1000],
    label="Cesium"
)

plt.xlabel("Sample")

plt.ylabel("Frequency Offset")

plt.title(
    "Rb vs Cs Frequency Stability"
)

plt.legend()

plt.grid(True)

plt.show()