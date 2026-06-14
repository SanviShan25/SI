# visualization/plot_data.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("rb_clock_data.csv")

plt.figure(figsize=(10,5))

plt.plot(df["frequency_offset"])

plt.title("Frequency Offset")

plt.xlabel("Time")

plt.ylabel("Offset")

plt.show()