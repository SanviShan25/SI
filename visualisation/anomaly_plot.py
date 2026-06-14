import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("rb_clock_data.csv")

normal = df[df["anomaly"]==0]
abnormal = df[df["anomaly"]==1]

plt.figure(figsize=(12,6))

plt.scatter(
    normal.index,
    normal["frequency_offset"],
    s=5,
    label="Normal"
)

plt.scatter(
    abnormal.index,
    abnormal["frequency_offset"],
    color="red",
    s=15,
    label="Anomaly"
)

plt.legend()

plt.show()