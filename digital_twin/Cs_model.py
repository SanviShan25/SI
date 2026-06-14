import numpy as np
import pandas as pd

np.random.seed(42)

n = 10000

time = np.arange(n)

vcsel_temp = 40 + 0.0015*time + np.random.normal(0,0.15,n)

vcsel_current = 5 + np.random.normal(0,0.04,n)

optical_power = 100 + np.random.normal(0,1,n)

cell_temp = 75 + np.random.normal(0,0.1,n)

contrast = (
    0.82
    - 0.0008*(vcsel_temp-40)
    + np.random.normal(0,0.008,n)
)

frequency_offset = (
    0.7e-11*(vcsel_temp-40)
    + 1.5e-11*(vcsel_current-5)
    - 3e-12*(contrast-0.8)
)

anomaly = np.zeros(n)

for idx in np.random.choice(n,80):
    frequency_offset[idx] *= 8
    vcsel_temp[idx] += 4
    anomaly[idx] = 1

df = pd.DataFrame({
    "time": time,
    "vcsel_temp": vcsel_temp,
    "vcsel_current": vcsel_current,
    "optical_power": optical_power,
    "cell_temp": cell_temp,
    "contrast": contrast,
    "frequency_offset": frequency_offset,
    "anomaly": anomaly
})

df.to_csv("cs_clock_data.csv", index=False)

print("Cesium Dataset Created Successfully")