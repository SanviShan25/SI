import numpy as np
import pandas as pd

np.random.seed(42)

n = 10000

time = np.arange(n)

# VCSEL temperature
vcsel_temp = 40 + 0.002*time + np.random.normal(0,0.2,n)

# VCSEL current
vcsel_current = 5 + np.random.normal(0,0.05,n)

# Optical power
optical_power = 100 + np.random.normal(0,1,n)

# Cell temperature
cell_temp = 75 + np.random.normal(0,0.15,n)

# Contrast
contrast = (
    0.8
    - 0.001*(vcsel_temp-40)
    + np.random.normal(0,0.01,n)
)

# Frequency offset
frequency_offset = (
    1e-11*(vcsel_temp-40)
    + 2e-11*(vcsel_current-5)
    - 5e-12*(contrast-0.8)
)

# Inject anomalies
anomaly = np.zeros(n)

for idx in np.random.choice(n,100):

    frequency_offset[idx] *= 10
    vcsel_temp[idx] += 5
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

df.to_csv("rb_clock_data.csv",index=False)

print("Dataset Created")