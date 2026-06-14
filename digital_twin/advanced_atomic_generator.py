import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

vcsel_temp = 45 + np.random.normal(0, 0.15, n)

vcsel_current = 3.0 + np.random.normal(0, 0.03, n)

optical_power = 100 + np.random.normal(0, 2, n)

cell_temp = 80 + np.random.normal(0, 0.2, n)

magnetic_field = 1.5 + np.random.normal(0, 0.02, n)

contrast = (
    0.95
    - 0.003 * np.abs(vcsel_temp - 45)
    + np.random.normal(0, 0.01, n)
)

frequency_offset = (
    (vcsel_temp - 45) * 2e-11
    + (vcsel_current - 3) * 1e-11
    + (100 - optical_power) * 5e-12
    + np.random.normal(0, 5e-12, n)
)

lock_status = []

for c, f in zip(contrast, frequency_offset):

    if c > 0.90 and abs(f) < 3e-11:
        lock_status.append("LOCKED")

    elif c > 0.85:
        lock_status.append("MARGINAL")

    else:
        lock_status.append("UNLOCKED")

df = pd.DataFrame({
    "vcsel_temp": vcsel_temp,
    "vcsel_current": vcsel_current,
    "optical_power": optical_power,
    "cell_temp": cell_temp,
    "magnetic_field": magnetic_field,
    "contrast": contrast,
    "frequency_offset": frequency_offset,
    "lock_status": lock_status
})

df.to_csv(
    "advanced_rb_clock.csv",
    index=False
)

print("Dataset Generated Successfully")
print(df.head())