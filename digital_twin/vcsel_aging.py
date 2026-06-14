import numpy as np
import pandas as pd

days = np.arange(0,365)

optical_power = (
    100
    - 0.01*days
    + np.random.normal(0,0.2,len(days))
)

df = pd.DataFrame({
    "day":days,
    "optical_power":optical_power
})

df.to_csv(
    "vcsel_aging.csv",
    index=False
)

print(df.head())    