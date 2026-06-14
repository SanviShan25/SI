import pandas as pd

rb = pd.read_csv("rb_clock_data.csv")
cs = pd.read_csv("cs_clock_data.csv")

print("\n==========================")
print(" RUBIDIUM CLOCK ")
print("==========================")

print(rb["frequency_offset"].describe())

print("\n==========================")
print(" CESIUM CLOCK ")
print("==========================")

print(cs["frequency_offset"].describe())