import pandas as pd

rb = pd.read_csv("rb_clock_data.csv")
cs = pd.read_csv("cs_clock_data.csv")

def stability_score(df):

    drift = abs(df["frequency_offset"].mean())

    noise = df["frequency_offset"].std()

    score = 100 - ((drift + noise)*1e12)

    return round(score,2)

print("\n====================")
print("STABILITY SCORES")
print("====================")

print("Rb Score :", stability_score(rb))
print("Cs Score :", stability_score(cs))