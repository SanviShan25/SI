import pandas as pd

from sklearn.ensemble import IsolationForest

df = pd.read_csv("rb_clock_data.csv")

X = df[
[
"vcsel_temp",
"vcsel_current",
"contrast",
"frequency_offset"
]
]

model = IsolationForest(
    contamination=0.01,
    random_state=42
)

preds = model.fit_predict(X)

df["predicted"] = preds

print(df["predicted"].value_counts())