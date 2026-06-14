import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("rb_clock_data.csv")

df["future_offset"] = (
    df["frequency_offset"]
    .shift(-10)
)

df = df.dropna()

X = df[
    [
        "vcsel_temp",
        "vcsel_current",
        "optical_power",
        "contrast"
    ]
]

y = df["future_offset"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "prediction/rb_future_model.pkl"
)

print("Model Saved Successfully")