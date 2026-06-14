# prediction/train_future_model.py

import pandas as pd
import joblib
from xgboost import XGBRegressor

df = pd.read_csv("rb_clock_data.csv")

features = [
    "vcsel_temp",
    "vcsel_current",
    "optical_power",
    "contrast"
]

X = df[features]

# predict 10 samples ahead
y = df["frequency_offset"].shift(-10)

data = pd.concat([X, y], axis=1).dropna()

X = data[features]
y = data["frequency_offset"]

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model.fit(X, y)

joblib.dump(
    model,
    "prediction/rb_future_model.pkl"
)

print("Model trained")