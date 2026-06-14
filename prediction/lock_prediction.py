import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv(
    "advanced_rb_clock.csv"
)

X = df[
[
"vcsel_temp",
"vcsel_current",
"optical_power",
"cell_temp",
"magnetic_field",
"contrast"
]
]

y = df["lock_status"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        pred
    )
)

joblib.dump(
    model,
    "prediction/lock_predictor.pkl"
)

print("Lock Predictor Saved")