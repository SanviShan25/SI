# prediction/predict.py

import joblib
import pandas as pd

model = joblib.load(
    "prediction/rb_future_model.pkl"
)

def predict_future_drift(data):

    X = pd.DataFrame([data])

    return float(
        model.predict(X)[0]
    )