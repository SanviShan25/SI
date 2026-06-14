import joblib
import pandas as pd

model = joblib.load(
    "prediction/lock_predictor.pkl"
)

def lock_probability(data):

    df = pd.DataFrame([data])

    probs = model.predict_proba(df)

    return probs[0]