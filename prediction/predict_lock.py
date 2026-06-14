import joblib
import pandas as pd

model = joblib.load(
    "prediction/lock_predictor.pkl"
)

def predict_lock(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return prediction[0]