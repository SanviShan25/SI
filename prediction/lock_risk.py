def risk_score(prob):

    if prob > 0.8:
        return "HIGH"

    elif prob > 0.5:
        return "MEDIUM"

    return "LOW"