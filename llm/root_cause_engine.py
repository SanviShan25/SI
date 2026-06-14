def diagnose(row):

    reasons = []

    if row["vcsel_temp"] > 45.3:
        reasons.append(
            "VCSEL temperature drift"
        )

    if row["contrast"] < 0.90:
        reasons.append(
            "Resonance contrast degradation"
        )

    if row["optical_power"] < 98:
        reasons.append(
            "Optical power drop"
        )

    if row["cell_temp"] > 80.4:
        reasons.append(
            "Cell temperature instability"
        )

    return reasons