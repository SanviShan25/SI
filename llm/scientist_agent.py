def recommendation(reasons):

    advice = []

    if "VCSEL temperature drift" in reasons:

        advice.append(
            "Reduce VCSEL temperature by 0.2°C"
        )

    if "Optical power drop" in reasons:

        advice.append(
            "Inspect laser power control loop"
        )

    if "Resonance contrast degradation" in reasons:

        advice.append(
            "Retune CPT resonance"
        )

    if "Cell temperature instability" in reasons:

        advice.append(
            "Check thermal controller"
        )

    return advice