def stabilization_action(data):

    actions = []

    if data["vcsel_temp"] > 45.2:
        actions.append(
            "Decrease VCSEL temperature by 0.2 C"
        )

    if data["vcsel_temp"] < 44.8:
        actions.append(
            "Increase VCSEL temperature by 0.2 C"
        )

    if data["contrast"] < 0.90:
        actions.append(
            "Retune CPT resonance"
        )

    if data["optical_power"] < 98:
        actions.append(
            "Increase laser optical power"
        )

    if data["cell_temp"] > 80.3:
        actions.append(
            "Reduce cell temperature"
        )

    if len(actions) == 0:

        actions.append(
            "System Stable"
        )

    return actions