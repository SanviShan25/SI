def analyze_clock(
    vcsel_temp,
    contrast,
    predicted_state
):

    if predicted_state == "UNLOCKED":

        return f"""
Likely lock loss predicted.

Possible causes:

- VCSEL temperature drift
- Resonance contrast degradation

Recommended action:

- Re-tune VCSEL temperature
- Verify optical power loop
"""

    if predicted_state == "MARGINAL":

        return f"""
Clock entering marginal regime.

Monitor contrast and frequency drift.
"""

    return """
Clock remains locked.
"""