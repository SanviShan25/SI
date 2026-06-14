"""
Frequency Stability Improvement Metrics
=======================================
Auxiliary metrology computations for stability improvement quantification,
following DRDO frequency metrology assessment methodology.

References
----------
[1] IEEE Std 1139-2022 — Standard Definitions of Physical Quantities for
    Fundamental Frequency and Time Metrology — Random Instabilities.
[2] Riley, W. J. & Howe, D. A. (2008). Handbook of Frequency Stability
    Analysis. NIST Technical Note 1337.
[3] Vanier, J. & Audoin, C. (1989). The Quantum Physics of Atomic Frequency
    Standards. Adam Hilger, Bristol.
"""
import numpy as np


def compute_allan_improvement(
    baseline_allan: float,
    optimized_allan: float,
) -> float:
    """
    Compute fractional improvement in Allan deviation σy following a
    stabilisation intervention.

    Δσy / σy_baseline × 100%

    A positive value indicates improved (lower) σy after stabilisation.
    A zero value indicates no measurable improvement or degradation.

    Parameters
    ----------
    baseline_allan : float
        σy(τ) before stabilisation.
    optimized_allan : float
        σy(τ) after stabilisation.

    Returns
    -------
    float : Percentage improvement in σy. Range [0, 100].
    """
    baseline  = abs(float(baseline_allan))
    optimized = abs(float(optimized_allan))
    if baseline <= 0 or optimized >= baseline:
        return 0.0
    return round(float(max(0.0, (baseline - optimized) / baseline * 100.0)), 1)


def compute_stability_improvement_from_budget(
    total_sigma_y: float,
    dominant_contribution_pct: float,
) -> dict:
    """
    Estimate expected σy improvement after fully suppressing the dominant
    environmental sensitivity channel.

    Method: RSS budget with dominant channel removed:
        σy_improved = √(σy_total² × (1 − dominant_pct / 100))

    This is a first-order estimate; it assumes the dominant channel is
    uncorrelated with residual channels and that the sensitivity coefficient
    and σx of the remaining channels are unchanged.

    Parameters
    ----------
    total_sigma_y : float
        RSS total stability budget σy (fractional frequency units).
    dominant_contribution_pct : float
        Fraction of total variance attributed to the dominant channel (0–100).

    Returns
    -------
    dict with keys:
        sigma_y_before, sigma_y_after, improvement_pct, residual_floor
    """
    sigma_before = abs(float(total_sigma_y))
    frac = max(0.0, min(100.0, float(dominant_contribution_pct))) / 100.0
    sigma_after = float(sigma_before * np.sqrt(max(0.0, 1.0 - frac)))
    improvement = (
        round((sigma_before - sigma_after) / sigma_before * 100.0, 1)
        if sigma_before > 0 else 0.0
    )
    return {
        "sigma_y_before":   round(sigma_before, 12),
        "sigma_y_after":    round(sigma_after,  12),
        "improvement_pct":  improvement,
        "residual_floor":   round(sigma_after,  12),
    }


def compute_excursion_rate_per_100(
    excursion_flags: np.ndarray,
    window: int = 100,
) -> float:
    """
    Compute the frequency excursion rate per `window` samples.

    Parameters
    ----------
    excursion_flags : array-like
        Binary array: 1 = excursion detected, 0 = nominal.
    window : int
        Normalisation window (default 100 samples).

    Returns
    -------
    float : Excursion rate per `window` samples.
    """
    flags = np.asarray(excursion_flags, dtype=int)
    n = len(flags)
    if n == 0:
        return 0.0
    rate = float(np.sum(flags)) / float(n) * float(window)
    return round(rate, 2)


def compute_post_stabilisation_budget(
    contributions: list[dict],
    suppressed_parameter: str,
    suppression_fraction: float = 1.0,
) -> dict:
    """
    Compute the expected RSS stability budget after suppressing one parameter's
    environmental fluctuation by a given fraction.

    Parameters
    ----------
    contributions : list of dict
        Stability budget contributions from compute_stability_budget().
        Each entry must have 'Parameter' and 'Contribution (%)'.
    suppressed_parameter : str
        Name of the parameter to be suppressed.
    suppression_fraction : float
        Fraction of variance reduction achievable (0 = none, 1 = complete).

    Returns
    -------
    dict with keys:
        new_contributions, sigma_y_reduction_pct
    """
    new_contributions = []
    for c in contributions:
        entry = dict(c)
        if c.get("Parameter") == suppressed_parameter:
            entry["Contribution (%)"] = round(
                float(c.get("Contribution (%)", 0)) * (1.0 - suppression_fraction), 2
            )
        new_contributions.append(entry)

    old_total = sum(c.get("Contribution (%)", 0) for c in contributions) or 1.0
    new_total = sum(c.get("Contribution (%)", 0) for c in new_contributions)
    sigma_y_reduction = round(
        (1.0 - np.sqrt(max(0.0, new_total) / old_total)) * 100.0, 1
    )
    return {
        "new_contributions": new_contributions,
        "sigma_y_reduction_pct": sigma_y_reduction,
    }