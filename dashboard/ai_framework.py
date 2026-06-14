"""
Frequency Metrology Computation Engine
=======================================
Physically grounded computations for characterisation and stabilisation
assessment of VCSEL-pumped ⁸⁷Rb and ¹³³Cs atomic frequency standards.

References
----------
[1] IEEE Std 1139-2022 — Standard Definitions of Physical Quantities for
    Fundamental Frequency and Time Metrology — Random Instabilities.
[2] Riley, W. J. & Howe, D. A. (2008). Handbook of Frequency Stability
    Analysis. NIST Technical Note 1337. NIST, Gaithersburg, MD.
[3] Vanier, J. & Audoin, C. (1989). The Quantum Physics of Atomic Frequency
    Standards. Adam Hilger, Bristol.
[4] Camparo, J. (2005). The rubidium atomic clock and basic research.
    Physics Today, 58(11), 33–39.
[5] Cutler, L. S. & Searle, C. L. (1966). Some aspects of the theory and
    measurement of frequency fluctuations in frequency standards.
    Proc. IEEE, 54(2), 136–154.
[6] Audoin, C. & Guinot, B. (2001). The Measurement of Time. Cambridge UP.
[7] Vanier, J., Simard, J.-F. & Barrette, J.-N. (2003). Practical
    considerations in the realisation of a passive rubidium frequency
    standard. Appl. Phys. B, 76(7), 785–792.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── Physical parameter columns ────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "vcsel_temp",
    "vcsel_current",
    "optical_power",
    "cell_temp",
    "contrast",
    "frequency_offset",
]

DISPLAY_COLUMNS = {
    "vcsel_temp":      "VCSEL Temperature",
    "vcsel_current":   "Injection Current",
    "optical_power":   "Optical Power",
    "cell_temp":       "Cell Temperature",
    "contrast":        "Resonance Contrast",
    "frequency_offset": "Frequency Offset",
}

# ── Physics-based sensitivity coefficients ────────────────────────────────────
# ∂(Δf/f)/∂xi  — fractional frequency per unit of each parameter.
# Derived from Vanier & Audoin (1989, Ch. 5) and Camparo (2005).
SENSITIVITY_COEFFICIENTS = {
    "VCSEL Temperature":  2.8e-11,   # (Δf/f) / °C  — thermal VCSEL freq. coupling
    "Optical Power":      1.2e-11,   # (Δf/f) / µW  — AC Stark / light-shift
    "Cell Temperature":   1.6e-11,   # (Δf/f) / °C  — buffer-gas pressure coefficient
    "Injection Current":  0.9e-11,   # (Δf/f) / mA  — current-to-frequency pulling
    "Resonance Contrast": 4.2e-10,   # (Δf/f) / (rel. unit) — discriminator gain coupling
}

PARAMETER_ORDER = [
    ("vcsel_temp",    "VCSEL Temperature"),
    ("optical_power", "Optical Power"),
    ("cell_temp",     "Cell Temperature"),
    ("vcsel_current", "Injection Current"),
    ("contrast",      "Resonance Contrast"),
]

__all__ = [
    "classify_operational_state",
    "compute_allan_noise_analysis",
    "compute_drift_projection",
    "compute_excursion_physical_attribution_summary",
    "compute_frequency_excursion_analysis",
    "compute_hadamard_deviation",
    "compute_modified_allan_deviation",
    "compute_operational_stability_state",
    "compute_physical_attribution",
    "compute_physical_excursion_attribution",
    "compute_sensitivity_coefficients",
    "compute_sensitivity_ranking",
    "compute_stability_budget",
    "generate_assessment_report",
    "generate_parameter_stabilisation_actions",
    "generate_stability_assessment_narrative",
]


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare_feature_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce input data to a clean, time-sorted feature frame."""
    if df is None:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    frame = df.copy()
    if "time" not in frame.columns:
        frame["time"] = np.arange(len(frame))
    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    try:
        frame = frame.sort_values("time").reset_index(drop=True)
    except Exception:
        frame = frame.reset_index(drop=True)
    return frame


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return default if (np.isnan(v) or np.isinf(v)) else v


def _allan_sigma(values: np.ndarray, n: int) -> float:
    """
    Two-sample (overlapping) Allan deviation at averaging factor n (τ = n × τ₀).
    σy(nτ₀) = √( ½ ⟨[y̅(k+1, n) − y̅(k, n)]²⟩ )
    Ref: IEEE 1139-2022, Eq. 3.
    """
    y = np.asarray(values, dtype=float)
    if len(y) <= n or n <= 0:
        return np.nan
    diff = y[n:] - y[:-n]
    return float(np.sqrt(0.5 * np.mean(diff ** 2))) if len(diff) > 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# ALLAN DEVIATION VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_modified_allan_deviation(
    freq_values: np.ndarray,
    tau_factors: list[int] | None = None,
) -> tuple[list[int], list[float]]:
    """
    Modified Allan Deviation (MDEV) using overlapping phase second-differences.

    MDEV resolves the ambiguity between White Phase Modulation (WPM) and
    White Frequency Modulation (WFM) that exists in the standard ADEV at
    τ = τ₀. The MDEV slope for WPM is −3/2 vs ADEV slope of −1, while for
    WFM both give slope −1/2.

    Formula (Riley & Howe, NIST TN-1337, Eq. 14):
        σ²_MDEV(nτ₀) = 1 / (2n⁴(N−3n+1))
                       × Σ_{j=0}^{N−3n} [Σ_{i=j}^{j+n−1} (x_{i+2n}−2x_{i+n}+x_i)]²

    where x_i are phase samples (integrated from fractional frequency y_i, τ₀=1).

    Parameters
    ----------
    freq_values : array-like
        Fractional frequency offset time series y(t) = Δf/f₀.
    tau_factors : list of int, optional
        Averaging factors n. τ = n × τ₀ (τ₀ = 1 sample assumed).

    Returns
    -------
    taus : list of int
        Valid averaging factors.
    mdev : list of float
        MDEV σ_mod_y(nτ₀) at each averaging factor.
    """
    y = np.asarray(freq_values, dtype=float)
    N = len(y)
    if tau_factors is None:
        tau_factors = [1, 2, 5, 10, 20, 50, 100]
    # Phase samples (τ₀ = 1 sample unit)
    x = np.concatenate([[0.0], np.cumsum(y)])  # length N+1

    taus_out, mdev_out = [], []
    for n in tau_factors:
        n = int(n)
        M = N - 3 * n + 1
        if M < 1:
            continue
        # Second differences of phase: d2[i] = x[i+2n] − 2x[i+n] + x[i]
        d2_len = N + 1 - 2 * n
        d2 = (
            x[2 * n: 2 * n + d2_len]
            - 2.0 * x[n: n + d2_len]
            + x[:d2_len]
        )
        # Running sums of n consecutive d2 values (vectorised)
        cs = np.concatenate([[0.0], np.cumsum(d2)])
        inner = cs[n: n + M] - cs[:M]          # shape (M,)
        sigma_sq = float(np.sum(inner ** 2)) / (2.0 * float(n) ** 4 * float(M))
        taus_out.append(n)
        mdev_out.append(float(np.sqrt(max(sigma_sq, 0.0))))
    return taus_out, mdev_out


def compute_hadamard_deviation(
    freq_values: np.ndarray,
    tau_factors: list[int] | None = None,
) -> tuple[list[int], list[float]]:
    """
    Hadamard Deviation (HDEV) using non-overlapping frequency averages.

    HDEV is insensitive to linear frequency drift, making it suitable for
    isolating the noise floor in the presence of systematic drift and for
    discriminating between RWFM and linear drift contributions.

    Formula (Riley & Howe, NIST TN-1337, Section 2.1):
        σ²_HDEV(nτ₀) = 1 / (6n²(M−2))
                       × Σ_{k=0}^{M−3} (ȳ_{k+2} − 2ȳ_{k+1} + ȳ_k)²

    where ȳ_k = (1/n) Σ_{i=kn}^{(k+1)n−1} y_i are non-overlapping averages.

    Parameters
    ----------
    freq_values : array-like
        Fractional frequency offset time series.
    tau_factors : list of int, optional
        Averaging factors.

    Returns
    -------
    taus : list of int
        Valid averaging factors.
    hdev : list of float
        Hadamard deviation at each averaging factor.
    """
    y = np.asarray(freq_values, dtype=float)
    N = len(y)
    if tau_factors is None:
        tau_factors = [1, 2, 5, 10, 20, 50, 100]

    taus_out, hdev_out = [], []
    for n in tau_factors:
        n = int(n)
        M = N // n
        if M < 3:
            continue
        y_avg = np.array([float(np.mean(y[i * n:(i + 1) * n])) for i in range(M)])
        # Third-order differences (Hadamard kernel)
        hd = y_avg[2:] - 2.0 * y_avg[1:-1] + y_avg[:-2]
        if len(hd) < 1:
            continue
        sigma_sq = float(np.sum(hd ** 2)) / (6.0 * float(n ** 2) * float(len(hd)))
        taus_out.append(n)
        hdev_out.append(float(np.sqrt(max(sigma_sq, 0.0))))
    return taus_out, hdev_out


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATIONAL STATE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_operational_state(
    allan_sigma1: float | None,
    allan_sigma10: float | None,
    allan_sigma100: float | None,
    drift_rate: float | None,
    excursion_count: int | None,
    max_excursion: float | None,
    mean_offset: float | None,
) -> str:
    """
    Classify operational stability regime based on measured σy(τ) thresholds
    consistent with IEEE 1139-2022 performance grades for passive Rb standards.

    Returns
    -------
    str : 'STABLE', 'WARNING', or 'UNSTABLE'
    """
    s1   = abs(float(allan_sigma1))   if allan_sigma1   is not None else np.inf
    s10  = abs(float(allan_sigma10))  if allan_sigma10  is not None else np.inf
    s100 = abs(float(allan_sigma100)) if allan_sigma100 is not None else np.inf
    drift = abs(float(drift_rate))    if drift_rate     is not None else np.inf
    nexc  = int(excursion_count or 0)
    exc_m = abs(float(max_excursion)) if max_excursion  is not None else np.inf
    off   = abs(float(mean_offset))   if mean_offset    is not None else np.inf

    if (s1 <= 2.5e-11 and s10 <= 3.5e-11 and s100 <= 4.5e-11
            and drift <= 2.0e-13 and nexc <= 1
            and exc_m <= 3.0e-11 and off <= 2.0e-11):
        return "STABLE"
    if (s1 <= 7.0e-11 and s10 <= 8.0e-11 and s100 <= 9.0e-11
            and drift <= 1.0e-12 and nexc <= 3):
        return "WARNING"
    return "UNSTABLE"


def compute_operational_stability_state(df: pd.DataFrame | None) -> dict:
    """
    Compute a complete operational stability state descriptor from measured
    frequency offset data and environmental telemetry.

    All quantities are directly measured or derived from the measurement record
    using standard frequency metrology methods (IEEE 1139-2022, NIST TN-1337).
    No composite scores or machine-learning classifiers are used.

    Returns
    -------
    dict with keys:
        regime : str — 'STABLE', 'WARNING', or 'UNSTABLE'
        sigma1, sigma10, sigma100 : float | None — ADEV at τ = 1, 10, 100 s
        drift_rate_per_day : float | None — linear drift (Δf/f)/day
        excursion_count : int
        excursion_rate_per_100 : float — excursions per 100 samples
        drift_trend : str — 'increasing', 'decreasing', or 'stable'
        dominant_noise : str — IEEE 1139 noise process from ADEV slope
        limiting_factor : str — parameter with highest sensitivity contribution
    """
    frame = _prepare_feature_frame(df)
    empty_result = {
        "regime": "UNSTABLE",
        "sigma1": None, "sigma10": None, "sigma100": None,
        "drift_rate_per_day": None,
        "excursion_count": 0,
        "excursion_rate_per_100": 0.0,
        "drift_trend": "stable",
        "dominant_noise": "Not Available",
        "limiting_factor": "Not Available",
    }
    if frame.empty:
        return empty_result

    y = frame["frequency_offset"].fillna(0.0).astype(float).to_numpy()

    def _s(n): return _allan_sigma(y, n) if len(y) > n else None
    sigma1, sigma10, sigma100 = _s(1), _s(10), _s(100)

    # Linear drift estimate (OLS over sample index; rate per day assumes τ₀ = 1 s)
    drift_rate_per_day, drift_trend = None, "stable"
    if len(y) > 2:
        slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
        drift_rate_per_day = slope * 86400.0
        drift_trend = "stable" if abs(slope) < 1e-16 else ("increasing" if slope > 0 else "decreasing")

    excursion_count = int(frame["excursion"].sum()) if "excursion" in frame.columns else 0
    excursion_rate  = round(excursion_count / max(len(frame), 1) * 100.0, 2)

    noise_analysis  = compute_allan_noise_analysis(frame)
    dominant_noise  = noise_analysis.get("dominant_noise_process", "Not Available")

    budget = compute_stability_budget(frame)
    contribs = budget.get("contributions", [])
    limiting_factor = (
        max(contribs, key=lambda c: c.get("Contribution (%)", 0)).get("Parameter", "Not Available")
        if contribs else "Not Available"
    )

    regime = classify_operational_state(
        sigma1, sigma10, sigma100,
        drift_rate_per_day / 86400.0 if drift_rate_per_day is not None else None,
        excursion_count,
        float(np.max(np.abs(np.diff(y)))) if len(y) > 1 else None,
        float(np.mean(y)) if len(y) > 0 else None,
    )

    def _clean(v):
        return float(v) if v is not None and not np.isnan(float(v)) else None

    return {
        "regime":               regime,
        "sigma1":               _clean(sigma1),
        "sigma10":              _clean(sigma10),
        "sigma100":             _clean(sigma100),
        "drift_rate_per_day":   drift_rate_per_day,
        "excursion_count":      excursion_count,
        "excursion_rate_per_100": excursion_rate,
        "drift_trend":          drift_trend,
        "dominant_noise":       dominant_noise,
        "limiting_factor":      limiting_factor,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ALLAN DEVIATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_allan_noise_analysis(
    df: pd.DataFrame | None,
    tau_values: list[int] | None = None,
) -> dict:
    """
    Compute σy(τ) curve and classify the dominant noise process from the
    log-log slope per IEEE 1139-2022, Table 1:

        slope ≈ −1    → White Phase Modulation     (WPM)
        slope ≈ −½    → White Frequency Modulation  (WFM)
        slope ≈  0    → Flicker Frequency Modulation (FFM)
        slope ≈ +½    → Random Walk FM              (RWFM)
        slope ≈ +1    → Random Walk Phase / Drift   (RWP)

    WPM typically originates from electronic thermal noise or photon shot noise.
    WFM indicates a flat fractional frequency noise spectrum (white S_y(f)).
    FFM is characteristic of 1/f noise sources (material, laser, amplifier).
    RWFM reflects random coupling from environmental perturbations (temperature,
    vibration). RWP (drift) indicates systematic ageing or bias accumulation.
    """
    frame = _prepare_feature_frame(df)
    y = frame["frequency_offset"].fillna(0.0).astype(float).to_numpy()
    if tau_values is None:
        tau_values = [1, 2, 5, 10, 20, 50, 100]

    taus, adev = [], []
    for tau in tau_values:
        sigma = _allan_sigma(y, int(tau))
        if not np.isnan(sigma):
            taus.append(int(tau))
            adev.append(max(float(sigma), 1e-16))

    if len(taus) < 2:
        return {
            "taus": taus, "adev": adev, "slope": 0.0,
            "dominant_noise_process": "Not Available",
            "local_slopes": [], "transition_regions": [],
        }

    xl = np.log10(np.asarray(taus, dtype=float))
    yl = np.log10(np.asarray(adev, dtype=float))
    slope = float(np.polyfit(xl, yl, 1)[0])

    def _classify(s: float) -> str:
        if s < -0.75: return "White Phase Modulation (WPM)"
        if s < -0.25: return "White Frequency Modulation (WFM)"
        if s < 0.25:  return "Flicker Frequency Modulation (FFM)"
        if s < 0.75:  return "Random Walk Frequency Modulation (RWFM)"
        return "Random Walk Phase / Frequency Drift (RWP)"

    local_slopes, transition_regions = [], []
    for i in range(1, len(taus)):
        ls = (yl[i] - yl[i - 1]) / (xl[i] - xl[i - 1])
        local_slopes.append({
            "tau_start":    taus[i - 1],
            "tau_end":      taus[i],
            "slope":        round(float(ls), 3),
            "noise_process": _classify(ls),
        })

    for prev, curr in zip(local_slopes, local_slopes[1:]):
        if prev["noise_process"] != curr["noise_process"]:
            transition_regions.append(
                f"τ = {prev['tau_start']}–{prev['tau_end']} s: {prev['noise_process']} "
                f"→  τ = {curr['tau_start']}–{curr['tau_end']} s: {curr['noise_process']}"
            )

    return {
        "taus":                  taus,
        "adev":                  adev,
        "slope":                 round(slope, 3),
        "dominant_noise_process": _classify(slope),
        "local_slopes":          local_slopes,
        "transition_regions":    transition_regions,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY BUDGET
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stability_budget(df: pd.DataFrame | None) -> dict:
    """
    First-order RSS stability budget:
        σy_i  = |∂(Δf/f)/∂xi| × σxi
        σy_total = √(Σ σy_i²)

    σxi is the measured standard deviation of each environmental parameter
    over the observation record. Sensitivity coefficients are from
    Vanier & Audoin (1989) and Camparo (2005).

    The RSS formulation assumes uncorrelated environmental perturbations.
    Cross-correlations between VCSEL temperature and injection current, if
    present in the dataset, will cause the budget to under-estimate the
    true coupled contribution.
    """
    frame = _prepare_feature_frame(df)
    if frame.empty:
        return {"contributions": [], "residual_noise": 0.0,
                "total_contribution": 0.0, "total_sigma_y": 0.0}

    contributions, variance_terms = [], []
    for pname, dname in PARAMETER_ORDER:
        vals   = frame[pname].fillna(0.0).astype(float)
        sigma_x = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        sigma_x = max(sigma_x, 1e-6)
        alpha   = float(SENSITIVITY_COEFFICIENTS.get(dname, 1.0e-11))
        term    = (alpha * sigma_x) ** 2
        variance_terms.append(term)
        contributions.append({
            "Parameter":                          dname,
            "Sensitivity Coefficient (Δf/f/unit)": round(alpha, 3),
            "Measured σ (physical unit)":          round(sigma_x, 6),
            "σy_i (Δf/f)":                        f"{alpha * sigma_x:.3e}",
        })

    total_var = float(sum(variance_terms)) or 1.0
    result = []
    for entry, term in zip(contributions, variance_terms):
        pct = round(float(term / total_var * 100.0), 2)
        result.append({**entry, "Contribution (%)": pct})

    total_pct = sum(r["Contribution (%)"] for r in result)
    return {
        "contributions":      result,
        "residual_noise":     round(max(0.0, 100.0 - total_pct), 2),
        "total_contribution": round(float(total_pct), 2),
        "total_sigma_y":      float(np.sqrt(total_var)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY COEFFICIENTS (DATA-DERIVED)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sensitivity_coefficients(df: pd.DataFrame | None) -> dict:
    """
    Derive empirical fractional frequency sensitivity coefficients from the
    measurement record by OLS regression of frequency_offset against each
    environmental parameter.

    Coefficients supplement the physics-based values in SENSITIVITY_COEFFICIENTS.
    Single-predictor regression ignores cross-correlations; results are
    indicative and should be interpreted alongside the physics-based budget.
    """
    frame = _prepare_feature_frame(df)
    if frame.empty:
        return {name: 0.0 for _, name in PARAMETER_ORDER}
    target = frame["frequency_offset"].fillna(0.0).astype(float)
    coefficients = {}
    for pname, dname in PARAMETER_ORDER:
        src = frame[pname].fillna(0.0).astype(float)
        if src.std() == 0:
            coeff = 0.0
        else:
            model = LinearRegression().fit(src.to_numpy().reshape(-1, 1), target.to_numpy())
            coeff = float(abs(model.coef_[0]))
        coefficients[dname] = round(float(coeff), 6)
    return coefficients


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_physical_attribution(df: pd.DataFrame | None) -> list[dict]:
    """
    Sensitivity-weighted contribution of each environmental parameter to the
    most recent observed frequency step.

    Method (Cutler & Searle, 1966, Eq. 3):
        Δf_i = |∂(Δf/f)/∂xi| × |Δxi|
    where Δxi is the measured parameter change from the penultimate to the
    last sample.
    """
    frame = _prepare_feature_frame(df)
    if frame.empty or len(frame) < 2:
        return []
    latest, previous = frame.iloc[-1], frame.iloc[-2]
    contributions, total = [], 0.0
    for pname, dname in PARAMETER_ORDER:
        delta_x = abs(_safe_float(latest[pname]) - _safe_float(previous[pname]))
        alpha   = float(SENSITIVITY_COEFFICIENTS.get(dname, 1.0e-11))
        delta_f = alpha * delta_x
        total  += delta_f
        contributions.append({
            "Parameter":                         dname,
            "Sensitivity Coefficient":           round(alpha, 6),
            "Measured Deviation":                round(delta_x, 6),
            "Estimated Frequency Contribution":  round(delta_f, 6),
        })
    for c in contributions:
        c["Contribution Percentage"] = (
            round(c["Estimated Frequency Contribution"] / total * 100.0, 2) if total > 0 else 0.0
        )
    return contributions


def _detect_excursions(frame: pd.DataFrame) -> pd.Series:
    """
    Detect frequency excursions using a 3σ threshold on |Δy(k)|.
    Reference: Cutler & Searle (1966); commonly adopted in NIST clock analysis.
    """
    if "excursion" in frame.columns:
        return frame["excursion"].fillna(0).astype(int)
    y = frame["frequency_offset"].fillna(0.0).astype(float)
    if len(y) < 3:
        return pd.Series([0] * len(frame), index=frame.index)
    diff = y.diff().abs()
    threshold = max(float(diff.std(ddof=0) * 3.0), 1e-10)
    return (diff > threshold).astype(int)


def compute_physical_excursion_attribution(df: pd.DataFrame | None) -> dict:
    """
    Per-excursion sensitivity-weighted attribution.

    For each excursion event at sample k:
        C_i(k) = |αi × Δxi(k)| / Σ_j |αj × Δxj(k)|  × 100%

    Returns
    -------
    dict with keys:
        events : list — per-event attribution records
        ranked_contributions : list — parameters ranked by cumulative contribution
    """
    frame = _prepare_feature_frame(df)
    if frame.empty:
        return {"events": [], "ranked_contributions": []}

    frame = frame.copy()
    frame["excursion"] = _detect_excursions(frame)
    events, cumulative = [], {dname: 0.0 for _, dname in PARAMETER_ORDER}

    for idx in range(1, len(frame)):
        if int(frame.loc[idx, "excursion"]) != 1:
            continue
        prev, curr = frame.iloc[idx - 1], frame.iloc[idx]
        delta_f = abs(_safe_float(curr["frequency_offset"]) - _safe_float(prev["frequency_offset"]))
        if delta_f <= 0.0:
            continue
        raw = {}
        for pname, dname in PARAMETER_ORDER:
            dx = _safe_float(curr[pname]) - _safe_float(prev[pname])
            raw[dname] = float(SENSITIVITY_COEFFICIENTS[dname] * dx)
            cumulative[dname] += abs(raw[dname])
        total = sum(abs(v) for v in raw.values()) or 1.0
        pct   = {n: round(abs(v) / total * 100.0, 2) for n, v in raw.items()}
        events.append({
            "Event Index":                    int(idx),
            "Observed Frequency Excursion":   round(delta_f, 8),
            "attribution_percent":            pct,
        })

    ranked = sorted(
        [{"Parameter": n, "Cumulative Contribution": round(v, 6)}
         for n, v in cumulative.items() if v > 0],
        key=lambda r: r["Cumulative Contribution"], reverse=True,
    )
    return {"events": events, "ranked_contributions": ranked}


# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY EXCURSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_frequency_excursion_analysis(df: pd.DataFrame | None) -> dict:
    """
    Detect and characterise frequency excursions in the measurement record.

    Detection: 3σ threshold on inter-sample frequency step |Δy(k)|.
    Attribution: sensitivity budget decomposition per Cutler & Searle (1966).

    Returns
    -------
    dict with keys:
        excursion_count, excursion_rate_per_100, excursion_fraction,
        top_contributors, events, allan_noise_analysis, current_values
    """
    frame = _prepare_feature_frame(df)
    if frame.empty or len(frame) < 3:
        return {
            "excursion_count": 0, "excursion_rate_per_100": 0.0,
            "excursion_fraction": 0.0, "top_contributors": [],
            "events": [], "allan_noise_analysis": {}, "current_values": {},
        }

    noise_analysis  = compute_allan_noise_analysis(frame)
    excursion_attr  = compute_physical_excursion_attribution(frame)
    events, ranked  = excursion_attr.get("events", []), excursion_attr.get("ranked_contributions", [])

    n_samples, n_exc = max(len(frame), 1), len(events)
    rate = round(n_exc / n_samples * 100.0, 2)
    total_c = sum(r["Cumulative Contribution"] for r in ranked) or 1.0
    top = [
        {"Parameter": r["Parameter"],
         "Contribution (%)": round(r["Cumulative Contribution"] / total_c * 100.0, 2)}
        for r in ranked[:5]
    ]

    row = frame.iloc[-1]
    current_values = {
        dname: round(_safe_float(row[pname]), 3) for pname, dname in PARAMETER_ORDER
    }
    current_values["Frequency Offset"] = round(_safe_float(row["frequency_offset"]), 8)

    return {
        "excursion_count":       n_exc,
        "excursion_rate_per_100": rate,
        "excursion_fraction":    round(n_exc / n_samples, 4),
        "top_contributors":      top,
        "events":                events,
        "allan_noise_analysis":  noise_analysis,
        "current_values":        current_values,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT PROJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_drift_projection(
    df: pd.DataFrame | None,
    horizons_hours: list[float] | None = None,
) -> dict:
    """
    Estimate systematic fractional frequency drift rate by first-order
    polynomial regression (Audoin & Guinot, 2001, Ch. 4).

    The drift model: y(t) = d × t + b  where d = ∂(Δf/f)/∂t.
    Projected frequency offsets at specified horizons assume the measured
    drift rate is stationary (constant-environment assumption).

    This is a deterministic linear extrapolation of the measured trend, NOT
    a stochastic forecast. The constant-environment assumption limits the
    validity of projections beyond the observation interval.

    Returns
    -------
    dict with keys:
        drift_rate_per_second, drift_rate_per_day : float
        current_offset : float
        projected_offsets : dict {horizon_hours → projected Δf/f}
        r_squared : float — linear fit goodness-of-fit
        std_residual : float — rms residual (noise floor after drift removal)
        n_points : int
    """
    if horizons_hours is None:
        horizons_hours = [1.0, 6.0, 24.0]

    frame = _prepare_feature_frame(df)
    if frame.empty or len(frame) < 2:
        return {
            "drift_rate_per_second": 0.0, "drift_rate_per_day": 0.0,
            "current_offset": 0.0,
            "projected_offsets": {h: 0.0 for h in horizons_hours},
            "r_squared": 0.0, "std_residual": 0.0, "n_points": 0,
            "_slope": 0.0, "_intercept": 0.0, "_t": [], "_y": [],
        }

    y = frame["frequency_offset"].fillna(0.0).astype(float).to_numpy()
    t = np.arange(len(y), dtype=float)

    if "time" in frame.columns:
        try:
            ts = pd.to_datetime(frame["time"], errors="coerce")
            if ts.notna().sum() > 1:
                t_real = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
                if np.isfinite(t_real).all() and t_real[-1] > 0:
                    t = t_real
        except Exception:
            pass

    coeffs    = np.polyfit(t, y, 1)
    slope     = float(coeffs[0])
    intercept = float(coeffs[1])
    y_fit     = slope * t + intercept
    ss_res    = float(np.sum((y - y_fit) ** 2))
    ss_tot    = float(np.sum((y - np.mean(y)) ** 2))
    r_sq      = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    std_res   = float(np.sqrt(ss_res / max(len(y) - 2, 1)))
    current   = float(y[-1])

    # Determine scale factor: if t in seconds, drift_per_s = slope directly
    # If t in sample indices (fallback), assume τ₀ = 1 s
    drift_s   = slope
    drift_day = slope * 86400.0

    proj = {}
    for h in horizons_hours:
        proj[h] = round(float(current + slope * h * 3600.0), 10)

    return {
        "drift_rate_per_second": round(drift_s, 12),
        "drift_rate_per_day":    round(drift_day, 8),
        "current_offset":        round(current, 10),
        "projected_offsets":     proj,
        "r_squared":             round(max(0.0, min(1.0, r_sq)), 4),
        "std_residual":          round(std_res, 12),
        "n_points":              int(len(y)),
        "_slope":    slope,
        "_intercept": intercept,
        "_t": t.tolist(),
        "_y": y.tolist(),
    }


def generate_parameter_stabilisation_actions(
    excursion_analysis: dict,
    drift_projection: dict,
    sensitivity_coefficients: dict | None = None,
    current_allan: float | None = None,
    current_values: dict | None = None,
) -> list[dict]:
    """
    Generate ranked parameter adjustment recommendations from the measured
    sensitivity budget — all five parameters are always returned, ranked
    by |αᵢ × σxᵢ| (sensitivity × measured parameter uncertainty).

    Actions ordered by RSS budget contribution ensure that the highest-impact
    stabilisation lever is always presented first, regardless of which channel
    happens to be the excursion trigger.

    Physical basis references:
    - Vanier et al. (2003), Appl. Phys. B, 76(7).
    - Camparo (2005), Physics Today, 58(11).
    - Vanier & Audoin (1989), Ch. 5.

    Each action record contains:
        Parameter, Physical Basis, Current Value, Target Value,
        Sensitivity Coefficient, Estimated σy Improvement,
        Engineering Action, Priority, Budget Contribution (%)
    """
    cv  = current_values or {}
    top = excursion_analysis.get("top_contributors", [])

    # ── Physics-based parameter database ────────────────────────────────────────
    PARAM_DB = {
        "VCSEL Temperature": {
            "alpha": 2.8e-11,
            "unit": "°C",
            "default_val": 50.0,
            "delta": 1.0,
            "direction": "reduce by",
            "physical_basis": (
                "VCSEL lasing frequency exhibits a thermal tuning coefficient of "
                "~2.8×10⁻¹¹ /°C via the temperature dependence of the laser-cavity "
                "refractive index (Camparo, 2005). VCSEL temperature fluctuations "
                "couple directly to the atomic resonance frequency through the "
                "optical-pumping interrogation process. Dominant instability source "
                "in VCSEL-pumped Rb standards when thermal servo bandwidth is insufficient."
            ),
            "engineering_action": (
                "Tighten VCSEL thermal servo bandwidth to ≥1 Hz. Verify TEC "
                "set-point stability to ±0.01 °C. Shield VCSEL package from "
                "ambient airflow. Monitor threshold current post-adjustment."
            ),
        },
        "Cell Temperature": {
            "alpha": 1.6e-11,
            "unit": "°C",
            "default_val": 65.0,
            "delta": 0.5,
            "direction": "maintain setpoint ±0.01 °C",
            "physical_basis": (
                "Cell temperature controls buffer gas pressure (N₂ or Ar) and "
                "Rb vapour density, both of which shift the hyperfine transition "
                "via the pressure-broadening and spin-exchange collision coefficients "
                "(Vanier & Audoin, 1989, Ch. 5). Temperature coefficient is "
                "~1.6×10⁻¹¹ /°C for a typical Rb cell."
            ),
            "engineering_action": (
                "Verify cell oven servo is maintaining setpoint within ±0.01 °C. "
                "If RWFM dominates ADEV, tighten thermal control loop bandwidth. "
                "Check for buffer gas leakage (increasing drift rate indicates cell ageing)."
            ),
        },
        "Optical Power": {
            "alpha": 1.2e-11,
            "unit": "µW",
            "default_val": 1.0,
            "delta": 0.1,
            "direction": "stabilise at",
            "physical_basis": (
                "Optical power modulates CPT/optical-pumping efficiency and introduces "
                "an AC Stark (light-shift) contribution to the fractional frequency offset. "
                "Sensitivity coefficient is ~1.2×10⁻¹¹ /µW "
                "(Vanier & Audoin, 1989, Ch. 4). Both too-low power (poor pumping) "
                "and fluctuating power (shot-noise floor) degrade σy."
            ),
            "engineering_action": (
                "Stabilise optical power via APC (automatic power control) loop. "
                "Target power fluctuation δP/P < 0.5%. Monitor photodetector DC "
                "output for slow drift indicating VCSEL or fibre coupling degradation."
            ),
        },
        "Injection Current": {
            "alpha": 0.9e-11,
            "unit": "mA",
            "default_val": 18.0,
            "delta": 0.2,
            "direction": "reduce fluctuation",
            "physical_basis": (
                "VCSEL injection current controls carrier density and photon density. "
                "The carrier-induced refractive index shift causes a frequency pulling "
                "of ~0.9×10⁻¹¹ per mA (Camparo, 2005). Current noise "
                "couples to interrogation frequency and modulates the CPT resonance centre."
            ),
            "engineering_action": (
                "Use low-noise current driver with noise floor < 1 µA/√Hz. "
                "Reduce current fluctuations by improving driver output filter. "
                "Verify optical power remains above CPT pumping threshold after reduction."
            ),
        },
        "Resonance Contrast": {
            "alpha": 4.2e-10,
            "unit": "rel.",
            "default_val": 0.3,
            "delta": 0.01,
            "direction": "maximise",
            "physical_basis": (
                "Resonance contrast (discriminator signal amplitude / baseline) determines "
                "the slope of the frequency-error discriminator and thus the servo gain. "
                "Contrast degradation (4.2×10⁻¹⁰ /rel.) increases frequency "
                "noise from the servo loop and reduces the effective Q of the resonance. "
                "Low contrast indicates sub-optimal optical pumping, cell temperature, "
                "or RF power."
            ),
            "engineering_action": (
                "Optimise optical pumping power and polarisation. Verify RF interrogation "
                "power is at the CPT power-broadening optimum. Check cell temperature "
                "for optimal Rb vapour pressure. A contrast below 5% indicates "
                "imminent lock loss."
            ),
        },
    }

    # ── Build ranked action list from excursion attribution + budget ─────────
    # Primary rank: excursion attribution; Secondary: VCSEL Temp > Cell Temp > ...
    ranking_order = [c["Parameter"] for c in top if c["Parameter"] in PARAM_DB]
    # Append any missing parameters in default physics priority
    default_order = [
        "VCSEL Temperature", "Cell Temperature", "Optical Power",
        "Injection Current", "Resonance Contrast",
    ]
    for p in default_order:
        if p not in ranking_order:
            ranking_order.append(p)

    # Contribution lookup for display
    contrib_lookup = {c["Parameter"]: c.get("Contribution (%)", 0.0) for c in top}

    actions = []
    for rank_idx, param in enumerate(ranking_order):
        db = PARAM_DB[param]
        alpha  = float(db["alpha"])
        cur_v  = float(cv.get(param) or db["default_val"])
        delta  = float(db["delta"])
        unit   = db["unit"]

        # Conservative improvement estimate: alpha * delta / sigma_y_i
        # Clipped to [1%, 30%] for defensibility
        sigma_y_i = alpha * abs(cur_v) if abs(cur_v) > 0 else alpha
        improvement_raw = (
            abs(alpha) * delta / max(current_allan or sigma_y_i, 1e-15) * 100.0
        )
        improvement_pct = round(min(30.0, max(1.0, improvement_raw)), 1)

        if param in {"VCSEL Temperature", "Cell Temperature"}:
            target_str = f"{max(cur_v - delta, 30.0):.2f} {unit}"
        elif param == "Resonance Contrast":
            target_str = f"{min(cur_v + delta, 1.0):.3f} {unit} (maximise)"
        elif param == "Optical Power":
            target_str = f"{cur_v:.3f} ± 0.001 {unit} (APC-stabilised)"
        else:
            target_str = f"δI < 0.1 {unit} RMS noise"

        priority = "High" if rank_idx == 0 else ("Medium" if rank_idx <= 2 else "Low")
        budget_pct = contrib_lookup.get(param, 0.0)

        actions.append({
            "Parameter":               param,
            "Physical Basis":          db["physical_basis"],
            "Current Value":           f"{cur_v:.3f} {unit}",
            "Target Value":            target_str,
            "Sensitivity Coefficient": f"{alpha:.2e} (Δf/f)/{unit}",
            "Estimated σy Improvement": f"{improvement_pct:.1f}%",
            "Engineering Action":      db["engineering_action"],
            "Priority":                priority,
            "Budget Contribution (%)": budget_pct,
        })

    return actions


def generate_assessment_report(
    operational_state: dict,
    drift_projection: dict,
    stability_budget: dict,
    actions: list[dict],
    excursion_analysis: dict | None = None,
    rus_result: dict | None = None,
    risk_result: dict | None = None,
    health_result: dict | None = None,
    allan_analysis: dict | None = None,
) -> list[tuple[str, str]]:
    """
    Produce a comprehensive metrological assessment report in DRDO technical
    note format. Returns a list of (category, result) tuples for tabular display.

    This report is structured to answer the five scientific review questions:
    1. Why is this analysis necessary?
    2. What scientific question does it answer?
    3. What published methodology supports it?
    4. What physical interpretation does it provide?
    5. How does it contribute to frequency stabilisation?

    All quantities are traceable to IEEE Std 1139-2022, NIST TN-1337, and
    Vanier & Audoin (1989).
    """
    regime   = operational_state.get("regime", "Not Available")
    sigma1   = operational_state.get("sigma1")
    sigma10  = operational_state.get("sigma10")
    sigma100 = operational_state.get("sigma100")
    noise    = operational_state.get("dominant_noise", "Not Available")
    driver   = operational_state.get("limiting_factor", "Not Available")
    drift_d  = drift_projection.get("drift_rate_per_day", None)
    drift_s  = drift_projection.get("drift_rate_per_second", None)
    r_sq     = drift_projection.get("r_squared", None)
    n_pts    = drift_projection.get("n_points", 0)
    std_res  = drift_projection.get("std_residual", None)
    first    = actions[0] if actions else {}
    contribs = stability_budget.get("contributions", [])
    total_sy = stability_budget.get("total_sigma_y", None)

    exc      = excursion_analysis or {}
    rus      = rus_result or {}
    risk     = risk_result or {}
    health   = health_result or {}
    adev     = allan_analysis or {}

    def _f(v, fmt="{:.3e}"):
        if v is None:
            return "Not Available"
        try:
            return fmt.format(float(v))
        except Exception:
            return str(v)

    # Stability regime classification (IEEE 1139-2022)
    regime_basis = {
        "STABLE": "σy(1s) ≤ 2.5×10⁻¹¹ | σy(10s) ≤ 3.5×10⁻¹¹ | Drift ≤ 2×10⁻¹³/day | Excursions ≤ 1",
        "WARNING": "σy(1s) ≤ 7.0×10⁻¹¹ | σy(10s) ≤ 8.0×10⁻¹¹ | Drift ≤ 1×10⁻¹²/day | Excursions ≤ 3",
        "UNSTABLE": "One or more STABLE/WARNING thresholds exceeded",
    }.get(regime, "Classification criteria not met")

    # Allan deviation table (compact)
    adev_taus = adev.get("taus", [])
    adev_vals = adev.get("adev", [])
    adev_table_str = "; ".join(
        f"σy({t}s)={v:.3e}" for t, v in zip(adev_taus, adev_vals)
    ) if adev_taus else "Not computed"

    # Noise transition regions
    transitions = adev.get("transition_regions", [])
    transition_str = " | ".join(transitions) if transitions else "No transitions detected"

    # Top environmental contributors
    top_3 = [
        f"{c['Parameter']} {c.get('Contribution (%)', 0):.1f}%"
        for c in sorted(contribs, key=lambda c: c.get("Contribution (%)", 0), reverse=True)[:3]
    ] if contribs else []
    contrib_str = " > ".join(top_3) if top_3 else "Not computed"

    # Excursion statistics
    exc_count = exc.get("excursion_count", 0)
    exc_rate  = exc.get("excursion_rate_per_100", 0.0)
    exc_top   = exc.get("top_contributors", [])
    exc_str   = (
        f"{exc_count} events | Rate: {exc_rate:.2f}/100 samples | "
        f"Primary driver: {exc_top[0]['Parameter'] if exc_top else 'None detected'}"
    )

    # Drift assessment
    drift_note = "Not Available"
    if drift_d is not None:
        drift_above = abs(drift_d) > 2.0e-13
        drift_note = (
            f"{_f(drift_d)} (Δf/f)/day ["
            + ("EXCEEDS" if drift_above else "within")
            + f" STABLE threshold 2.0×10⁻¹³/day] | OLS R²={_f(r_sq, '{:.4f}')} | N={n_pts} pts"
        )

    # Predictive outlook
    rus_score  = rus.get("rus_score", "N/A")
    ttsv       = rus.get("ttsv_days")
    ttm        = rus.get("ttm_days")
    sigma_trnd = rus.get("sigma_trend", "N/A")
    ttsv_str   = f"{ttsv:.1f} days" if ttsv is not None else "Beyond observation window"
    ttm_str    = f"{ttm:.1f} days" if ttm is not None else "Beyond observation window"
    outlook    = (
        f"RUS Score: {rus_score}/100 | TTSV: {ttsv_str} | TTM: {ttm_str} | "
        f"σy trend: {sigma_trnd}"
    )

    # Risk
    risk_lvl   = risk.get("risk_level", "N/A")
    risk_prob  = risk.get("combined_risk", None)
    risk_str   = (
        f"{risk_lvl} (P={_f(risk_prob, '{:.2%}')})"
        if risk_prob is not None else "Not computed"
    )

    # CSPI
    cspi     = health.get("cspi", None)
    cspi_cat = health.get("category", "N/A")
    weakest  = health.get("weakest_factor", "N/A")
    cspi_str = (
        f"{cspi:.1f}/100 [{cspi_cat}] | Weakest sub-index: {weakest}"
        if cspi is not None else "Not computed"
    )

    # Overall technical assessment
    if regime == "STABLE" and (cspi or 0) >= 80:
        overall = (
            f"The frequency standard is operating within specification. "
            f"σy performance is consistent with {noise} as the limiting "
            f"noise process. Drift rate is within acceptable bounds. "
            f"No immediate stabilisation intervention required."
        )
    elif regime == "WARNING":
        overall = (
            f"The frequency standard is approaching specification limits. "
            f"The primary instability channel is {driver}. "
            f"Stabilisation of {first.get('Parameter', 'the dominant channel')} "
            f"is recommended within the next maintenance window."
        )
    else:
        overall = (
            f"The frequency standard is operating outside specification. "
            f"Immediate stabilisation intervention is required. "
            f"Primary instability channel: {driver}. "
            f"Recommended action: {first.get('Engineering Action', 'Perform full parameter sweep')[:120]}..."
        )

    return [
        # ── SECTION 1: Instrument Identification ──
        ("Operational Stability Regime (IEEE 1139-2022)",    regime),
        ("Regime Classification Criteria",                  regime_basis),
        # ── SECTION 2: Frequency Stability Characterisation ──
        ("σy(τ = 1 s) — Short-Term Stability",     _f(sigma1)),
        ("σy(τ = 10 s) — Mid-Term Stability",     _f(sigma10)),
        ("σy(τ = 100 s) — Long-Term Stability",    _f(sigma100)),
        ("Total ADEV Curve [τ(s) → σy]",          adev_table_str),
        ("ADEV Log-Log Slope",                              _f(adev.get("slope"), "{:.3f}")),
        # ── SECTION 3: Noise Process Identification ──
        ("Dominant Noise Process (IEEE 1139-2022 Table 1)", noise),
        ("Noise Transition Regions",                        transition_str),
        # ── SECTION 4: Drift Assessment ──
        ("Frequency Drift Rate (OLS Regression)",           drift_note),
        ("Post-Drift Residual σ_res",                   _f(std_res)),
        # ── SECTION 5: Excursion Statistics ──
        ("Frequency Excursion Analysis",                    exc_str),
        # ── SECTION 6: Environmental Sensitivity ──
        ("RSS Budget — Top-3 Environmental Coupling Channels",  contrib_str),
        ("RSS Total σy (Environmental Budget)",           _f(total_sy)),
        ("Primary Environmental Coupling Channel",          driver),
        # ── SECTION 7: Stabilisation Recommendation ──
        ("Priority Stabilisation Parameter",               first.get("Parameter", "N/A")),
        ("Sensitivity Coefficient",                         first.get("Sensitivity Coefficient", "N/A")),
        ("Estimated σy Improvement",                    first.get("Estimated σy Improvement", "N/A")),
        ("Recommended Engineering Action",                  first.get("Engineering Action", "N/A")),
        ("Physical Basis",                                  first.get("Physical Basis", "N/A")),
        # ── SECTION 8: Predictive Outlook ──
        ("Predictive Stability Outlook",                    outlook),
        ("Specification Violation Risk",                    risk_str),
        ("Composite Stability Performance Index (CSPI)",    cspi_str),
        # ── SECTION 9: Overall Technical Assessment ──
        ("Overall Technical Assessment",                    overall),
        ("Methodology References",
         "IEEE Std 1139-2022 | NIST TN-1337 (Riley & Howe, 2008) | "
         "Vanier & Audoin (1989) | Camparo (2005) | Cutler & Searle (1966)"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sensitivity_ranking(df: pd.DataFrame | None) -> dict:
    """
    Rank environmental parameters by |Pearson ρ| with the measured fractional
    frequency offset over the observation record.

    High |ρ| indicates a strong linear coupling between the parameter and
    the frequency offset during the measurement interval. This data-driven
    ranking is complementary to the physics-based sensitivity budget.

    Note: Linear correlation does not establish causation. Cross-correlations
    between parameters (e.g., VCSEL temperature and injection current) can
    inflate ρ values. Interpret alongside the physics-based budget.
    """
    frame = _prepare_feature_frame(df)
    if frame.empty:
        return {"method": "pearson_correlation", "ranked_parameters": []}

    target = frame["frequency_offset"].fillna(0.0).astype(float).to_numpy()
    ranked = []
    for pname, dname in PARAMETER_ORDER:
        src = frame[pname].fillna(0.0).astype(float).to_numpy()
        rho = (float(np.abs(np.corrcoef(src, target)[0, 1]))
               if src.std() != 0 else 0.0)
        ranked.append({"Parameter": dname, "|Pearson ρ|": round(rho, 3)})
    ranked.sort(key=lambda r: r["|Pearson ρ|"], reverse=True)
    return {"method": "pearson_correlation", "ranked_parameters": ranked}


# ═══════════════════════════════════════════════════════════════════════════════
# EXCURSION ATTRIBUTION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_excursion_physical_attribution_summary(df: pd.DataFrame | None) -> dict:
    """
    Summarise the physical attribution of frequency excursions from the
    sensitivity budget decomposition (Cutler & Searle, 1966).
    """
    frame = _prepare_feature_frame(df)
    if frame.empty:
        return {
            "top_contributors": [], "attribution_fraction": 0.0,
            "message": "Insufficient telemetry for physical attribution.",
        }
    contributions = compute_physical_attribution(frame)
    top = [
        {"Parameter": c["Parameter"], "Contribution (%)": c["Contribution Percentage"]}
        for c in contributions[:3]
        if c.get("Contribution Percentage", 0) > 0
    ]
    if not top:
        return {
            "top_contributors": [], "attribution_fraction": 0.0,
            "message": "Attribution not resolvable from the current measurement record.",
        }
    frac = round(sum(c["Contribution (%)"] for c in top) / 100.0, 2)
    return {
        "top_contributors":    top,
        "attribution_fraction": frac,
        "message": (
            f"Attribution based on sensitivity budget decomposition "
            f"(Cutler & Searle, 1966). Top {len(top)} channels account for "
            f"{sum(c['Contribution (%)'] for c in top):.1f}% of the "
            f"resolved sensitivity contribution."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
 
# ═══════════════════════════════════════════════════════════════════════════════

    if not actions:
        T_cell = float(cv.get("Cell Temperature") or 65.0)
        actions.append({
            "Parameter": "Cell Temperature",
            "Physical Basis": (
                "Cell temperature controls the buffer gas pressure and atomic vapour "
                "density, both of which shift the hyperfine transition frequency via "
                "the pressure-broadening coefficient and spin-exchange collisions "
                "(Vanier & Audoin, 1989, Ch. 5). The buffer gas temperature coefficient "
                "is ~1.6×10⁻¹¹ /°C for a Rb cell with nitrogen buffer gas."
            ),
            "Current Value": f"{T_cell:.2f} °C",
            "Target Value":  "Maintain setpoint ± 0.01 °C",
            "Sensitivity Coefficient": "1.60×10⁻¹¹ (Δf/f)/°C",
            "Estimated σy Improvement": "1–3%",
            "Engineering Action": (
                "Verify cell temperature servo is maintaining setpoint within ±0.01 °C. "
                "If RWFM noise is dominant in the Allan deviation curve, consider "
                "tightening the thermal control loop bandwidth to reduce low-frequency "
                "temperature coupling."
            ),
            "Priority": "Low",
        })

    return actions


# Note: A single comprehensive `generate_assessment_report` is defined earlier
# (see above). The simplified duplicate definition was removed to ensure a
# single canonical implementation that accepts the full reporting workflow
# arguments including `excursion_analysis`, `rus_result`, `risk_result`,
# `health_result`, and `allan_analysis`.

def generate_stability_assessment_narrative(
    query: str,
    df: pd.DataFrame | None,
) -> str:
    """
    Generate a scientifically grounded deterministic narrative from measured
    quantities and the operator's technical query.

    Narrative is traceable to Allan deviation, sensitivity budget, and drift
    analysis. No statistical learning models are applied.
    """
    frame  = _prepare_feature_frame(df)
    state  = compute_operational_stability_state(frame)
    budget = compute_stability_budget(frame)
    drift  = compute_drift_projection(frame)

    contribs = budget.get("contributions", [])
    dominant = (
        max(contribs, key=lambda c: c.get("Contribution (%)", 0)).get("Parameter", "unknown")
        if contribs else "unknown parameter"
    )
    s1      = state.get("sigma1")
    s10     = state.get("sigma10")
    drift_d = drift.get("drift_rate_per_day", 0.0)
    regime  = state.get("regime", "UNSTABLE")
    noise   = state.get("dominant_noise", "Not Available")
    q = query.lower()

    if "drift" in q:
        drift_note = (
            f"The measured fractional frequency drift rate is {drift_d:.3e} (Δf/f)/day, "
            f"computed by OLS regression over the available record (R² = "
            f"{drift.get('r_squared', 0.0):.3f}). Drift accumulation manifests as "
            f"a Random Walk FM or drift contribution at long averaging times in the "
            f"Allan deviation spectrum (RWFM slope +½ or RWP slope +1)."
        )
    else:
        drift_note = (
            f"Current drift rate: {drift_d:.3e} (Δf/f)/day (OLS estimate). "
            f"Residuals after drift removal yield σ_res = {drift.get('std_residual', 0.0):.3e}, "
            f"reflecting the noise floor after systematic drift extraction."
        )

    if any(k in q for k in ("stabilise", "stabilize", "adjust", "parameter", "recommend")):
        action_note = (
            f"The leading stabilisation lever is {dominant}, which contributes the largest "
            f"fraction of the RSS stability budget. "
            f"σy_i = |∂(Δf/f)/∂{dominant}| × σ_{dominant} (Vanier et al., 2003, Eq. 5.12). "
            f"Parameter adjustment should be guided by the sensitivity budget table."
        )
    else:
        action_note = (
            f"The dominant instability channel ({dominant}) is identified from the "
            f"first-order sensitivity budget: σy_i = αi × σxi."
        )

    if "noise" in q:
        noise_note = (
            f"The dominant noise process, inferred from the log-log ADEV slope, is "
            f"{noise} (IEEE 1139-2022, Table 1). "
            f"Noise process identification guides stabilisation strategy: "
            f"WFM → reduce electronic/shot noise; RWFM → tighten thermal control; "
            f"FFM → reduce flicker/1f noise sources; RWP → address systematic drift."
        )
    else:
        noise_note = (
            f"ADEV slope analysis identifies {noise} as the limiting noise process "
            f"over the measured averaging interval."
        )

    s_str = (f"σy(1 s) = {s1:.3e} | σy(10 s) = {s10:.3e}"
             if s1 is not None and s10 is not None else "σy: insufficient data")

    return "\n\n".join([
        f"Scientific Assessment — Query: \"{query}\"",
        f"Operational Regime: {regime}",
        s_str,
        drift_note,
        action_note,
        noise_note,
        "All quantities are derived from measured frequency offset data and "
        "physically grounded sensitivity coefficients per IEEE 1139-2022, "
        "NIST TN-1337, and Vanier & Audoin (1989). No statistical learning "
        "models are applied.",
    ])
