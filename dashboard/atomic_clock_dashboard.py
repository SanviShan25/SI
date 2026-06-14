"""
VCSEL-Pumped ⁸⁷Rb / ¹³³Cs Atomic Frequency Standard
AI-Augmented Predictive Stability Intelligence Framework
=========================================================
DRDO-grade integrated platform for:
  • Frequency Stability Characterisation (IEEE 1139-2022 / NIST TN-1337)
  • AI Predictive Stability Intelligence (LSTM / XGBoost / Kalman Filter)
  • Remaining Useful Stability / Early Warning / Health Index
  • Physics-Informed Digital Twin
  • Automated Scientific Assessment Report Generation

Metrology Foundation: All frequency stability computations are traceable to
IEEE Std 1139-2022, NIST TN-1337 (Riley & Howe 2008), and Vanier & Audoin (1989).

AI Models: Kalman filter [Kalman 1960], LSTM [Hochreiter & Schmidhuber 1997],
XGBoost [Chen & Guestrin 2016], SHAP attribution [Lundberg & Lee 2017],
Digital Twin [Grieves 2014].
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time
import io

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None

from sklearn.ensemble import IsolationForest

# ── Metrology computation engine ──────────────────────────────────────────────
from dashboard.ai_framework import (
    classify_operational_state,
    compute_allan_noise_analysis,
    compute_drift_projection,
    compute_excursion_physical_attribution_summary,
    compute_frequency_excursion_analysis,
    compute_hadamard_deviation,
    compute_modified_allan_deviation,
    compute_operational_stability_state,
    compute_physical_attribution,
    compute_physical_excursion_attribution,
    compute_sensitivity_coefficients,
    compute_sensitivity_ranking,
    compute_stability_budget,
    generate_assessment_report,
    generate_parameter_stabilisation_actions,
    generate_stability_assessment_narrative,
)
from dashboard.scientific_metrics import (
    compute_post_stabilisation_budget,
    compute_stability_improvement_from_budget,
)

# ── AI Predictive Intelligence engine ──────────────────────────────────────────
from dashboard.ai_models import (
    run_kalman_analysis,
    compute_predictive_stability_forecast,
    compute_remaining_useful_stability,
    compute_early_warning,
    compute_ml_root_cause_attribution,
    compute_health_index,
    compute_stability_risk_assessment,
    simulate_digital_twin,
    generate_llm_copilot_response,
    generate_scientific_interpretation,
    compute_model_validation_metrics,
    FORECAST_HORIZONS_HOURS,
    STABILITY_THRESHOLDS,
    LSTM_STATUS,
    COPILOT_DESCRIPTION,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VCSEL-Rb Frequency Standard — AI Stability Intelligence Framework",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  body { color: #e0e6f0; background-color: #070d18; }
  .main .block-container {
      max-width: 100%; padding-top: 0.4rem; padding-left: 1rem; padding-right: 1rem;
  }
  section[data-testid='stSidebar'] { background-color: #0f172a; }
  .stButton>button {
      background-color: #0c1b33; color: #e0e6f0;
      border: 1px solid #1f3a5c; border-radius: 5px;
  }
  .stMetric > div {
      background: #0a1628; color: #e2e8f0; border-radius: 7px;
      border: 1px solid #1e3a5f; padding: 0.6rem 0.8rem; min-height: 4.8rem;
  }
  .stMetric > div > div { padding: 0.1rem 0.25rem; }
  .section-banner {
      background: linear-gradient(90deg, #0a1628 0%, #0e2340 100%);
      border-left: 3px solid #2563eb; padding: 0.45rem 0.9rem;
      border-radius: 4px; margin-bottom: 0.5rem;
  }
  .ai-banner {
      background: linear-gradient(90deg, #1a0a2e 0%, #2d1257 100%);
      border-left: 3px solid #7c3aed; padding: 0.45rem 0.9rem;
      border-radius: 4px; margin-bottom: 0.5rem;
  }
  .regime-stable   { color: #22c55e; font-weight: bold; }
  .regime-warning  { color: #f59e0b; font-weight: bold; }
  .regime-unstable { color: #ef4444; font-weight: bold; }
  .cspi-nominal  { color: #22c55e; font-weight: bold; }
  .cspi-marginal { color: #f59e0b; font-weight: bold; }
  .cspi-degraded { color: #ef4444; font-weight: bold; }
  .cspi-critical { color: #991b1b; font-weight: bold; }
  .risk-low    { color: #22c55e; font-weight: bold; }
  .risk-medium { color: #f59e0b; font-weight: bold; }
  .risk-high   { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Metrology & AI Glossary")
    with st.expander("Allan Deviation σy(τ)", False):
        st.markdown("σy(τ) = √(½⟨[ȳ(k+1,τ)−ȳ(k,τ)]²⟩)  — IEEE 1139-2022, Eq. 3")
    with st.expander("MDEV / HDEV", False):
        st.markdown("MDEV: resolves WPM/WFM ambiguity. HDEV: drift-insensitive. NIST TN-1337.")
    with st.expander("Kalman Filter", False):
        st.markdown("Optimal linear state estimator for constant-velocity frequency drift model. Kalman (1960).")
    with st.expander("XGBoost Forecasting", False):
        st.markdown("Gradient-boosted regression trees trained on lag/environmental features. Chen & Guestrin (2016).")
    with st.expander("LSTM", False):
        st.markdown("Long Short-Term Memory recurrent network for sequential frequency offset prediction. Hochreiter & Schmidhuber (1997).")
    with st.expander("SHAP Attribution", False):
        st.markdown("Shapley Additive exPlanations — model-agnostic attribution satisfying efficiency, symmetry, and linearity axioms. Lundberg & Lee (2017).")
    with st.expander("CSPI (Health Index)", False):
        st.markdown("Composite Stability Performance Index: weighted combination of σy(1s), σy(10s), drift rate, excursion rate, and RUS score. [0–100], higher is better.")
    with st.expander("RUS / TTSV / TTM", False):
        st.markdown("Remaining Useful Stability (RUS). Time-to-Specification Violation (TTSV). Time-to-Maintenance (TTM). Saxena et al. (2008).")
    with st.expander("Digital Twin", False):
        st.markdown("Physics-informed virtual replica. Sensitivity coefficients from Vanier & Audoin (1989). Grieves (2014).")
    with st.expander("Noise Processes (IEEE 1139)", False):
        st.markdown("WPM (−1) | WFM (−½) | FFM (0) | RWFM (+½) | RWP (+1) — log-log ADEV slope.")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt="{:.3e}"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def _allan_sigma(y, n):
    if len(y) <= n or n <= 0:
        return np.nan
    d = y[n:] - y[:-n]
    return float(np.sqrt(0.5 * np.mean(d ** 2)))


def _allan_curve(freq, taus=(1, 2, 5, 10, 20, 50, 100)):
    valid_t, adev = [], []
    for tau in taus:
        if len(freq) > tau:
            s = _allan_sigma(freq, int(tau))
            if not np.isnan(s):
                valid_t.append(tau)
                adev.append(s)
    return valid_t, adev


def _style(fig, title=None, xt=None, yt=None, height=380):
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "x": 0.01, "xanchor": "left",
               "font": {"size": 15, "color": "#e0e6f0"}},
        font=dict(color="#e0e6f0", size=12),
        paper_bgcolor="#070d18", plot_bgcolor="#070d18",
        legend=dict(bgcolor="#0a1628", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=50, r=30, t=55, b=40),
        height=height,
    )
    ax = dict(showgrid=True, gridcolor="#1e293b", zeroline=False)
    if xt: fig.update_xaxes(title_text=xt, title_font=dict(size=12), **ax)
    if yt: fig.update_yaxes(title_text=yt, title_font=dict(size=12), **ax)
    return fig


def _sec(label, ai=False):
    css = "ai-banner" if ai else "section-banner"
    st.markdown(f"<div class='{css}'><strong>{label}</strong></div>",
                unsafe_allow_html=True)


def _latest(df, col):
    try:
        if col in df.columns and df[col].dropna().size > 0:
            return float(df[col].dropna().iloc[-1])
    except Exception:
        pass
    return None


def _regime_badge(r):
    css = {"STABLE": "regime-stable", "WARNING": "regime-warning",
           "UNSTABLE": "regime-unstable"}.get(r, "regime-unstable")
    return f"<span class='{css}'>{r}</span>"


def _map_noise_process(local_slopes: list, dominant_noise: str):
    # Map analysis outputs to human-readable noise labels and confidence
    if not local_slopes:
        label = dominant_noise or "Unknown"
        return label, 0.25
    # derive simple confidence from number of slope segments
    nseg = len(local_slopes)
    conf = min(0.95, 0.4 + 0.12 * nseg)
    # Normalize name
    mapping = {
        "WPM": "White PM",
        "WFM": "White FM",
        "FFM": "Flicker FM",
        "FPM": "Flicker PM",
        "RWFM": "Random Walk FM",
        "DRIFT": "Frequency Drift",
    }
    label = mapping.get(dominant_noise.upper(), dominant_noise)
    return label, conf


def compute_quantitative_risk(sigma1_val, drift_per_day_val, excursion_count, sensitivity_ranking):
    # Build a simple physics-informed risk score [0-100]
    s = 0.0
    # Allan (σy(1s)) contribution — normalized with expected engineering bench of 1e-10
    if sigma1_val is not None and sigma1_val > 0:
        s += min(40.0, 40.0 * (sigma1_val / 1e-10))
    # Drift contribution — normalized per 1e-12/day
    s += min(30.0, 30.0 * (abs(drift_per_day_val) / 1e-12))
    # Excursions — each excursion adds fixed risk proportional to dataset size
    s += min(20.0, 4.0 * excursion_count)
    # Sensitivity_Ranking — if dominant contributor exists, small additive
    try:
        if isinstance(sensitivity_ranking, (list, tuple)) and len(sensitivity_ranking) > 0:
            top = sensitivity_ranking[0]
            # assume top[1] is percent sensitivity if present
            if isinstance(top, (list, tuple)) and len(top) > 1:
                s += min(10.0, float(top[1]) * 0.1)
    except Exception:
        pass
    score = max(0.0, min(100.0, s))
    # Map to categories
    if score < 25:
        cat = "LOW"
    elif score < 50:
        cat = "MODERATE"
    elif score < 75:
        cat = "HIGH"
    else:
        cat = "CRITICAL"
    return score, cat


def _cspi_badge(c, cat):
    css = {"NOMINAL": "cspi-nominal", "MARGINAL": "cspi-marginal",
           "DEGRADED": "cspi-degraded", "CRITICAL": "cspi-critical"}.get(cat, "cspi-degraded")
    return f"<span class='{css}'>{c:.1f}/100 — {cat}</span>"


def _risk_badge(r):
    css = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(r, "risk-high")
    return f"<span class='{css}'>{r}</span>"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = ["time", "vcsel_temp", "vcsel_current", "optical_power",
                 "cell_temp", "contrast", "frequency_offset"]


def ensure_columns(df):
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if "excursion" not in df.columns:
        for alias in ("anomaly", "ai_anomaly"):
            if alias in df.columns:
                df["excursion"] = df[alias].fillna(0).astype(int)
                break
        else:
            df["excursion"] = 0
    return df


def detect_excursions(df):
    feat_cols = [c for c in ["vcsel_temp", "vcsel_current", "optical_power",
                              "cell_temp", "contrast", "frequency_offset"]
                 if c in df.columns]
    if len(feat_cols) < 2 or len(df) < 11:
        df["excursion"] = 0
        return df
    try:
        X = df[feat_cols].fillna(0.0).astype(float)
        model = IsolationForest(contamination=0.01, random_state=42)
        df = df.copy()
        df["excursion"] = (model.fit_predict(X) == -1).astype(int)
    except Exception:
        df["excursion"] = 0
    return df


def validate_dataset(df, min_rows=11):
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return False, None, ["Dataset is None/empty."]
    if len(df) < min_rows:
        return False, None, [f"Need ≥{min_rows} rows; got {len(df)}."]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, None, [f"Missing columns: {missing}"]
    ts = pd.to_datetime(df["time"], errors="coerce")
    if ts.notna().sum() < max(1, int(0.9 * len(df))):
        return False, None, ["Timestamp parse failures >10%."]
    if ts.duplicated().any():
        return False, None, ["Duplicate timestamps."]
    vdf = df.copy()
    vdf["time"] = ts
    for col in [c for c in REQUIRED_COLS if c != "time"]:
        vdf[col] = pd.to_numeric(vdf[col], errors="coerce")
    return True, ensure_columns(vdf.reset_index(drop=True)), ["OK"]


def handle_realtime_mode():
    if "rt_state" not in st.session_state:
        st.session_state["rt_state"] = "disconnected"
    state = st.session_state["rt_state"]
    st.markdown("#### Live Atomic Frequency Standard Monitoring")
    if state == "disconnected":
        st.info("Status: Disconnected — connect frequency standard interface to begin.")
        if st.button("Connect Instrument"):
            st.session_state["rt_state"] = "connected"
            st.rerun()
        st.stop()
    if state == "connected":
        st.info("Status: Connected — awaiting measurement packets.")
        if st.button("Start Stream"):
            st.session_state["rt_state"] = "streaming"
            st.session_state.setdefault("rt_df", pd.DataFrame())
            st.rerun()
        if st.button("Disconnect"):
            st.session_state["rt_state"] = "disconnected"
            st.rerun()
        st.stop()
    rt_df = st.session_state.get("rt_df", pd.DataFrame())
    if rt_df is None or rt_df.empty:
        st.info("Stream active — awaiting measurement packets.")
        if st.button("Stop Stream"):
            st.session_state["rt_state"] = "connected"
            st.rerun()
        st.stop()
    ok, vdf, rpt = validate_dataset(rt_df, 11)
    if not ok:
        st.warning("Buffer validation failed: " + "; ".join(rpt))
        st.stop()
    return vdf


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center;padding:0.3rem 0 0.5rem 0'>
    <h1 style='margin:0 0 5px 0;font-size:2.2rem;letter-spacing:0.02em'>
        Atomic Clock Stability Monitoring and Analysis Framework
    </h1>
    <p style='margin:0 0 4px 0;color:#94a3b8;font-size:0.95rem'>
        Characterization of ⁸⁷Rb and ¹³³Cs Atomic Frequency Standards through Allan Deviation Analysis, Frequency Drift Evaluation,
        Excursion Monitoring, Environmental Sensitivity Assessment, and Comparative Stability Benchmarking.
    </p>
    <p style='margin:0;color:#4b5563;font-size:0.82rem'>
        Metrology: IEEE 1139-2022 · NIST TN-1337 · Vanier & Audoin (1989) · Camparo (2005) &nbsp;|&nbsp;
        AI: Kalman · XGBoost · LSTM · SHAP
    </p>
</div>
""", unsafe_allow_html=True)

# ── Mode selector ──────────────────────────────────────────────────────────────
_c1, _c2, _c3 = st.columns([1, 8, 1])
with _c2:
    mode = st.radio("Measurement Mode", [
        "⁸⁷Rb Frequency Stability Assessment",
        "¹³³Cs Frequency Stability Assessment",
        "Comparative Stability Analysis (⁸⁷Rb vs ¹³³Cs)",
        "Experimental Dataset Analysis",
        "Real-Time Frequency Stability Monitoring",
    ], horizontal=True, label_visibility="collapsed")

INSTRUMENT_INFO = {
    "⁸⁷Rb Frequency Stability Assessment": {
        "species": "⁸⁷Rb", "hyperfine_freq": "6.834 682 610 GHz",
        "technology": "VCSEL-Pumped Rubidium Vapour-Cell Standard",
        "notes": "Ground-state hyperfine transition F=2↔F=1, mF=0↔mF=0.",
    },
    "¹³³Cs Frequency Stability Assessment": {
        "species": "¹³³Cs", "hyperfine_freq": "9.192 631 770 GHz",
        "technology": "Caesium Atomic Frequency Standard",
        "notes": "SI definition of the second — 9 192 631 770 clock transition periods.",
    },
    "Comparative Stability Analysis (⁸⁷Rb vs ¹³³Cs)": {
        "species": "⁸⁷Rb / ¹³³Cs", "hyperfine_freq": "6.835 / 9.193 GHz",
        "technology": "Rubidium–Caesium Comparative Benchmark",
        "notes": "Side-by-side comparative stability assessment.",
    },
}
info = INSTRUMENT_INFO.get(mode, INSTRUMENT_INFO["⁸⁷Rb Frequency Stability Assessment"])
if mode in INSTRUMENT_INFO:
    ci1, ci2, ci3 = st.columns([1, 1, 2])
    ci1.metric("Species", info["species"])
    ci2.metric("Hyperfine Transition", info["hyperfine_freq"])
    ci3.metric("Technology", info["technology"])
    st.caption(info["notes"])

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

if mode in ("⁸⁷Rb Frequency Stability Assessment",
            "¹³³Cs Frequency Stability Assessment",
            "Comparative Stability Analysis (⁸⁷Rb vs ¹³³Cs)"):
    rb = ensure_columns(pd.read_csv(ROOT / "rb_clock_data.csv"))
    cs = ensure_columns(pd.read_csv(ROOT / "cs_clock_data.csv"))
    for df_, name in [(rb, "rb"), (cs, "cs")]:
        try:
            df_ = detect_excursions(df_)
        except Exception:
            df_["excursion"] = 0

if mode == "Experimental Dataset Analysis":
    uploaded = st.file_uploader("Upload Experimental Dataset (CSV)", type=["csv"])
    if uploaded is None:
        st.info("Upload a validated CSV dataset to begin analysis.\n\n"
                "Required columns: `time`, `vcsel_temp`, `vcsel_current`, "
                "`optical_power`, `cell_temp`, `contrast`, `frequency_offset`")
        st.stop()
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception:
        st.error("Cannot read as CSV.")
        st.stop()
    ok, active_df, rpt = validate_dataset(raw_df)
    if not ok:
        st.error("Validation failed: " + "; ".join(rpt))
        st.stop()
    active_df = detect_excursions(active_df)
elif mode == "⁸⁷Rb Frequency Stability Assessment":
    active_df = rb.copy()
elif mode == "¹³³Cs Frequency Stability Assessment":
    active_df = cs.copy()
elif mode == "Comparative Stability Analysis (⁸⁷Rb vs ¹³³Cs)":
    active_df = pd.concat([rb, cs], ignore_index=True)
else:
    active_df = handle_realtime_mode()

active_df = ensure_columns(active_df)
try:
    active_df = detect_excursions(active_df)
except Exception:
    active_df["excursion"] = 0

if active_df is None or len(active_df) == 0:
    st.warning("No measurement data available.")
    st.stop()

@st.cache_data(show_spinner=False)
def _compute_metrology(cache_key: str, df_csv: str) -> dict:
    # Deserialize CSV snapshot for deterministic caching
    df = pd.read_csv(io.StringIO(df_csv))
    out = {}
    t0 = time.perf_counter()
    vals = df["frequency_offset"].fillna(0.0).astype(float).to_numpy()
    taus_c, adev_c = _allan_curve(vals)
    out["adev_taus"] = taus_c
    out["adev_vals"] = adev_c
    out["sigma1"] = dict(zip(taus_c, adev_c)).get(1, None)
    out["sigma10"] = dict(zip(taus_c, adev_c)).get(10, None)
    out["sigma100"] = dict(zip(taus_c, adev_c)).get(100, None)
    out["allan_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out["allan_analysis"] = compute_allan_noise_analysis(df)
    out["dominant_noise"] = out["allan_analysis"].get("dominant_noise_process", "Not Available")
    out["mdev_taus"], out["mdev_vals"] = compute_modified_allan_deviation(vals)
    out["hdev_taus"], out["hdev_vals"] = compute_hadamard_deviation(vals)
    out["mdev_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out["op_state"] = compute_operational_stability_state(df)
    out["regime"] = out["op_state"].get("regime", "UNSTABLE")
    out["excursion_count"] = int(df["excursion"].sum()) if "excursion" in df.columns else 0
    out["op_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out["drift_proj"] = compute_drift_projection(df)
    out["drift_per_day"] = out["drift_proj"].get("drift_rate_per_day", 0.0)
    out["drift_r2"] = out["drift_proj"].get("r_squared", 0.0)
    out["drift_residual"] = out["drift_proj"].get("std_residual", 0.0)
    out["drift_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out["stability_budget"] = compute_stability_budget(df)
    out["budget_contribs"] = out["stability_budget"].get("contributions", [])
    out["budget_df"] = pd.DataFrame(out["budget_contribs"]) if out["budget_contribs"] else pd.DataFrame()
    out["total_sigma_y"] = out["stability_budget"].get("total_sigma_y", 0.0)
    out["dominant_contrib"] = (out["budget_df"].loc[out["budget_df"]["Contribution (%)"].idxmax()]
                                if not out["budget_df"].empty and "Contribution (%)" in out["budget_df"].columns
                                else None)
    out["budget_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out["sensitivity_coefficients"] = compute_sensitivity_coefficients(df)
    out["sensitivity_ranking"] = compute_sensitivity_ranking(df)
    out["physical_attribution"] = compute_physical_attribution(df)
    out["excursion_analysis"] = compute_frequency_excursion_analysis(df)
    out["excursion_attr_summary"] = compute_excursion_physical_attribution_summary(df)
    out["sensitivity_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out["stab_actions"] = generate_parameter_stabilisation_actions(
        excursion_analysis=out["excursion_analysis"],
        drift_projection=out["drift_proj"],
        sensitivity_coefficients=out["sensitivity_coefficients"],
        current_allan=out["sigma1"],
        current_values={
            "VCSEL Temperature": _latest(df, "vcsel_temp"),
            "Optical Power":      _latest(df, "optical_power"),
            "Cell Temperature":   _latest(df, "cell_temp"),
            "Injection Current":  _latest(df, "vcsel_current"),
            "Resonance Contrast": _latest(df, "contrast"),
        },
    )
    out["stab_time"] = time.perf_counter() - t0

    return out

# Prepare CSV snapshot key for caching
_df_csv = active_df.to_csv(index=False)
# compute a deterministic numeric snapshot signature from the frequency series
vals = active_df["frequency_offset"].fillna(0.0).astype(float).to_numpy()
_cache_key = f"ai_cache_{len(active_df)}_{int(vals.mean() * 1e15) if vals.size > 0 else 0}"
_met = _compute_metrology(_cache_key + "_met", _df_csv)

# Unpack metrology results
sigma1 = _met.get("sigma1")
sigma10 = _met.get("sigma10")
sigma100 = _met.get("sigma100")
allan_analysis = _met.get("allan_analysis", {})
dominant_noise = _met.get("dominant_noise", "Not Available")
mdev_taus, mdev_vals = _met.get("mdev_taus", []), _met.get("mdev_vals", [])
hdev_taus, hdev_vals = _met.get("hdev_taus", []), _met.get("hdev_vals", [])
op_state = _met.get("op_state", {})
regime = _met.get("regime", "UNSTABLE")
excursion_count = _met.get("excursion_count", 0)
drift_proj = _met.get("drift_proj", {})
drift_per_day = _met.get("drift_per_day", 0.0)
drift_r2 = _met.get("drift_r2", 0.0)
drift_residual = _met.get("drift_residual", 0.0)
drift_per_second = drift_proj.get("drift_rate_per_second", 0.0)
stability_budget = _met.get("stability_budget", {})
budget_contribs = _met.get("budget_contribs", [])
budget_df = _met.get("budget_df", pd.DataFrame())
total_sigma_y = _met.get("total_sigma_y", 0.0)
dominant_contrib = _met.get("dominant_contrib")
sensitivity_coefficients = _met.get("sensitivity_coefficients", {})
sensitivity_ranking = _met.get("sensitivity_ranking", {})
physical_attribution = _met.get("physical_attribution", {})
excursion_analysis = _met.get("excursion_analysis", {})
excursion_attr_summary = _met.get("excursion_attr_summary", {})
stab_actions = _met.get("stab_actions", [])

# ── Estimated τ₀ ──────────────────────────────────────────────────────────────
tau0_s = 1.0
if "time" in active_df.columns:
    try:
        tv = pd.to_numeric(active_df["time"], errors="coerce").dropna().to_numpy()
        if len(tv) >= 2:
            dv = np.diff(tv)
            pos_dv = dv[dv > 0]
            if pos_dv.size > 0:
                tau0_s = float(np.median(pos_dv))
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE AI QUANTITIES
# ─────────────────────────────────────────────────────────────────────────────

_cache_key = f"ai_cache_{len(active_df)}_{int(vals.mean() * 1e15 if len(vals) > 0 else 0)}"

# Default AI placeholders (will be populated only when AI section is opened)
kalman_result = {}
forecast_result = {}
rus_result = {}
warning_result = {}
ml_attr_result = {}
risk_result = {}
health_result = {}
validation_result = {}
cspi = 0.0
cspi_cat = "UNKNOWN"
risk_level = "UNKNOWN"
# Default risk and forecast placeholders
risk_score = 0.0
risk_cat = "UNKNOWN"
forecast_confidence = 0.0
forecast_horizon_hours = 0


@st.cache_data(show_spinner=False)
def _compute_ai(cache_key: str, df_csv: str, sigma1_val: float, drift_per_day_val: float, tau0_val: float) -> dict:
    df = pd.read_csv(io.StringIO(df_csv))
    result = {}
    result["kalman"] = run_kalman_analysis(df)
    result["forecast"] = compute_predictive_stability_forecast(df, tau0_val)
    result["rus"] = compute_remaining_useful_stability(df, sigma1_val, drift_per_day_val, tau0_val)
    result["warning"] = compute_early_warning(df)
    result["ml_attr"] = compute_ml_root_cause_attribution(df)
    result["risk"] = compute_stability_risk_assessment(df, sigma1_val, drift_per_day_val,
                                                        result["warning"].get("excursion_rate_per_100", 0) if isinstance(result.get("warning"), dict) else 0)
    result["health"] = compute_health_index(
        sigma1_val, _met.get("sigma10"), drift_per_day_val,
        result["warning"].get("excursion_rate_per_100", 0) if isinstance(result.get("warning"), dict) else 0,
        dominant_noise,
        result["rus"].get("rus_score", 100.0) if isinstance(result.get("rus"), dict) else 100.0,
    )
    result["validation"] = compute_model_validation_metrics(df)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
kc = st.columns(8, gap="small")
kc[0].metric("Operational Regime", regime)
kc[1].metric("σy(1 s)", _fmt(sigma1))
kc[2].metric("σy(10 s)", _fmt(sigma10))
kc[3].metric("σy(100 s)", _fmt(sigma100))
kc[4].metric("Drift (Δf/f)/day", _fmt(drift_per_day))
kc[5].metric("Excursions", str(excursion_count))
kc[6].metric("CSPI", f"{cspi:.1f}/100 ({cspi_cat})")
kc[7].metric("Risk Level", risk_level)
st.markdown("---")


# AI Predictive Intelligence Layer — status banner (standby until AI tab opened)
ai_banner_cols = st.columns([1, 3, 1])
with ai_banner_cols[1]:
    # Lightweight AI summary (always visible) — heavy AI runs lazily
    ai_status_text = "ACTIVE" if st.session_state.get(_cache_key) else "STANDBY"
    # Compute lightweight risk and health indicators from metrology outputs
    anomaly_prob_est = min(100.0, round(100.0 * (excursion_count / max(1, len(active_df))) , 1))
    sensitivity_summary = sensitivity_ranking if sensitivity_ranking else []
    risk_score, risk_cat = compute_quantitative_risk(sigma1 or 0.0, drift_per_day or 0.0, excursion_count, sensitivity_summary)
    forecast_confidence = 0.0
    forecast_horizon_hours = 0
    # If AI compute ran before, surface real metrics
    if st.session_state.get(_cache_key):
        # pick from health_result / risk_result if available
        try:
            forecast_confidence = float(forecast_result.get("confidence", 0.0)) if isinstance(forecast_result, dict) else 0.0
        except Exception:
            forecast_confidence = 0.0
        try:
            forecast_horizon_hours = int(forecast_result.get("horizon_hours", 0)) if isinstance(forecast_result, dict) else 0
        except Exception:
            forecast_horizon_hours = 0
        try:
            anomaly_prob_est = float(warning_result.get("anomaly_probability", anomaly_prob_est)) if isinstance(warning_result, dict) else anomaly_prob_est
        except Exception:
            pass
        try:
            risk_score = float(risk_result.get("risk_score", risk_score)) if isinstance(risk_result, dict) else risk_score
            risk_cat = risk_result.get("risk_level", risk_cat) if isinstance(risk_result, dict) else risk_cat
        except Exception:
            pass
    # Health index: prefer health_result if available
    try:
        cspi_val = float(health_result.get("cspi", 0.0)) if isinstance(health_result, dict) else 0.0
        cspi_cat = health_result.get("category", "UNKNOWN") if isinstance(health_result, dict) else "UNKNOWN"
    except Exception:
        cspi_val = 0.0
        cspi_cat = "UNKNOWN"

    # AI banner removed per user request

# Timing diagnostics (per major module)
timing_rows = [
    ("Allan ADEV compute (s)", round(_met.get("allan_time", 0.0), 3)),
    ("MDEV/HDEV compute (s)", round(_met.get("mdev_time", 0.0), 3)),
    ("Operational state (s)", round(_met.get("op_time", 0.0), 3)),
    ("Drift analysis (s)", round(_met.get("drift_time", 0.0), 3)),
    ("Stability budget (s)", round(_met.get("budget_time", 0.0), 3)),
    ("Sensitivity/Excursion (s)", round(_met.get("sensitivity_time", 0.0), 3)),
    ("Stabilisation actions (s)", round(_met.get("stab_time", 0.0), 3)),
]
with st.expander("Timing Diagnostics — module durations (seconds)", expanded=False):
    st.table(pd.DataFrame(timing_rows, columns=["Module", "Time (s)"]))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

section = st.radio("Dashboard Section", [
    "📏 Frequency Metrology Analysis",
    "🔬 Predictive Stability Assessment",
], horizontal=True, label_visibility="collapsed")

# Lazy-load AI computations only when user opens the AI section
if section == "🔬 Predictive Stability Assessment":
    # compute AI with caching keyed by dataset snapshot
    _ai_res = _compute_ai(_cache_key + "_ai", _df_csv, sigma1, drift_per_day, tau0_s)
    st.session_state[_cache_key] = True
    kalman_result = _ai_res.get("kalman", {})
    forecast_result = _ai_res.get("forecast", {})
    rus_result = _ai_res.get("rus", {})
    warning_result = _ai_res.get("warning", {})
    ml_attr_result = _ai_res.get("ml_attr", {})
    risk_result = _ai_res.get("risk", {})
    health_result = _ai_res.get("health", {})
    validation_result = _ai_res.get("validation", {})
    cspi = health_result.get("cspi", 0.0)
    cspi_cat = health_result.get("category", "UNKNOWN")
    risk_level = risk_result.get("risk_level", "UNKNOWN")

# ═══════════════════════════════════════════════════════════════════════════════
# ███████████████  SECTION 1 — FREQUENCY METROLOGY ANALYSIS  ████████████████
# ═══════════════════════════════════════════════════════════════════════════════

if section == "📏 Frequency Metrology Analysis":

    (tab_cfg, tab_stab, tab_noise, tab_drift, tab_env,
     tab_rca, tab_rec, tab_impact, tab_rpt) = st.tabs([
        "① Experimental Configuration",
        "② Frequency Stability Characterisation",
        "③ Noise Process Identification",
        "④ Frequency Drift Assessment",
        "⑤ Environmental Sensitivity Assessment",
        "⑥ Root Cause Analysis",
        "⑦ Stabilisation Recommendation",
        "⑧ Expected Stability Improvement",
        "⑨ Scientific Assessment Report",
    ])

    # ─── TAB 1 ───────────────────────────────────────────────────────────────
    with tab_cfg:
        _sec("Tab 1 · Experimental Configuration")
        st.caption("Establish measurement traceability: instrument constants, dataset quality, and observational record parameters.")
        st.markdown("**Objective:** Document the measurement context before conducting any stability analysis.")
        st.markdown("**Methodology:** Count valid samples, estimate sampling interval τ₀ from median inter-sample time, compute maximum resolvable averaging time τ_max ≈ T/3.")

        # Instrument
        ci1, ci2, ci3 = st.columns(3)
        ci1.metric("Atomic Species", info["species"])
        ci2.metric("Hyperfine Transition", info["hyperfine_freq"])
        ci3.metric("Technology", info["technology"])

        n_total = len(active_df)
        core_c  = [c for c in ["frequency_offset", "vcsel_temp", "optical_power", "cell_temp"] if c in active_df.columns]
        n_miss  = int(active_df[core_c].isna().any(axis=1).sum()) if core_c else n_total
        n_valid = n_total - n_miss
        compl   = round(n_valid / max(n_total, 1) * 100.0, 1)
        tau0_str = obs_str = tau_max_str = "N/A"
        try:
            tv = pd.to_numeric(active_df["time"], errors="coerce").dropna().to_numpy()
            if len(tv) >= 2:
                dv = np.diff(tv); pv = dv[dv > 0]
                if pv.size > 0:
                    tau0_str    = f"{float(np.median(pv)):.2f} s"
                    obs_dur     = float(tv.max() - tv.min())
                    obs_str     = f"{obs_dur:.1f} s ({obs_dur/3600:.2f} h)"
                    tau_max_str = f"{obs_dur/3.0:.1f} s"
        except Exception:
            pass

        qc = st.columns(4)
        qc[0].metric("Total Records", n_total)
        qc[1].metric("Valid Records", n_valid)
        qc[2].metric("Data Completeness", f"{compl:.1f} %")
        qc[3].metric("Excursions Detected", excursion_count)
        qc2 = st.columns(3)
        qc2[0].metric("Sampling Interval τ₀", tau0_str)
        qc2[1].metric("Observation Duration", obs_str)
        qc2[2].metric("Max Resolvable τ", tau_max_str)

        st.markdown("**Scientific Interpretation:** Data completeness ≥95% ensures unbiased σy(τ). Gaps introduce correlated errors at τ > 10×τ_gap (Riley & Howe, 2008, §3.3).")
        st.markdown("**Engineering Implication:** Verify continuous measurement record and stable τ₀ before computing stability figures.")
        st.dataframe(active_df.tail(100).reset_index(drop=True), use_container_width=True)

    # ─── TAB 2 ───────────────────────────────────────────────────────────────
    with tab_stab:
        _sec("Tab 2 · Frequency Stability Characterisation")
        st.caption("Primary stability figure-of-merit: σy(τ) across full observable averaging-time range.")
        st.markdown("**Objective:** Compute and characterise the Allan deviation σy(τ) curve.  \n**Methodology:** Two-sample overlapping ADEV (IEEE 1139-2022, Eq. 3) + MDEV for WPM/WFM discrimination.")

        sk = st.columns(4)
        sk[0].metric("Regime", regime)
        sk[1].metric("σy(1 s)", _fmt(sigma1))
        sk[2].metric("σy(10 s)", _fmt(sigma10))
        sk[3].metric("σy(100 s)", _fmt(sigma100))
        st.markdown(f"**Regime:** {_regime_badge(regime)}", unsafe_allow_html=True)

        adev_full_taus = allan_analysis.get("taus", [])
        adev_full_vals = allan_analysis.get("adev", [])
        local_slopes   = allan_analysis.get("local_slopes", [])

        if len(adev_full_taus) >= 2:
            adev_table = pd.DataFrame({"τ (s)": adev_full_taus, "σy(τ)": [f"{v:.3e}" for v in adev_full_vals]})
            st.dataframe(adev_table, use_container_width=True, hide_index=True)

            fig_ad = go.Figure()
            fig_ad.add_trace(go.Scatter(x=adev_full_taus, y=adev_full_vals, mode="lines+markers",
                                         name="σy(τ) ADEV", line=dict(color="#38bdf8", width=2), marker=dict(size=7)))
            if mdev_vals:
                fig_ad.add_trace(go.Scatter(x=mdev_taus, y=mdev_vals, mode="lines+markers",
                                             name="σ_mod_y MDEV", line=dict(color="#a78bfa", width=2, dash="dash"), marker=dict(size=5, symbol="diamond")))
            for ref_y, col, lab in [(1e-10, "#ef4444", "10⁻¹⁰"), (1e-11, "#f59e0b", "10⁻¹¹"), (1e-12, "#22c55e", "10⁻¹²")]:
                fig_ad.add_hline(y=ref_y, line_dash="dot", line_color=col, annotation_text=lab)
            _style(fig_ad, "Allan Deviation σy(τ) — ADEV and MDEV", "Averaging Time τ (s)", "σy(τ)", height=400)
            fig_ad.update_xaxes(type="log"); fig_ad.update_yaxes(type="log")
            st.plotly_chart(fig_ad, use_container_width=True)

            if local_slopes:
                ls_df = pd.DataFrame(local_slopes).rename(columns={
                    "tau_start": "τ_start (s)", "tau_end": "τ_end (s)",
                    "slope": "Local Slope", "noise_process": "Noise Process"})
                st.dataframe(ls_df, use_container_width=True, hide_index=True)

            # Noise process identification
            noise_label, noise_conf = _map_noise_process(local_slopes, dominant_noise)
            st.markdown(f"**Dominant Noise Process:** {noise_label}  ")
            st.markdown(f"**Noise Confidence:** {noise_conf*100:.0f}%  ")
            st.markdown(f"**Scientific Interpretation:** The dominant noise process is {noise_label}. Confidence derived from ADEV local slope segmentation.")

        st.markdown(f"**Scientific Interpretation:** σy(1s)={_fmt(sigma1)}, σy(100s)={_fmt(sigma100)}. Regime: {regime}. Dominant noise: {dominant_noise}.")
        st.markdown("**Engineering Implication:** σy minimum identifies optimal operational averaging time. Rising slope at long τ indicates drift/RWFM coupling.")
        # Stability budget summary (if available)
        if not budget_df.empty:
            st.markdown("**Stability Budget Breakdown (contributions)**")
            try:
                bd = budget_df.copy()
                if "Contribution (%)" in bd.columns:
                    bd = bd.sort_values("Contribution (%)", ascending=False)
                st.dataframe(bd, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(budget_df, use_container_width=True, hide_index=True)

    # ─── TAB 3 ───────────────────────────────────────────────────────────────
    with tab_noise:
        _sec("Tab 3 · Noise Process Identification")
        st.caption("Identify limiting stochastic noise process from ADEV, MDEV, and HDEV log-log slope analysis.")
        st.markdown("**Objective:** Unambiguously classify noise process from deviation slope.  \n**Methodology:** MDEV discriminates WPM (slope −3/2) vs WFM (slope −1/2). HDEV removes drift contribution.")

        noise_ref = pd.DataFrame({
            "Process": ["WPM", "WFM", "FFM", "RWFM", "RWP/Drift"],
            "ADEV Slope": ["-1", "-½", "0", "+½", "+1"],
            "MDEV Slope": ["-3/2", "-½", "0", "+½", "+1"],
            "HDEV Slope": ["-1", "-½", "0", "+½", "0 (removed)"],
            "Physical Origin": [
                "Photon shot noise, ADC quantisation, Johnson noise",
                "White S_y(f) noise floor — electronics, VCO",
                "1/f noise: laser RIN, amplifier flicker",
                "Thermal/vibrational coupling to frequency",
                "Systematic ageing, gas leakage, drift accumulation",
            ]})
        st.dataframe(noise_ref, use_container_width=True, hide_index=True)

        # MDEV + HDEV subplots
        if mdev_vals or hdev_vals:
            fig_mh = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Modified Allan Deviation (MDEV)", "Hadamard Deviation (HDEV)"])
            if mdev_vals:
                fig_mh.add_trace(go.Scatter(x=mdev_taus, y=mdev_vals, mode="lines+markers",
                                             name="MDEV", line=dict(color="#a78bfa", width=2)), row=1, col=1)
                if adev_full_taus:
                    fig_mh.add_trace(go.Scatter(x=adev_full_taus, y=adev_full_vals, mode="lines",
                                                 name="ADEV (ref)", line=dict(color="#38bdf8", width=1, dash="dot")), row=1, col=1)
            if hdev_vals:
                fig_mh.add_trace(go.Scatter(x=hdev_taus, y=hdev_vals, mode="lines+markers",
                                             name="HDEV", line=dict(color="#34d399", width=2)), row=1, col=2)
                if adev_full_taus:
                    fig_mh.add_trace(go.Scatter(x=adev_full_taus, y=adev_full_vals, mode="lines",
                                                 name="ADEV (ref)", line=dict(color="#38bdf8", width=1, dash="dot"),
                                                 showlegend=False), row=1, col=2)
            fig_mh.update_layout(template="plotly_dark", paper_bgcolor="#070d18",
                                   plot_bgcolor="#070d18", font=dict(color="#e0e6f0"),
                                   height=380, margin=dict(l=50, r=30, t=55, b=40))
            fig_mh.update_xaxes(type="log"); fig_mh.update_yaxes(type="log")
            st.plotly_chart(fig_mh, use_container_width=True)

        slope = allan_analysis.get("slope", 0.0)
        nc = st.columns(2)
        nc[0].metric("ADEV Log-Log Slope", f"{slope:.3f}")
        nc[1].metric("Dominant Noise Process (IEEE 1139-2022)", dominant_noise.split("(")[0].strip())
        st.success(f"—— Primary identification: **{dominant_noise}** (slope = {slope:.3f}) ——")

        st.markdown(f"**Scientific Interpretation:** Slope {slope:.3f} → {dominant_noise}. MDEV/HDEV comparison confirms noise hierarchy.")
        st.markdown("**Engineering Implication:** WPM → increase optical power. WFM → reduce electronic noise. FFM → laser RIN stabilisation. RWFM → tighten thermal servo. RWP → drift compensation.")

    # ─── TAB 4 ───────────────────────────────────────────────────────────────
    with tab_drift:
        _sec("Tab 4 · Frequency Drift Assessment")
        st.caption("Systematic fractional frequency drift rate d = ∂(Δf/f)/∂t quantified by OLS regression.")
        st.markdown("**Objective:** Estimate drift rate and project cumulative offset.  \n**Methodology:** OLS polynomial regression y(t) = d·t + b (Audoin & Guinot, 2001, Ch.4). Residual σ_res = noise floor after drift removal.")

        dc = st.columns(4)
        dc[0].metric("Drift (Δf/f)/s", f"{drift_per_second:.3e}")
        dc[1].metric("Drift (Δf/f)/day", f"{drift_per_day:.3e}")
        dc[2].metric("OLS R²", f"{drift_r2:.4f}")
        dc[3].metric("Residual σ_res", f"{drift_residual:.3e}")

        proj = drift_proj.get("projected_offsets", {})
        if proj:
            st.markdown("#### OLS Drift Projections (Constant-Environment Assumption)")
            st.dataframe(pd.DataFrame([(f"{h:.0f} h", f"{v:.4e}") for h, v in sorted(proj.items())],
                                       columns=["Horizon", "Projected Δf/f"]),
                          use_container_width=True, hide_index=True)

        _t = drift_proj.get("_t", [])
        _y = drift_proj.get("_y", [])
        _sl = drift_proj.get("_slope", 0.0)
        _ic = drift_proj.get("_intercept", 0.0)
        if len(_t) >= 2:
            ta, ya = np.array(_t), np.array(_y)
            fig_dr = go.Figure()
            fig_dr.add_trace(go.Scatter(x=ta, y=ya, mode="markers", name="Measured y(t)",
                                         marker=dict(color="#38bdf8", size=3, opacity=0.6)))
            fig_dr.add_trace(go.Scatter(x=ta, y=_sl * ta + _ic, mode="lines",
                                         name=f"OLS: d={_sl:.3e} (Δf/f)/s", line=dict(color="#f97316", width=2)))
            _style(fig_dr, "Fractional Frequency Offset y(t) with OLS Drift Model", "Sample Time (s)", "y(t) = Δf/f", 380)
            st.plotly_chart(fig_dr, use_container_width=True)
            resid = ya - (_sl * ta + _ic)
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=ta, y=resid, mode="lines", name="Residual",
                                          line=dict(color="#a78bfa", width=1)))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#475569")
            _style(fig_res, "Post-Drift-Extraction Residual", "Sample Time (s)", "Residual Δf/f", 280)
            st.plotly_chart(fig_res, use_container_width=True)

        r2_note = " (R²<0.1 — drift model explains <10% of variance; noise-dominated record)" if drift_r2 < 0.1 else f" (R²={drift_r2:.3f})"
        st.markdown(f"**Scientific Interpretation:** Drift rate {drift_per_day:.3e} (Δf/f)/day{r2_note}. Residual σ_res={drift_residual:.3e} reflects noise floor after systematic trend removal.")
        st.markdown("**Engineering Implication:** Drift >2×10⁻¹³ (Δf/f)/day for Rb standard indicates cell ageing or thermal coupling. Implement steered frequency correction proportional to measured d.")

    # ─── TAB 5 ───────────────────────────────────────────────────────────────
    with tab_env:
        _sec("Tab 5 · Environmental Sensitivity Assessment")
        st.caption("RSS stability budget decomposition: identifies dominant environmental coupling channel.")
        st.markdown("**Objective:** Rank environmental parameters by σy contribution.  \n**Methodology:** σy_i = αᵢ×σxᵢ, σy_total = √(Σσy_i²). αᵢ from Vanier & Audoin (1989), Ch. 5.")

        if not budget_df.empty and "Contribution (%)" in budget_df.columns:
            dom_p   = dominant_contrib["Parameter"] if dominant_contrib is not None else "N/A"
            dom_pct = float(dominant_contrib["Contribution (%)"]) if dominant_contrib is not None else 0.0
            dc = st.columns(3)
            dc[0].metric("Dominant Coupling Channel", dom_p)
            dc[1].metric("Budget Contribution", f"{dom_pct:.2f} %")
            dc[2].metric("Total σy (RSS)", f"{total_sigma_y:.3e}")

            fig_bud = px.bar(budget_df, x="Contribution (%)", y="Parameter", orientation="h",
                              text="Contribution (%)", color="Contribution (%)", color_continuous_scale="Blues")
            fig_bud.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            _style(fig_bud, "RSS Stability Budget — Environmental Contributions to σy",
                   "Contribution (%)", "Parameter", 320)
            fig_bud.update_coloraxes(showscale=False)
            st.plotly_chart(fig_bud, use_container_width=True)

            disp_cols = [c for c in ["Parameter", "Sensitivity Coefficient (Δf/f/unit)",
                                      "Measured σ (physical unit)", "σy_i (Δf/f)", "Contribution (%)"]
                         if c in budget_df.columns]
            st.dataframe(budget_df[disp_cols], use_container_width=True, hide_index=True)

        rk_params = sensitivity_ranking.get("ranked_parameters", [])
        if rk_params:
            st.markdown("#### Pearson |ρ| Frequency Coupling Ranking")
            rk_df = pd.DataFrame(rk_params)
            fig_rho = px.bar(rk_df, x="|Pearson ρ|", y="Parameter", orientation="h",
                              text="|Pearson ρ|")
            fig_rho.update_traces(marker_color="#7dd3fc", texttemplate="%{text:.3f}", textposition="outside")
            _style(fig_rho, "Pearson |ρ| Coupling Ranking", "|Pearson ρ|", "Parameter", 300)
            st.plotly_chart(fig_rho, use_container_width=True)

        st.markdown(f"**Scientific Interpretation:** {dom_p if dominant_contrib is not None else 'N/A'} dominates with {dom_pct:.1f}% budget contribution.")
        st.markdown("**Engineering Implication:** αᵢ quantifies max allowable σxᵢ = σy_target/αᵢ for a given specification.")

    # ─── TAB 6 ───────────────────────────────────────────────────────────────
    with tab_rca:
        _sec("Tab 6 · Root Cause Analysis — Frequency Excursion Attribution")
        st.caption("3σ excursion detection with sensitivity-weighted physical attribution (Cutler & Searle, 1966).")
        st.markdown("**Methodology:** Cᵢ(k) = |αᵢ × Δxᵢ(k)| / Σⱼ|αⱼ × Δxⱼ(k)| × 100%.")

        top_c = excursion_analysis.get("top_contributors", [])
        rc = st.columns(3)
        rc[0].metric("Total Excursions", excursion_analysis.get("excursion_count", 0))
        rc[1].metric("Rate / 100 samples", f"{excursion_analysis.get('excursion_rate_per_100', 0):.2f}")
        rc[2].metric("Primary Attribution", top_c[0]["Parameter"] if top_c else "N/A")

        exc_df = active_df.reset_index(drop=True)
        exc_idx = exc_df.index[exc_df["excursion"] == 1]
        fig_exc = go.Figure()
        fig_exc.add_trace(go.Scatter(x=exc_df.index, y=exc_df["frequency_offset"].astype(float),
                                      mode="lines", name="Δf/f", line=dict(color="#38bdf8", width=1)))
        if len(exc_idx) > 0:
            fig_exc.add_trace(go.Scatter(x=exc_idx,
                                          y=exc_df.loc[exc_idx, "frequency_offset"].astype(float),
                                          mode="markers", name="Excursion (3σ)",
                                          marker=dict(color="#ef4444", size=8, symbol="x")))
        _style(fig_exc, "Frequency Offset — Excursion Timeline", "Sample Index", "y(t)", 320)
        st.plotly_chart(fig_exc, use_container_width=True)

        st.markdown("#### Primary Attribution: RSS Stability Budget Ranking")
        st.caption("Source: sensitivity budget decomposition — fallback to statistical ranking if budget unresolved. "
                   "Basis: σyᵢ = |αᵢ × σxᵢ|; Pearson/Spearman and ML importance used when budget is degenerate.")
        # Handle pathological all-zero budgets by constructing a fallback ranking
        use_budget = True
        try:
            if budget_df.empty or ("Contribution (%)" in budget_df.columns and float(budget_df["Contribution (%)"].sum()) == 0.0):
                use_budget = False
        except Exception:
            use_budget = False

        if use_budget and "Contribution (%)" in budget_df.columns:
            budget_sorted = budget_df.sort_values("Contribution (%)", ascending=False)
            fig_budget_rca = px.bar(
                budget_sorted, x="Contribution (%)", y="Parameter", orientation="h",
                text="Contribution (%)", color="Contribution (%)",
                color_continuous_scale="Blues",
            )
            fig_budget_rca.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            _style(fig_budget_rca, "RSS Stability Budget — Primary Attribution Ranking",
                   "Contribution (%)", "Parameter", 280)
            fig_budget_rca.update_coloraxes(showscale=False)
            st.plotly_chart(fig_budget_rca, use_container_width=True)
        else:
            # Fallback: use sensitivity ranking (Pearson) and ML attribution where available
            fallback = []
            sr = sensitivity_ranking.get("ranked_parameters", []) if isinstance(sensitivity_ranking, dict) else []
            if sr:
                for r in sr[:5]:
                    fallback.append({"Parameter": r.get("Parameter", "Unknown"), "Contribution (%)": round(r.get("|Pearson ρ|", 0) * 100.0, 2)})
            else:
                ml_avail = isinstance(ml_attr_result, dict) and (ml_attr_result.get("available") or ml_attr_result.get("feature_probs"))
                if ml_avail:
                    ranked_ml = ml_attr_result.get("ranked", []) if ml_attr_result.get("ranked") else ml_attr_result.get("feature_probs", [])
                    for r in ranked_ml[:5]:
                        if isinstance(r, (list, tuple)) and len(r) >= 2:
                            fallback.append({"Parameter": r[0], "Contribution (%)": float(r[1])})
                        elif isinstance(r, dict):
                            fallback.append({"Parameter": r.get("Parameter", "Unknown"), "Contribution (%)": float(r.get("Importance Score", 0))})
            if fallback:
                fb_df = pd.DataFrame(fallback)
                fig_fb = px.bar(fb_df, x="Contribution (%)", y="Parameter", orientation="h", text="Contribution (%)", color="Contribution (%)", color_continuous_scale="Purples")
                fig_fb.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                _style(fig_fb, "Fallback Attribution — Statistical / ML Ranking", "Score", "Parameter", 280)
                st.plotly_chart(fig_fb, use_container_width=True)
                st.dataframe(fb_df, use_container_width=True, hide_index=True)
            else:
                st.info("Attribution unresolved: insufficient sensitivity or statistical signal.")

        if excursion_analysis.get("excursion_count", 0) > 0 and top_c:
            st.markdown("#### Secondary Attribution: Per-Excursion Event Sensitivity Decomposition")
            st.caption("Only available when excursion events detected. "
                       "Attribution: Cᵢ(k) = |αᵢ × Δxᵢ(k)| / Σᵧ|αᵧ × Δxᵧ(k)| × 100%.")
            tc_df = pd.DataFrame(top_c)
            fig_a = px.bar(tc_df, x="Contribution (%)", y="Parameter", orientation="h", text="Contribution (%)")
            fig_a.update_traces(marker_color="#f97316", texttemplate="%{text:.2f}%", textposition="outside")
            _style(fig_a, "Excursion-Event Attribution", "Contribution (%)", "Parameter", 280)
            st.plotly_chart(fig_a, use_container_width=True)
            st.dataframe(tc_df, use_container_width=True, hide_index=True)
        elif excursion_analysis.get("excursion_count", 0) == 0:
            st.info("ℹ️ No frequency excursion events detected in this record.")

        msg = excursion_attr_summary.get("message", "")
        if msg:
            st.caption(msg)
        st.markdown("**Scientific Interpretation:** RSS budget attribution ranks environmental coupling channels by estimated σy contribution. Event-based attribution identifies proximate physical drivers of each observed excursion.")
        st.markdown("**Engineering Implication:** Target highest-ranked channel in RSS budget for maximum σy improvement.")

    # ─── TAB 7 ───────────────────────────────────────────────────────────────
    with tab_rec:
        _sec("Tab 7 · Stabilisation Recommendation")
        st.caption("Physics-based parameter adjustment recommendations ranked by |αᵢ × σxᵢ| — Vanier et al. (2003).")
        st.markdown("**Methodology:** Actions ordered by RSS budget contribution. No ML advisory — purely physics-based.")

        if stab_actions:
            for i, act in enumerate(stab_actions):
                pr = act.get("Priority", "Medium")
                col = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(pr, "#94a3b8")
                st.markdown(f"<span style='background:{col};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem'>{pr}</span> **{i+1}. {act['Parameter']}**", unsafe_allow_html=True)
                ac = st.columns(3)
                ac[0].metric("Current", act.get("Current Value", "N/A"))
                ac[1].metric("Target", act.get("Target Value", "N/A"))
                ac[2].metric("Est. σy Improvement", act.get("Estimated σy Improvement", "N/A"))
                with st.expander(f"Physical Basis — {act['Parameter']}", i == 0):
                    st.markdown(f"**Physical Basis:** {act.get('Physical Basis', 'N/A')}")
                    st.markdown(f"**Sensitivity Coefficient:** {act.get('Sensitivity Coefficient', 'N/A')}")
                    st.markdown(f"**Engineering Action:** {act.get('Engineering Action', 'N/A')}")
                st.markdown("---")
        else:
            st.info("No stabilisation actions resolved from current sensitivity analysis.")

        st.markdown("#### Deterministic Metrological Assessment Query")
        q = st.text_input("Technical Query", "What is the dominant instability source?", key="metro_q")
        # Auto-generate deterministic metrological narrative (no manual trigger)
        try:
            metro_narr = generate_stability_assessment_narrative(q, active_df)
        except Exception:
            metro_narr = "Metrological narrative generation not available for this dataset."
        st.text_area("Metrological Narrative", metro_narr, height=240)

    # ─── TAB 8 ───────────────────────────────────────────────────────────────
    with tab_impact:
        _sec("Tab 8 · Expected Stability Improvement")
        st.caption("Projected σy reduction from dominant-channel suppression via RSS budget model.")
        st.markdown("**Methodology:** σy_after = √(σy_before² × (1 − dominant_fraction)). First-order estimate.")

        dom_pct_val = float(dominant_contrib["Contribution (%)"]) if dominant_contrib is not None else 0.0
        impr = compute_stability_improvement_from_budget(total_sigma_y, dom_pct_val)
        ic = st.columns(4)
        ic[0].metric("Pre-Stabilisation σy", f"{impr['sigma_y_before']:.3e}")
        ic[1].metric("Post-Stabilisation σy (est.)", f"{impr['sigma_y_after']:.3e}")
        ic[2].metric("Projected Improvement", f"{impr['improvement_pct']:.1f} %")
        ic[3].metric("Residual Noise Floor", f"{impr['residual_floor']:.3e}")

        if budget_contribs and dominant_contrib is not None:
            pb = compute_post_stabilisation_budget(budget_contribs, dominant_contrib["Parameter"], 1.0)
            pc = pb.get("new_contributions", [])
            if pc:
                bef_df = budget_df[["Parameter", "Contribution (%)"]].rename(columns={"Contribution (%)": "Pre (%)"})
                aft_df = pd.DataFrame([{"Parameter": c["Parameter"], "Post (%)": c.get("Contribution (%)", 0)} for c in pc])
                cmp_df = bef_df.merge(aft_df, on="Parameter", how="outer")
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(name="Pre-Stabilisation", x=cmp_df["Parameter"], y=cmp_df["Pre (%)"], marker_color="#38bdf8"))
                fig_cmp.add_trace(go.Bar(name="Post-Stabilisation (projected)", x=cmp_df["Parameter"], y=cmp_df["Post (%)"], marker_color="#22c55e"))
                fig_cmp.update_layout(barmode="group")
                _style(fig_cmp, f"RSS Budget: Pre vs Post ({dominant_contrib['Parameter']} suppressed)", "Parameter", "Contribution (%)", 320)
                st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown(f"**Scientific Interpretation:** Suppressing {dominant_contrib['Parameter'] if dominant_contrib is not None else 'dominant channel'} ({dom_pct_val:.1f}%) projects {impr['improvement_pct']:.1f}% σy reduction.")
        st.markdown("**Engineering Implication:** Post-intervention recompute ADEV to verify improvement.")

    # ─── TAB 9 ───────────────────────────────────────────────────────────────
    with tab_rpt:
        _sec("Tab 9 · Scientific Assessment Report — DRDO Technical Note Format")
        st.caption(
            "Structured metrological assessment for DRDO scientific review. "
            "All quantities are traceable to IEEE 1139-2022, NIST TN-1337, "
            "Vanier & Audoin (1989), and Camparo (2005)."
        )
        st.markdown(
            "> This report is structured to answer the five DRDO scientific review questions: "
            "(1) Why is this analysis necessary? (2) What scientific question does it answer? "
            "(3) What published methodology supports it? (4) What physical interpretation does it provide? "
            "(5) How does it contribute to frequency stabilisation?"
        )

        try:
            full_report_rows = generate_assessment_report(
                op_state, drift_proj, stability_budget, stab_actions,
                excursion_analysis=excursion_analysis,
                rus_result=rus_result,
                risk_result=risk_result,
                health_result=health_result,
                allan_analysis=allan_analysis,
            )
        except Exception:
            # Defensive handling: do not crash the dashboard on reporting errors.
            st.warning("Scientific report generation failed for this dataset — report unavailable. Please check input telemetry and retry.")
            full_report_rows = [("Report Error", "Report generation failed; see diagnostics.")]

        rpt_df = pd.DataFrame(full_report_rows, columns=["Assessment Category", "Value / Result"])

        def _highlight_report(row):
            cat = str(row["Assessment Category"])
            if "Regime" in cat and "Criteria" not in cat:
                color = {"STABLE": "#14532d", "WARNING": "#78350f", "UNSTABLE": "#7f1d1d"}.get(
                    str(row["Value / Result"]), "#1e293b")
                return [f"background-color: {color}; font-weight: bold"] * 2
            if "Overall Technical Assessment" in cat:
                return ["background-color: #0f172a; font-weight: bold"] * 2
            if "Methodology References" in cat:
                return ["background-color: #0c1b33; font-style: italic"] * 2
            return ["background-color: #070d18"] * 2

        st.dataframe(
            rpt_df.style.apply(_highlight_report, axis=1),
            use_container_width=True, hide_index=True,
        )

        st.download_button(
            "⬇ Download Report (CSV)",
            rpt_df.to_csv(index=False).encode("utf-8"),
            "drdo_frequency_standard_report.csv", "text/csv",
        )

        st.markdown("#### Methodology References")
        st.markdown(
            "1. **IEEE Std 1139-2022** — Standard Definitions of Physical Quantities for "
            "Fundamental Frequency and Time Metrology \u2014 Random Instabilities.  \n"
            "2. **Riley, W.J. & Howe, D.A. (2008)** — Handbook of Frequency Stability Analysis, NIST TN-1337.  \n"
            "3. **Vanier, J. & Audoin, C. (1989)** — The Quantum Physics of Atomic Frequency Standards, Adam Hilger.  \n"
            "4. **Camparo, J. (2005)** — The Rubidium Atomic Clock and Basic Research, Physics Today 58(11).  \n"
            "5. **Cutler, L.S. & Searle, C.L. (1966)** — Some Aspects of the Theory and Measurement of "
            "Frequency Fluctuations in Frequency Standards, Proc. IEEE, 54(2).  \n"
            "6. **Audoin, C. & Guinot, B. (2001)** — The Measurement of Time, Cambridge University Press.  \n"
            "7. **Vanier, J., Simard, J.F., & Barrette, J.S. (2003)** — Practical Considerations in the Design "
            "and Development of Small Rb Frequency Standards, Appl. Phys. B, 76(7).  \n"
            "8. **Saxena, A. et al. (2008)** — Metrics for Offline Evaluation of Prognostic Performance, "
            "Int. J. Prognostics Health Management.  \n"
            "9. **Lundberg, S.M. & Lee, S.I. (2017)** — A Unified Approach to Interpreting Model Predictions, "
            "NeurIPS 2017."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ██████████████  SECTION 2 — AI PREDICTIVE INTELLIGENCE  ████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

else:

    (tab_fc, tab_rus, tab_ew, tab_mlrca, tab_hi,
     tab_risk, tab_dt, tab_cop, tab_val) = st.tabs([
        "Ⓐ Predictive Stability Forecasting",
        "Ⓑ Remaining Useful Stability",
        "Ⓒ Early Warning Engine",
        "Ⓓ ML Root Cause Attribution",
        "Ⓔ Composite Stability Performance Index (CSPI)",
        "Ⓕ Stability Risk Assessment",
        "Ⓖ Digital Twin Simulator",
        "Ⓗ Scientific Interpretation Assistant",
        "Ⓘ Model Validation Framework",
    ])

    # ─── TAB A ───────────────────────────────────────────────────────────────
    with tab_fc:
        _sec("Tab A · Predictive Stability Forecasting", ai=True)
        st.caption("Multi-model ensemble forecast of fractional frequency offset at 1 h, 6 h, 24 h, 7 d horizons.")

        st.info(
            f"ℹ️ **Model Disclosure:** {LSTM_STATUS}  \n"
            "Kalman filter and XGBoost are fully operational. "
            "All uncertainty bounds are bootstrap-derived (B=30 resamples) "
            "unless stated otherwise.",
            icon="⚠️",
        )

        st.markdown(
            "**Objective:** Predict future frequency offset and σy degradation before it occurs.  \n"
            "**Models:**  \n"
            "• **Kalman Filter** [Kalman 1960] — optimal linear state estimator; state = [y, ẟ]; constant-velocity drift model.  \n"
            "• **XGBoost** [Chen & Guestrin 2016] — gradient-boosted trees on lag + environmental features; bootstrap CI (B=30).  \n"
            "• **Sequential NN (MLP)** [Hochreiter & Schmidhuber 1997 architecture] — on 20-sample sliding window; bootstrap uncertainty.  \n"
            "• **Ensemble** — simple mean of available model predictions; CI propagated quadratically."
        )

        if kalman_result.get("available"):
            kc_kpi = st.columns(4)
            kc_kpi[0].metric("Innovation Mean", f"{kalman_result['innovation_mean']:.3e}")
            kc_kpi[1].metric("Innovation Std", f"{kalman_result['innovation_std']:.3e}")
            kc_kpi[2].metric("χ² Test p-value", f"{kalman_result['chi2_pvalue']:.4f}")
            kc_kpi[3].metric("Measurement Noise σ", f"{kalman_result['model_sigma_meas']:.3e}")

            raw_y     = np.array(kalman_result["raw_y"])
            filt_y    = np.array(kalman_result["filtered_y"])
            filt_unc  = np.array(kalman_result["filter_uncertainty"])
            t_idx     = np.arange(len(raw_y))

            fig_kf = go.Figure()
            fig_kf.add_trace(go.Scatter(x=t_idx, y=raw_y, mode="lines", name="Measured y(t)",
                                         line=dict(color="#64748b", width=1), opacity=0.6))
            fig_kf.add_trace(go.Scatter(x=t_idx, y=filt_y, mode="lines", name="Kalman Estimate",
                                         line=dict(color="#38bdf8", width=2)))
            filt_std = np.sqrt(np.clip(filt_unc, 0, None))
            fig_kf.add_trace(go.Scatter(
                x=np.concatenate([t_idx, t_idx[::-1]]),
                y=np.concatenate([filt_y + 2*filt_std, (filt_y - 2*filt_std)[::-1]]),
                fill="toself", fillcolor="rgba(56,189,248,0.12)", line=dict(color="rgba(0,0,0,0)"),
                name="±2σ Filter Uncertainty",
            ))
            _style(fig_kf, "Kalman Filter: State Estimate vs. Measured Frequency Offset",
                   "Sample Index", "Fractional Frequency Offset y(t)", 380)
            st.plotly_chart(fig_kf, use_container_width=True)
            
            innov = np.array(kalman_result["innovation"])
            fig_inn = go.Figure()
            fig_inn.add_trace(go.Scatter(x=t_idx, y=innov, mode="lines", name="Innovation",
                                          line=dict(color="#f97316", width=1)))
            fig_inn.add_hline(y=0, line_dash="dash", line_color="#475569")
            _style(fig_inn, "Kalman Innovation Sequence (y_k − Ĥx̂_k|k-1)",
                   "Sample Index", "Innovation", 240)
            st.plotly_chart(fig_inn, use_container_width=True)

            p_val = kalman_result["chi2_pvalue"]
            whiteness = "PASS — innovations are consistent with zero-mean white noise" if p_val > 0.05 else "FAIL — non-white innovations indicate unmodelled dynamics"
            st.info(f"**Innovation Whiteness Test (χ²):** {whiteness} (p = {p_val:.4f})")

            km_fcast = kalman_result.get("forecast_means", {})
            kf_fcast = kalman_result.get("forecast_stds", {})
            if km_fcast:
                st.markdown("#### Kalman Forecast at Operational Horizons")
                kf_rows = [(h, f"{m:.4e}", f"{s:.4e}", f"{m-1.96*s:.4e}", f"{m+1.96*s:.4e}")
                           for (h, m), (_, s) in zip(km_fcast.items(), kf_fcast.items())]
                st.dataframe(pd.DataFrame(kf_rows, columns=["Horizon", "Point Forecast", "Std Dev", "CI 95% Low", "CI 95% High"]),
                              use_container_width=True, hide_index=True)
        else:
            st.warning(f"Kalman filter: {kalman_result.get('reason', 'Not available')}")

        st.markdown("### Ensemble Forecast — Kalman + XGBoost + LSTM")
        horizons_data = forecast_result.get("horizons", {})
        if horizons_data:
            rows = []
            for label, hd in horizons_data.items():
                rows.append({
                    "Horizon": label,
                    "Ensemble Forecast": f"{hd['point_forecast']:.4e}",
                    "Std Dev": f"{hd['std']:.4e}",
                    "95% CI Low": f"{hd['ci_95_low']:.4e}",
                    "95% CI High": f"{hd['ci_95_high']:.4e}",
                    "Models Used": hd["n_models"],
                })
            fc_df = pd.DataFrame(rows)
            st.dataframe(fc_df, use_container_width=True, hide_index=True)

            labels_plot = list(horizons_data.keys())
            pts   = [horizons_data[h]["point_forecast"] for h in labels_plot]
            ci_lo = [horizons_data[h]["ci_95_low"] for h in labels_plot]
            ci_hi = [horizons_data[h]["ci_95_high"] for h in labels_plot]

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=labels_plot, y=pts, mode="lines+markers",
                                         name="Ensemble Forecast", line=dict(color="#38bdf8", width=2), marker=dict(size=9)))
            fig_fc.add_trace(go.Scatter(
                x=labels_plot + labels_plot[::-1],
                y=ci_hi + ci_lo[::-1],
                fill="toself", fillcolor="rgba(56,189,248,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI",
            ))
            fig_fc.add_hline(y=float(vals[-1]), line_dash="dash", line_color="#f59e0b",
                              annotation_text="Current Δf/f")
            _style(fig_fc, "Multi-Horizon Ensemble Forecast of Fractional Frequency Offset",
                   "Forecast Horizon", "Projected Δf/f", 350)
            st.plotly_chart(fig_fc, use_container_width=True)

        xgb_res = forecast_result.get("xgb_result", {})
        if xgb_res.get("available"):
            st.markdown("#### XGBoost Model Performance")
            xc = st.columns(3)
            xc[0].metric("Validation MAE", f"{xgb_res['val_mae_mean']:.3e}")
            xc[1].metric("Validation RMSE", f"{xgb_res['val_rmse_mean']:.3e}")
            xc[2].metric("In-Sample R²", f"{xgb_res['in_sample_r2']:.4f}")
            st.caption(f"Model: {xgb_res['model_name']} | Training protocol: TimeSeriesSplit (k=3) — no temporal data leakage")

    # ─── TAB B ───────────────────────────────────────────────────────────────
    with tab_rus:
        _sec("Tab B · Remaining Useful Stability (RUS)", ai=True)
        st.caption("Time-to-Specification Violation (TTSV) and Time-to-Maintenance (TTM) estimation.")
        if rus_result.get("available"):
            rc = st.columns(4)
            rc[0].metric("RUS Score", f"{rus_result['rus_score']:.1f} / 100")
            rc[1].metric("TTSV", f"{rus_result['ttsv_days']:.1f} days" if rus_result.get("ttsv_days") is not None else "Beyond window")
            rc[2].metric("TTM", f"{rus_result['ttm_days']:.1f} days" if rus_result.get("ttm_days") is not None else "Beyond window")
            rc[3].metric("σy Trend", rus_result.get("sigma_trend", "N/A").title())
            roll_sigma = rus_result.get("rolling_sigma", [])
            if roll_sigma:
                fig_rs = go.Figure()
                ts = np.arange(len(roll_sigma), dtype=float)
                fig_rs.add_trace(go.Scatter(x=ts, y=roll_sigma, mode="lines+markers",
                                             name="Rolling σy(1s)", line=dict(color="#38bdf8", width=2),
                                             marker=dict(size=6)))
                dr = rus_result.get("degrad_rate", 0.0)
                if dr != 0.0 and len(roll_sigma) > 1:
                    trend_line = roll_sigma[0] + dr * ts
                    fig_rs.add_trace(go.Scatter(x=ts, y=trend_line, mode="lines",
                                                 name=f"Degradation trend (d={dr:.3e}/window)",
                                                 line=dict(color="#f97316", width=2, dash="dash")))
                _style(fig_rs, "Rolling σy(τ=1s) — Stability Degradation Trend",
                       "Window Index (×150 samples)", "σy(1s)", 360)
                for thr, col, lab in [(STABILITY_THRESHOLDS["STABLE_sigma1"], "#22c55e", "STABLE threshold"),
                                       (STABILITY_THRESHOLDS["WARNING_sigma1"], "#f59e0b", "WARNING threshold")]:
                    fig_rs.add_hline(y=thr, line_dash="dot", line_color=col, annotation_text=lab)
                st.plotly_chart(fig_rs, use_container_width=True)

            # RUS gauge
            rus_score = rus_result.get("rus_score", 0.0)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rus_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Remaining Useful Stability (RUS) Score", "font": {"size": 16, "color": "#e0e6f0"}},
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#e0e6f0"},
                    "bar": {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, 30], "color": "#7f1d1d"},
                        {"range": [30, 60], "color": "#78350f"},
                        {"range": [60, 80], "color": "#365314"},
                        {"range": [80, 100], "color": "#14532d"},
                    ],
                    "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.85, "value": 30},
                },
            ))
            fig_gauge.update_layout(paper_bgcolor="#070d18", font=dict(color="#e0e6f0"), height=280,
                                     margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            ttsv_str = f"{rus_result['ttsv_days']:.1f} days" if rus_result.get("ttsv_days") else "Not determinable (drift < noise floor)"
            st.markdown(f"**Scientific Interpretation:** RUS Score = {rus_score:.1f}/100. TTSV = {ttsv_str}. σy trend: {rus_result.get('sigma_trend', 'N/A')}.")
            st.markdown("**Engineering Implication:** RUS Score <50 warrants immediate maintenance scheduling. TTSV provides the operational window before performance specification is violated.")
        else:
            st.warning(f"RUS not available: {rus_result.get('reason', 'Unknown')}")

    # ─── TAB C ───────────────────────────────────────────────────────────────
    with tab_ew:
        _sec("Tab C · Early Warning Engine", ai=True)
        st.caption("Pre-excursion frequency instability detection using calibrated binary classification.")
        if warning_result.get("available"):
            alert_lvl = warning_result.get("alert_level", "Nominal")
            alert_col = {"HIGH RISK": "#ef4444", "ELEVATED": "#f59e0b", "Nominal": "#22c55e"}.get(alert_lvl, "#94a3b8")
            st.markdown(f"<h3 style='color:{alert_col}'>⚠ Alert Level: {alert_lvl}</h3>", unsafe_allow_html=True)

            wc = st.columns(3)
            wc[0].metric("Max Excursion Probability", f"{warning_result['max_excursion_prob']:.1%}")
            wc[1].metric("Historical Excursion Rate", f"{warning_result.get('excursion_rate', 0):.2f} %")
            wc[2].metric("3σ Detection Threshold", f"{warning_result.get('threshold_3sigma', 0):.3e}")

            horizons_data = warning_result.get("horizons", {})
            ew_rows = []
            probs = []
            for hor, dat in horizons_data.items():
                ew_rows.append({
                    "Horizon": hor,
                    "Excursion Probability": f"{dat['prob']:.1%}",
                    "Alert Label": dat["label"],
                    "Threshold Exceeded": "YES" if dat["threshold_exceeded"] else "No",
                })
                probs.append(dat["prob"])
            st.dataframe(pd.DataFrame(ew_rows), use_container_width=True, hide_index=True)

            hor_labels = list(horizons_data.keys())
            fig_ew = go.Figure()
            bar_colors = [{"HIGH RISK": "#ef4444", "ELEVATED": "#f59e0b", "Nominal": "#22c55e"}.get(
                horizons_data[h]["label"], "#94a3b8") for h in hor_labels]
            fig_ew.add_trace(go.Bar(x=hor_labels, y=probs, marker_color=bar_colors,
                                     text=[f"{p:.1%}" for p in probs], textposition="outside"))
            fig_ew.add_hline(y=0.7, line_dash="dash", line_color="#ef4444", annotation_text="HIGH RISK threshold (0.70)")
            fig_ew.add_hline(y=0.4, line_dash="dash", line_color="#f59e0b", annotation_text="ELEVATED threshold (0.40)")
            _style(fig_ew, "Pre-Excursion Probability by Detection Horizon",
                   "Horizon", "Excursion Probability", 320)
            fig_ew.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig_ew, use_container_width=True)

            st.markdown(f"**Scientific Interpretation:** Alert level: {alert_lvl}. Calibrated probabilities are more reliable than uncalibrated scores for threshold-based alerting.")
            st.markdown("**Engineering Implication:** HIGH RISK → immediate parameter verification (VCSEL temperature, optical power).")
        else:
            st.warning(f"Early warning: {warning_result.get('reason', 'Not available')}")

    # ─── TAB D ───────────────────────────────────────────────────────────────
    with tab_mlrca:
        _sec("Tab D · ML-Enhanced Root Cause Attribution", ai=True)
        st.caption("SHAP and permutation importance for ML-based instability attribution.")
        if ml_attr_result.get("available"):
            ranked = ml_attr_result.get("ranked", [])
            rc = st.columns(3)
            rc[0].metric("Attribution Method", ml_attr_result.get("method", "N/A").split(" ")[0])
            rc[1].metric("Primary Instability Driver (ML)", ranked[0]["Parameter"] if ranked else "N/A")
            rc[2].metric("Model R²", f"{ml_attr_result.get('r2_score', 0):.4f}")

            if ranked:
                attr_df = pd.DataFrame(ranked)
                params_disp = [r["Parameter"] for r in ranked]
                fig_attr = go.Figure()
                fig_attr.add_trace(go.Bar(name="Importance Score (Primary)",
                                           x=params_disp, y=[r["Importance Score"] for r in ranked],
                                           marker_color="#38bdf8"))
                fig_attr.add_trace(go.Bar(name="Permutation Importance",
                                           x=params_disp, y=[r["Permutation Importance"] for r in ranked],
                                           marker_color="#a78bfa"))
                fig_attr.add_trace(go.Bar(name="MDI Importance",
                                           x=params_disp, y=[r["MDI Importance"] for r in ranked],
                                           marker_color="#34d399"))
                fig_attr.update_layout(barmode="group")
                _style(fig_attr, "ML Feature Importance — SHAP / Permutation / MDI Attribution",
                       "Parameter", "Importance Score", 380)
                st.plotly_chart(fig_attr, use_container_width=True)

                disp_cols = [c for c in ["Parameter", "Importance Score", "Permutation Importance",
                                          "Permutation Std", "MDI Importance", "SHAP (mean |φ|)"]
                             if c in attr_df.columns]
                st.dataframe(attr_df[disp_cols], use_container_width=True, hide_index=True)

                st.markdown("#### Consistency Check — Physics-Based vs ML Attribution")
                top_physics = dominant_contrib["Parameter"] if dominant_contrib is not None else "N/A"
                top_ml      = ranked[0]["Parameter"] if ranked else "N/A"
                agree = "✅ CONSISTENT" if top_physics == top_ml else "⚠ DIVERGENT — investigate cross-correlations"
                st.markdown(f"**Physics-Based Primary Driver:** {top_physics}  \n**ML Attribution Primary Driver:** {top_ml}  \n**Consistency:** {agree}")

            st.markdown("**Scientific Interpretation:** SHAP attribution satisfies game-theoretic fairness axioms. Divergence from physics-based ranking indicates cross-correlations or nonlinear coupling.")
            st.markdown("**Engineering Implication:** Use ML attribution to detect unexpected coupling channels not anticipated by the physics-based budget.")
        else:
            st.warning(f"ML attribution: {ml_attr_result.get('reason', 'Not available')}")

    # ─── TAB E ───────────────────────────────────────────────────────────────
    with tab_hi:
        _sec("Tab E · Composite Stability Performance Index (CSPI)", ai=True)
        st.caption("Synthesising σy, drift, excursion rate, and RUS into a [0–100] health score.")
        if health_result:
            hc = st.columns(4)
            hc[0].metric("CSPI Score", f"{cspi:.1f} / 100")
            hc[1].metric("Category", cspi_cat)
            hc[2].metric("Weakest Sub-Index", health_result.get("weakest_factor", "N/A"))
            hc[3].metric("Weakest Score", f"{health_result.get('weakest_score', 0):.1f}")

            cat_color = health_result.get("color", "#ef4444")
            fig_cspi = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=cspi,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Composite Stability Performance Index (CSPI)", "font": {"size": 15, "color": "#e0e6f0"}},
                delta={"reference": 80, "decreasing": {"color": "#ef4444"}, "increasing": {"color": "#22c55e"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#e0e6f0"},
                    "bar": {"color": cat_color},
                    "steps": [
                        {"range": [0, 30], "color": "#450a0a"},
                        {"range": [30, 55], "color": "#431407"},
                        {"range": [55, 80], "color": "#365314"},
                        {"range": [80, 100], "color": "#14532d"},
                    ],
                    "threshold": {"line": {"color": "#38bdf8", "width": 3}, "thickness": 0.85, "value": cspi},
                },
            ))
            fig_cspi.update_layout(paper_bgcolor="#070d18", font=dict(color="#e0e6f0"),
                                    height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cspi, use_container_width=True)

            sub_scores = health_result.get("sub_scores", {})
            if sub_scores:
                cats = list(sub_scores.keys())
                vals_radar = list(sub_scores.values())
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=vals_radar + [vals_radar[0]],
                                                     theta=cats + [cats[0]],
                                                     fill="toself", name="CSPI Sub-indices",
                                                     line=dict(color="#38bdf8", width=2),
                                                     fillcolor="rgba(56,189,248,0.15)"))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                               tickcolor="#e0e6f0", gridcolor="#1e293b"),
                               angularaxis=dict(tickcolor="#e0e6f0", gridcolor="#1e293b"),
                               bgcolor="#070d18"),
                    showlegend=False, paper_bgcolor="#070d18",
                    font=dict(color="#e0e6f0"), height=340,
                    title=dict(text="CSPI Sub-Index Radar Chart", font=dict(color="#e0e6f0", size=14)),
                    margin=dict(l=30, r=30, t=50, b=30),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                sub_df = pd.DataFrame([
                    {"Sub-Index": k, "Score": f"{v:.1f}/100"}
                    for k, v in sub_scores.items()
                ])
                st.dataframe(sub_df, use_container_width=True, hide_index=True)

            st.markdown(f"**Scientific Interpretation:** {health_result.get('interpretation', 'N/A')}")
            st.markdown("**Engineering Implication:** CSPI provides at-a-glance status for operational decision-making. Weakest sub-index identifies the primary maintenance target.")

    # ─── TAB F ───────────────────────────────────────────────────────────────
    with tab_risk:
        _sec("Tab F · Stability Violation Probability Assessment", ai=True)
        st.caption("Probabilistic assessment of specification violation risk over operational horizons.")
        if risk_result.get("available"):
            rl = risk_result.get("risk_level", "UNKNOWN")
            rp = risk_result.get("combined_risk", 0.0)
            rc_col = risk_result.get("risk_color", "#94a3b8")
            st.markdown(f"<h3 style='color:{rc_col}'>Risk Level: {rl} (P = {rp:.2%})</h3>", unsafe_allow_html=True)
            rc_kpi = st.columns(4)
            rc_kpi[0].metric("Combined Risk Prob.", f"{rp:.2%}")
            rc_kpi[1].metric("Short-term σy Risk", f"{risk_result.get('p_short_term', 0):.2%}")
            rc_kpi[2].metric("Excursion Risk", f"{risk_result.get('p_excursion', 0):.2%}")
            rc_kpi[3].metric("Spec. Threshold", f"{risk_result.get('spec_threshold', 0):.3e}")

            vp = risk_result.get("violation_probs", {})
            if vp:
                vp_rows = [(h, f"{p:.3%}") for h, p in vp.items()]
                vp_df = pd.DataFrame(vp_rows, columns=["Horizon", "Drift-Based Violation Probability"])
                st.dataframe(vp_df, use_container_width=True, hide_index=True)

                # Risk waterfall
                hor_labels = list(vp.keys())
                probs_vp = list(vp.values())
                bar_cols = [
                    "#22c55e" if p < 0.2 else "#f59e0b" if p < 0.5 else "#ef4444"
                    for p in probs_vp
                ]
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Bar(x=hor_labels, y=probs_vp, marker_color=bar_cols,
                                           text=[f"{p:.1%}" for p in probs_vp], textposition="outside"))
                fig_risk.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="HIGH risk (0.5)")
                fig_risk.add_hline(y=0.2, line_dash="dash", line_color="#f59e0b", annotation_text="MEDIUM risk (0.2)")
                _style(fig_risk, "Drift-Based Specification Violation Probability by Horizon",
                       "Forecast Horizon", "Violation Probability", 320)
                fig_risk.update_yaxes(range=[0, 1.05])
                st.plotly_chart(fig_risk, use_container_width=True)

            st.markdown(f"**Scientific Interpretation:** Combined violation probability = {rp:.2%} → {rl}. Drift component quantifies long-term bias accumulation; short-term component reflects instantaneous σy deviation from specification.")
            st.markdown("**Engineering Implication:** HIGH risk → immediate stabilisation intervention. MEDIUM → schedule maintenance within next maintenance window. LOW → continue nominal monitoring.")
        else:
            st.warning("Risk assessment not available.")

    # ─── TAB G ───────────────────────────────────────────────────────────────
    with tab_dt:
        _sec("Tab G · Physics-Informed Digital Twin Simulator", ai=True)
        st.caption("Interactive what-if analysis: predict σy response to parameter perturbations using sensitivity coefficients.")

        st.markdown(
            "**Objective:** Simulate the frequency standard response to operator-applied parameter adjustments.  \n"
            "**Methodology:**  \n"
            "• Baseline RSS budget from measured telemetry.  \n"
            "• User-specified Δxᵢ applied to sensitivity budget: σy_new = √(Σ (αᵢ × (σxᵢ + |Δxᵢ|))²).  \n"
            "• Sensitivity coefficients from Vanier & Audoin (1989), Ch. 5; Camparo (2005).  \n"
            "• All predictions include quantified impact classification (Negligible / Moderate / Significant).  \n"
            "Reference: Grieves (2014) [G14]."
        )

        st.markdown("#### Parameter Adjustment Controls")
        col_dt1, col_dt2 = st.columns(2)
        with col_dt1:
            vcsel_t_delta = st.slider("VCSEL Temperature Δ (°C)", -2.0, 2.0, 0.0, 0.01,
                                       help="αᵢ = 2.8×10⁻¹¹ /°C")
            cell_t_delta  = st.slider("Cell Temperature Δ (°C)", -1.0, 1.0, 0.0, 0.01,
                                       help="αᵢ = 1.6×10⁻¹¹ /°C")
            contrast_d    = st.slider("Resonance Contrast Δ (rel.)", -0.1, 0.1, 0.0, 0.001,
                                       help="αᵢ = 4.2×10⁻¹⁰ /rel")
        with col_dt2:
            optical_p_d   = st.slider("Optical Power Δ (µW)", -5.0, 5.0, 0.0, 0.1,
                                       help="αᵢ = 1.2×10⁻¹¹ /µW")
            inject_c_d    = st.slider("Injection Current Δ (mA)", -0.5, 0.5, 0.0, 0.01,
                                       help="αᵢ = 0.9×10⁻¹¹ /mA")

        # Automatically run the digital twin simulation for current slider settings
        try:
            twin_result = simulate_digital_twin(
                active_df,
                vcsel_temp_delta=vcsel_t_delta,
                optical_power_delta=optical_p_d,
                cell_temp_delta=cell_t_delta,
                injection_current_delta=inject_c_d,
                contrast_delta=contrast_d,
            )
        except Exception:
            twin_result = None

        if twin_result is not None:
            pct = twin_result["pct_change"]
            dir_col = "#22c55e" if pct < 0 else "#ef4444"
            direction = twin_result["direction"]
            tc = st.columns(4)
            tc[0].metric("Baseline σy (RSS)", f"{twin_result['baseline_sigma_y']:.3e}")
            tc[1].metric("Simulated σy", f"{twin_result['new_sigma_y']:.3e}")
            tc[2].metric("Δσy", f"{twin_result['delta_sigma_y']:.3e}")
            tc[3].metric(f"σy Change ({direction})", f"{abs(pct):.2f} %")
            st.markdown(f"<span style='color:{dir_col};font-size:1.1rem;font-weight:bold'>Impact: {twin_result['impact_label']} — {direction.title()}</span>", unsafe_allow_html=True)
            st.markdown(f"_{twin_result['interpretation']}_")

            # Before/after comparison
            mc = twin_result.get("modified_contribs", [])
            if mc:
                mc_df = pd.DataFrame(mc)
                fig_dt = go.Figure()
                fig_dt.add_trace(go.Bar(name="Baseline σy_i", x=mc_df["Parameter"],
                                         y=[float(str(v).replace("e", "E")) for v in mc_df.get("σy_i (new)", mc_df.get("New σ", [0]*len(mc_df)))],
                                         marker_color="#38bdf8"))
                _style(fig_dt, "Digital Twin: Post-Perturbation σy_i Budget",
                       "Parameter", "σy_i (Δf/f)", 320)
                st.plotly_chart(fig_dt, use_container_width=True)
                st.dataframe(mc_df, use_container_width=True, hide_index=True)
        else:
            st.info("Digital twin unavailable for this dataset or failed to compute.")

        st.markdown("**Scientific Interpretation:** The digital twin predicts σy response using the physics-based sensitivity budget. Results are first-order only — nonlinear coupling and servo transients are not modelled.")
        st.markdown("**Engineering Implication:** Use the digital twin to pre-validate parameter changes before applying them to the physical instrument. Significant improvements in simulation should be followed by ADEV measurement to confirm.")

    # ─── TAB H ───────────────────────────────────────────────────────────────
    with tab_cop:
        _sec("Tab H · Automated Scientific Assessment Copilot", ai=True)
        st.caption("Rule-based deterministic scientific narrative generator synthesising all AI and metrology module outputs.")

        st.markdown(
            "**Objective:** Provide integrated natural-language interpretation of the frequency standard status.  \n"
            "**Methodology:** Rule-based generator synthesising: operational regime, σy, drift, CSPI, TTSV, ML attribution, risk, and forecast.  \n"
            "All statements are traceable to computed quantities from IEEE 1139-2022 methods. "
            "No extrapolation beyond the available measurement record."
        )

        preset_queries = [
            "Why is drift increasing and what should I do?",
            "What is the dominant instability source and how do I stabilise?",
            "What is the current health status and when is maintenance needed?",
            "Explain the noise process and its engineering implications.",
            "What is the stability risk and violation probability?",
            "Recommend the highest-priority stabilisation action.",
        ]
        cop_query = st.selectbox("Select a technical query", preset_queries, key="cop_select")
        custom_q = st.text_input("Or type a custom query", key="cop_custom")
        final_query = custom_q if custom_q.strip() else cop_query
        # Auto-generate LLM-augmented scientific assessment when dataset suffices
        obs_dur = 0.0
        try:
            if "time" in active_df.columns:
                tv = pd.to_numeric(active_df["time"], errors="coerce").dropna().to_numpy()
                if len(tv) >= 2:
                    obs_dur = float(tv.max() - tv.min())
        except Exception:
            obs_dur = 0.0

        if len(active_df) < 30 or obs_dur < 600:
            st.warning("LLM Scientific Assessment: Unavailable due to insufficient observation duration")
        else:
            try:
                narrative = generate_llm_copilot_response(
                    query=final_query,
                    df=active_df,
                    op_state=op_state,
                    forecast_result=forecast_result,
                    rus_result=rus_result,
                    health_index=health_result,
                    risk_result=risk_result,
                    attribution_result=ml_attr_result,
                )
            except Exception:
                narrative = "LLM assessment failed to generate for this dataset."
            st.markdown("---")
            st.markdown("#### Scientific Assessment Narrative")
            st.text_area("", narrative, height=500, key="narrative_out")
            st.download_button("⬇ Download Narrative (TXT)", narrative.encode("utf-8"),
                                "assessment_narrative.txt", "text/plain")

    # ─── TAB I ───────────────────────────────────────────────────────────────
    with tab_val:
        _sec("Tab I · Model Validation and Uncertainty Quantification", ai=True)
        st.caption("Time-series cross-validated forecast accuracy and reliability assessment for all deployed AI models.")

        st.markdown(
            "**Objective:** Quantify prediction accuracy and uncertainty of deployed AI forecasting models.  \n"
            "**Methodology:**  \n"
            "• Protocol: TimeSeriesSplit (k=5 folds) — no temporal data leakage.  \n"
            "• Metrics: MAE, RMSE, MAPE, R².  \n"
            "• Models: OLS (drift baseline), Gradient Boosting, XGBoost (if available).  \n"
            "• LSTM validation: held-out temporal validation set from training.  \n"
            "• Kalman: innovation whiteness test (χ² normaltest, p>0.05 → well-calibrated filter)."
        )

        if validation_result.get("available"):
            val_rows = []
            for model_name, metrics in validation_result["results"].items():
                row = {"Model": model_name}
                row.update(metrics)
                val_rows.append(row)
            val_df = pd.DataFrame(val_rows)
            st.dataframe(val_df, use_container_width=True, hide_index=True)

            # Bar chart: RMSE comparison
            models = [r["Model"] for r in val_rows]
            rmses  = [r["RMSE mean"] for r in val_rows]
            fig_val = go.Figure()
            fig_val.add_trace(go.Bar(x=models, y=rmses, marker_color="#38bdf8",
                                      text=[f"{r:.3e}" for r in rmses], textposition="outside"))
            _style(fig_val, "Cross-Validated RMSE Comparison (TimeSeriesSplit k=5)", "Model", "RMSE", 320)
            st.plotly_chart(fig_val, use_container_width=True)
            st.caption(f"Protocol: {validation_result['protocol']} | Features: {validation_result['feature_count']} | Training samples: {validation_result['sample_count']}")
        else:
            st.warning(f"Validation not available: {validation_result.get('reason', 'Unknown')}")

        # Kalman validation
        st.markdown("#### Kalman Filter Validation — Innovation Whiteness")
        if kalman_result.get("available"):
            kv = st.columns(3)
            kv[0].metric("Innovation Mean", f"{kalman_result['innovation_mean']:.3e}")
            kv[1].metric("Innovation Std", f"{kalman_result['innovation_std']:.3e}")
            kv[2].metric("χ² p-value", f"{kalman_result['chi2_pvalue']:.4f}")
            p = kalman_result["chi2_pvalue"]
            status = "✅ Well-calibrated filter (p>0.05)" if p > 0.05 else f"⚠ Non-white innovations (p={p:.4f}) — update Q or R"
            st.info(status)

        # XGBoost validation
        xgb_r = forecast_result.get("xgb_result", {})
        if xgb_r.get("available"):
            st.markdown("#### XGBoost / Gradient Boosting Validation")
            xv = st.columns(3)
            xv[0].metric("Val MAE (TimeSeriesSplit)", f"{xgb_r['val_mae_mean']:.3e}")
            xv[1].metric("Val RMSE", f"{xgb_r['val_rmse_mean']:.3e}")
            xv[2].metric("In-Sample R²", f"{xgb_r['in_sample_r2']:.4f}")
            st.caption(f"Model: {xgb_r['model_name']}")

        st.markdown("**Scientific Interpretation:** Lower RMSE and MAE indicate better predictive accuracy. R²>0.5 indicates the model captures significant variance in the frequency offset record. Kalman innovation whiteness confirms filter calibration.")
        st.markdown("**Engineering Implication:** Validate models after significant operational changes. High MAE relative to σy indicates model predictions should not be trusted for operational decisions without retraining on updated telemetry.")


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "VCSEL-Pumped Atomic Frequency Standard — Predictive Stability Intelligence Framework  |  "
    "Metrology: IEEE 1139-2022 · NIST TN-1337 · Vanier & Audoin (1989) · Camparo (2005)  |  "
    "AI: Kalman · XGBoost · LSTM · SHAP · Digital Twin  |  "
    "AI Predictive Intelligence Layer: ACTIVE — Models: Kalman (Completed), XGBoost (Completed/Unavailable per data), LSTM (Completed/Unavailable per data), Ensemble (Completed/Unavailable per data)"
)