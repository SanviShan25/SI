import pandas as pd

from dashboard.ai_framework import (
    build_ai_anomaly_analysis,
    build_ai_copilot_response,
    build_ai_stability_intelligence,
    build_excursion_attribution_summary,
    build_predictive_stability_summary,
    build_scientific_report,
    build_stability_forecast,
    compute_allan_noise_analysis,
    compute_atomic_clock_health_score,
    compute_physical_attribution,
    compute_shap_feature_importance,
    compute_stability_budget,
    compute_sensitivity_coefficients,
    classify_operational_state,
    generate_stabilization_recommendations,
    build_ai_stabilization_recommendations,
)


def make_sample_frame():
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=40, freq="min"),
            "vcsel_temp": [35.0 + i * 0.01 for i in range(40)],
            "vcsel_current": [18.0 + i * 0.005 for i in range(40)],
            "optical_power": [1.10 + i * 0.001 for i in range(40)],
            "cell_temp": [40.0 + i * 0.008 for i in range(40)],
            "contrast": [0.82 + 0.002 * (i % 5) for i in range(40)],
            "frequency_offset": [0.0, 0.1, -0.2, 0.3, -0.4] + [0.0] * 35,
        }
    )


def test_ai_anomaly_analysis_returns_scores_and_risk_levels():
    df = make_sample_frame()
    analysis = build_ai_anomaly_analysis(df)
    assert 0 <= analysis["anomaly_score"] <= 100
    assert analysis["risk_level"] in {"Normal", "Warning", "Critical"}
    assert len(analysis["timeline"]) == len(df)
    assert analysis["top_contributors"]


def test_stability_forecast_contains_all_requested_horizons():
    df = make_sample_frame()
    forecast = build_stability_forecast(df)
    assert forecast["horizons_minutes"] == [0, 15, 30, 60]
    assert len(forecast["frequency_offset_forecast"]) == 4
    assert len(forecast["allan_forecast"]) == 4
    assert len(forecast["drift_forecast"]) == 4
    assert len(forecast["confidence_intervals"]) == 4


def test_stabilization_recommendations_have_action_and_priority():
    df = make_sample_frame()
    anomaly = build_ai_anomaly_analysis(df)
    forecast = build_stability_forecast(df)
    recommendations = generate_stabilization_recommendations(anomaly, forecast)
    assert recommendations
    assert all("Recommended Action" in rec for rec in recommendations)
    assert all("Priority" in rec for rec in recommendations)
    assert all("Current Value" in rec for rec in recommendations)
    assert all("Target Value" in rec for rec in recommendations)
    assert all("Sensitivity Coefficient" in rec for rec in recommendations)
    assert all("Estimated Stability Improvement" in rec for rec in recommendations)


def test_allan_noise_analysis_identifies_a_measurable_regime():
    df = make_sample_frame()
    analysis = compute_allan_noise_analysis(df)
    assert analysis["dominant_noise_process"] in {
        "White Phase Noise",
        "White Frequency Noise",
        "Flicker Frequency Noise",
        "Random Walk Frequency Noise",
    }
    assert analysis["local_slopes"]
    assert analysis["transition_regions"]


def test_health_score_is_bounded_and_has_grade():
    score = compute_atomic_clock_health_score(
        stability_score=82.0,
        drift_score=74.0,
        excursion_score=88.0,
        environmental_score=79.0,
        predictive_risk=24.0,
    )
    assert 0 <= score["overall_health"] <= 100
    assert score["stability_grade"] in {"A", "B", "C", "D", "F"}
    assert score["risk_indicator"] in {"Low", "Moderate", "High", "Critical"}


def test_operational_state_is_scientifically_classified():
    state = classify_operational_state(
        allan_sigma1=2.5e-11,
        allan_sigma10=3.0e-11,
        allan_sigma100=4.0e-11,
        drift_rate=1.0e-13,
        excursion_count=1,
        max_excursion=2.0e-11,
        mean_offset=1.0e-11,
    )
    assert state in {"STABLE", "WARNING", "UNSTABLE"}


def test_stability_budget_contains_physical_contributions():
    df = make_sample_frame()
    budget = compute_stability_budget(df)
    assert budget["total_contribution"] > 0.0
    assert budget["residual_noise"] >= 0.0
    assert budget["contributions"]


def test_sensitivity_coefficients_are_produced_for_physical_parameters():
    df = make_sample_frame()
    coefficients = compute_sensitivity_coefficients(df)
    assert coefficients["VCSEL Temperature"] >= 0.0
    assert coefficients["Optical Power"] >= 0.0


def test_physical_attribution_uses_sensitivity_and_deviation():
    df = make_sample_frame()
    attribution = compute_physical_attribution(df)
    assert attribution
    assert all(
        {"Parameter", "Sensitivity Coefficient", "Measured Deviation", "Estimated Frequency Contribution", "Contribution Percentage"}.issubset(set(item))
        for item in attribution
    )


def test_scientific_report_contains_structured_sections():
    df = make_sample_frame()
    report = build_scientific_report(
        anomaly_analysis=build_ai_anomaly_analysis(df),
        forecast_analysis=build_stability_forecast(df),
        recommendations=generate_stabilization_recommendations(build_ai_anomaly_analysis(df), build_stability_forecast(df)),
    )
    assert "System Summary" in report
    assert "Current Stability Assessment" in report
    assert "Recommended Stabilization Actions" in report


def test_ai_stability_intelligence_produces_health_and_risk_summary():
    df = make_sample_frame()
    intelligence = build_ai_stability_intelligence(df)
    assert 0 <= intelligence["health_score"] <= 100
    assert intelligence["health_category"] in {"Stable", "Mild Risk", "Elevated Risk", "Critical"}
    assert intelligence["current_risk_level"] in {"Stable", "Mild Risk", "Elevated Risk", "Critical"}
    assert intelligence["predicted_risk_level"] in {"Stable", "Mild Risk", "Elevated Risk", "Critical"}


def test_predictive_stability_summary_reports_horizons_and_placeholder_notes():
    df = make_sample_frame()
    summary = build_predictive_stability_summary(df)
    assert summary["horizons_hours"] == [1, 6, 24]
    assert len(summary["physics_forecast"]) == 3
    assert len(summary["ml_forecast"]) == 3
    assert "placeholder" in summary["notes"].lower()


def test_shap_feature_importance_returns_ranked_parameters():
    df = make_sample_frame()
    importance = compute_shap_feature_importance(df)
    assert importance["method"] in {"shap", "fallback"}
    assert importance["feature_importance"]
    assert all(item["Parameter"] for item in importance["feature_importance"])


def test_excursion_attribution_summary_returns_ranked_contributions():
    df = make_sample_frame()
    summary = build_excursion_attribution_summary(df)
    assert summary["top_contributors"]
    assert summary["confidence_score"] >= 0.0


def test_ai_stabilization_recommendations_include_confidence_scores():
    df = make_sample_frame()
    recommendations = build_ai_stabilization_recommendations(df)
    assert recommendations
    assert all("Confidence Score" in rec for rec in recommendations)


def test_copilot_response_mentions_stability_context():
    df = make_sample_frame()
    response = build_ai_copilot_response("Why is drift increasing?", df)
    assert "stability" in response.lower() or "drift" in response.lower()
