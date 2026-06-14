import numpy as np

from dashboard.scientific_metrics import (
    build_short_horizon_forecast,
    compute_allan_improvement,
    compute_closed_loop_validation,
    compute_deployment_readiness,
    compute_digital_twin_validation,
    compute_forecast_validation,
    compute_state_estimation_confidence,
)


def test_allan_improvement_is_positive_when_optimized_is_better():
    assert compute_allan_improvement(1e-10, 7e-11) == 30.0


def test_digital_twin_validation_metrics_are_computed_from_data():
    measured = np.array([1.0, 1.2, 1.4, 1.6])
    predicted = np.array([1.0, 1.1, 1.3, 1.5])
    metrics = compute_digital_twin_validation(measured, predicted, baseline_allan=1e-10)
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert 0.0 <= metrics["r2"] <= 1.0
    assert metrics["fidelity_score"] > 0


def test_short_horizon_forecast_contains_expected_horizons():
    forecast = build_short_horizon_forecast(
        allan_value=1e-10,
        drift_rate=2e-12,
        excursion_rate=0.05,
        horizons_minutes=[1, 5, 15, 30, 60],
    )
    assert list(forecast["horizons_minutes"]) == [1, 5, 15, 30, 60]
    assert len(forecast["forecast_allan"]) == 5
    assert len(forecast["excursion_probability"]) == 5
    assert len(forecast["confidence_intervals"]) == 5


def test_forecast_validation_metrics_are_computed_from_data():
    actual = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    forecast = np.array([1.02, 1.08, 1.18, 1.28, 1.38])
    validation = compute_forecast_validation(actual, forecast)
    assert validation["forecast_mae"] > 0
    assert validation["forecast_rmse"] > 0
    assert validation["confidence_interval_width"] > 0


def test_closed_loop_validation_metrics_are_physically_consistent():
    validation = compute_closed_loop_validation(
        baseline_allan=1e-10,
        optimized_allan=7e-11,
        baseline_sri=35.0,
        optimized_sri=22.0,
        baseline_chi=70.0,
        optimized_chi=82.0,
        baseline_excursions=4,
        optimized_excursions=2,
    )
    assert validation["allan_improvement"] > 0
    assert validation["sri_reduction"] > 0
    assert validation["chi_improvement"] > 0
    assert validation["excursion_reduction"] > 0


def test_state_estimation_confidence_is_computed_from_uncertainty():
    confidence = compute_state_estimation_confidence(covariance_trace=1e-8, innovation_norm=0.1, gain_norm=0.2)
    assert 0.0 <= confidence["kalman_confidence"] <= 100.0
    assert 0.0 <= confidence["innovation_consistency"] <= 100.0


def test_deployment_readiness_uses_the_subsystem_scores():
    readiness = compute_deployment_readiness(
        stability_improvement=30.0,
        forecast_accuracy=85.0,
        twin_fidelity=92.0,
        excursion_reduction=40.0,
        control_confidence=88.0,
    )
    assert readiness["score"] > 0
    assert readiness["category"] in {"Experimental", "Development", "Pre-Deployment", "Operational", "Field Ready"}