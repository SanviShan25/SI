"""
Frequency Metrology & AI Intelligence Package — Dashboard Module
================================================================
Exports all metrology and AI intelligence computation functions.
"""
from .ai_framework import (
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

from .ai_models import (
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
    STABILITY_THRESHOLDS,
    FORECAST_HORIZONS_HOURS,
    LSTM_STATUS,
    COPILOT_IS_RULE_BASED,
    COPILOT_DESCRIPTION,
)

__all__ = [
    # Metrology engine (ai_framework.py)
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
    # AI intelligence engine (ai_models.py)
    "run_kalman_analysis",
    "compute_predictive_stability_forecast",
    "compute_remaining_useful_stability",
    "compute_early_warning",
    "compute_ml_root_cause_attribution",
    "compute_health_index",
    "compute_stability_risk_assessment",
    "simulate_digital_twin",
    "generate_llm_copilot_response",
    "generate_scientific_interpretation",
    "compute_model_validation_metrics",
    "STABILITY_THRESHOLDS",
    "FORECAST_HORIZONS_HOURS",
    "LSTM_STATUS",
    "COPILOT_IS_RULE_BASED",
    "COPILOT_DESCRIPTION",
]
