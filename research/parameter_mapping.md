# Parameter Mapping for AI Atomic Clock Monitoring System

## Rubidium Atomic Clock Parameters

| Parameter         | Physical Meaning              | Research Source           | Module Used          |
| ----------------- | ----------------------------- | ------------------------- | -------------------- |
| VCSEL Temperature | Laser operating temperature   | VCSEL Atomic Clock Papers | Digital Twin         |
| VCSEL Current     | Laser drive current           | Rubidium Clock Literature | Digital Twin         |
| Optical Power     | Light intensity entering cell | Camparo Aging Paper       | Root Cause Analysis  |
| Cell Temperature  | Vapor cell temperature        | NIST CSAC Paper           | Stability Monitoring |
| Magnetic Field    | Zeeman Shift Influence        | Atomic Clock Literature   | Root Cause Analysis  |
| Contrast          | CPT Resonance Quality         | Rubidium Clock Papers     | Lock Prediction      |
| Frequency Offset  | Clock Error                   | NIST Handbook             | Dashboard            |
| Allan Deviation   | Stability Metric              | NIST Handbook             | Health Score         |

## AI Modules Mapping

| Module          | Inputs              | Output                    |
| --------------- | ------------------- | ------------------------- |
| Digital Twin    | Physical Parameters | Simulated Clock Behaviour |
| One-Class SVM   | Frequency Offset    | Anomaly Flag              |
| Drift Predictor | Historical Offset   | Future Drift              |
| Physical Sensitivity | Metrology Model | Root Cause Attribution |
| Lock Predictor  | Physical Parameters | Lock / Unlock Prediction  |

## Research Justification

The selected parameters are directly derived from atomic clock literature and represent the dominant physical factors affecting frequency stability, clock aging, lock performance, and long-term drift behaviour.