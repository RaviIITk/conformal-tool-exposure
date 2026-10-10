# Phase 3 results: stress tests (hybrid, α = 0.1, stage2 view, K = 50)

Threshold q̂ = 0.9914, calibrated on 1086 clean single-tool steps.

## RQ3a: catalog drift (remove 20% of tools, add 20% from other catalogs, rename 20%, near-duplicate lookalikes for half of the gold tools)

| condition | n | prefilter recall | coverage | avg set size | flagged tool in set | flagged prob. mass |
|---|---|---|---|---|---|---|
| clean test | 1079 | 0.615 | 0.886 | 20.074 | – | – |
| drifted, static q̂ (flagged = lookalikes) | 1067 | 0.608 | 0.910 | 20.257 | 0.535 | 0.053 |

| stream metric | value |
|---|---|
| static_coverage | 0.910 |
| aci_coverage | 0.904 |
| aci_coverage_second_half | 0.910 |
| oracle_recalibrated_coverage_second_half | 0.897 |
| aci_final_qhat | 0.987 |
| oracle_qhat | 0.991 |
| aci_recovery_steps | 200 |
| aci_avg_size | 20.599 |

## RQ3b: distribution shift (calibrate on one catalog, test on the others)

| calibration | n cal | n test | q̂ | static coverage | ACI coverage |
|---|---|---|---|---|---|
| calibrate_web | 1286 | 879 | 0.989 | 0.826 | 0.900 |
| calibrate_code | 619 | 1546 | 0.994 | 0.950 | 0.903 |

## RQ4: poisoned tool descriptions (flagged = poisoned tools)

### 1% of tools poisoned (445 tools)

| condition | n | prefilter recall | coverage | avg set size | flagged tool in set | flagged prob. mass |
|---|---|---|---|---|---|---|
| clean catalog (same tools, unpoisoned) | 1079 | 0.615 | 0.886 | 20.074 | 0.145 | 0.009 |
| poisoned | 1081 | 0.616 | 0.886 | 20.140 | 0.177 | 0.010 |
| poisoned + sanity check | 1081 | 0.616 | 0.890 | 20.036 | 0.148 | 0.009 |

Sanity-check detector: precision 0.843, recall 1.000, false-positive rate on clean tools 0.0019.

### 5% of tools poisoned (2223 tools)

| condition | n | prefilter recall | coverage | avg set size | flagged tool in set | flagged prob. mass |
|---|---|---|---|---|---|---|
| clean catalog (same tools, unpoisoned) | 1079 | 0.615 | 0.886 | 20.074 | 0.558 | 0.040 |
| poisoned | 1077 | 0.614 | 0.890 | 20.069 | 0.554 | 0.044 |
| poisoned + sanity check | 1081 | 0.616 | 0.891 | 20.043 | 0.562 | 0.040 |

Sanity-check detector: precision 0.966, recall 1.000, false-positive rate on clean tools 0.0019.

Figure: `drift_rolling_coverage.png`.
