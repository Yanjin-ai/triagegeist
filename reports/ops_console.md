# ED operations console (prototype) — text-blind structured model

Incoming cohort = test set (20,000). Snapshot = 50 sampled patients as a waiting-room view. `wait_hours` is simulated for the demo (a real-time input in deployment).

> Honesty carryover (P5): buckets/priority encode physiology+outcome logic that can diverge from the text-determined acuity labels; in a real ED that divergence is the *point*.

## Ranked queue — policy = `safety_first` (top 12 of snapshot)

| patient_id   |   acuity |   p_critical |   p_admit |   entropy | bucket                      |   wait_hours |   priority | senior_review   |
|:-------------|---------:|-------------:|----------:|----------:|:----------------------------|-------------:|-----------:|:----------------|
| TG-VC7LOICZA |        1 |            1 |    0.8742 |    0.011  | resuscitation/immediate bed |         0.4  |   0.806153 | False           |
| TG-OUHYKKVIP |        1 |            1 |    0.9683 |    0      | resuscitation/immediate bed |         0.02 |   0.800267 | False           |
| TG-JOFVOA5JU |        2 |            1 |    0.8355 |    0      | high-frequency monitoring   |         2.89 |   0.726033 | False           |
| TG-Q7WZM2YV1 |        2 |            1 |    0.8469 |    0      | high-frequency monitoring   |         2.19 |   0.7167   | False           |
| TG-UII6DIFK2 |        2 |            1 |    0.7535 |    0.0025 | high-frequency monitoring   |         1.53 |   0.708086 | False           |
| TG-IWDAOI3N7 |        2 |            1 |    0.8613 |    0      | high-frequency monitoring   |         1.34 |   0.705367 | False           |
| TG-MLRJITBQT |        2 |            1 |    0.8484 |    0      | high-frequency monitoring   |         1.17 |   0.7031   | False           |
| TG-3O2RRF8WY |        2 |            1 |    0.8582 |    0.0285 | high-frequency monitoring   |         0.94 |   0.702158 | False           |
| TG-F4H8CTHA2 |        2 |            1 |    0.8693 |    0.0285 | high-frequency monitoring   |         0.7  |   0.698958 | False           |
| TG-ZGNS7GQQM |        2 |            1 |    0.8096 |    0      | high-frequency monitoring   |         0.66 |   0.6963   | False           |
| TG-29B6CU00A |        2 |            1 |    0.8515 |    0      | high-frequency monitoring   |         0.31 |   0.691633 | False           |
| TG-LW5SMTCXX |        2 |            1 |    0.8692 |    0.0285 | high-frequency monitoring   |         0.07 |   0.690558 | False           |

## Ranked queue — policy = `throughput` (top 12 of snapshot)

| patient_id   |   acuity |   p_critical |   p_admit |   entropy | bucket                      |   wait_hours |   priority | senior_review   |
|:-------------|---------:|-------------:|----------:|----------:|:----------------------------|-------------:|-----------:|:----------------|
| TG-VC7LOICZA |        1 |            1 |    0.8742 |    0.011  | resuscitation/immediate bed |         0.4  |   0.717008 | False           |
| TG-OUHYKKVIP |        1 |            1 |    0.9683 |    0      | resuscitation/immediate bed |         0.02 |   0.700833 | False           |
| TG-JOFVOA5JU |        2 |            1 |    0.8355 |    0      | high-frequency monitoring   |         2.89 |   0.682917 | False           |
| TG-Q7WZM2YV1 |        2 |            1 |    0.8469 |    0      | high-frequency monitoring   |         2.19 |   0.65375  | False           |
| TG-UII6DIFK2 |        2 |            1 |    0.7535 |    0.0025 | high-frequency monitoring   |         1.53 |   0.626328 | False           |
| TG-IWDAOI3N7 |        2 |            1 |    0.8613 |    0      | high-frequency monitoring   |         1.34 |   0.618333 | False           |
| TG-MLRJITBQT |        2 |            1 |    0.8484 |    0      | high-frequency monitoring   |         1.17 |   0.61125  | False           |
| TG-3O2RRF8WY |        2 |            1 |    0.8582 |    0.0285 | high-frequency monitoring   |         0.94 |   0.602552 | False           |
| TG-F4H8CTHA2 |        2 |            1 |    0.8693 |    0.0285 | high-frequency monitoring   |         0.7  |   0.592552 | False           |
| TG-ZGNS7GQQM |        2 |            1 |    0.8096 |    0      | high-frequency monitoring   |         0.66 |   0.59     | False           |
| TG-29B6CU00A |        2 |            1 |    0.8515 |    0      | high-frequency monitoring   |         0.31 |   0.575417 | False           |
| TG-118CJ3GVD |        2 |            1 |    0.8963 |    0.0025 | high-frequency monitoring   |         0.13 |   0.567994 | False           |

## ED load summary (snapshot, n_beds=30)

|                           | value                                                                                                                                      |
|:--------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| n_patients                | 50                                                                                                                                         |
| acuity_counts             | {1: 2, 2: 12, 3: 14, 4: 13, 5: 9}                                                                                                          |
| bucket_counts             | {'high-frequency monitoring': 21, 'likely discharge': 13, 'fast-track/minor care': 9, 'standard bed': 5, 'resuscitation/immediate bed': 2} |
| expected_admissions       | 22.3                                                                                                                                       |
| admission_rate            | 0.446                                                                                                                                      |
| bed_demand                | 28                                                                                                                                         |
| bed_pressure              | 0.93                                                                                                                                       |
| mean_pred_los_h           | 3.58                                                                                                                                       |
| high_uncertainty_critical | 0                                                                                                                                          |

## Scenario flow by shift (full cohort — flow counts, not point-in-time census)

| scenario   |   n_patients |   expected_admissions |   admission_rate |   high_uncertainty_critical |   mean_pred_los_h |
|:-----------|-------------:|----------------------:|-----------------:|----------------------------:|------------------:|
| morning    |         6631 |                2865.1 |            0.432 |                          60 |              3.48 |
| afternoon  |         5041 |                2191.8 |            0.435 |                          58 |              3.51 |
| night      |         4965 |                2152.3 |            0.433 |                          60 |              3.49 |
| evening    |         3363 |                1470.8 |            0.437 |                          34 |              3.52 |
