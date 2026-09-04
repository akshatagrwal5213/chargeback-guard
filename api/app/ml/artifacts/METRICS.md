# Metrics

Model `lgbm-1` · 120,000 orders · 3.53% disputed
Split: **time-based (60/15/25 chronological)** · operating threshold **0.18**

| Metric | Value |
|---|---|
| Precision @ threshold | **0.337** |
| Recall @ threshold | **0.343** |
| F1 | **0.340** |
| PR-AUC | 0.289 |
| ROC-AUC | 0.807 |
| Brier score | 0.0600 |
| Orders flagged for enhanced capture | 7.5% |

Test set: 30,000 orders, 2,219 disputed — a 8.2x
lift in PR-AUC over the base rate.

Lead with precision and recall, not ROC-AUC. At a 3.53% base
rate ROC-AUC (0.807) flatters the model and says little about
whether the flags are worth acting on.

## Leakage audit

The same model and threshold, scored on a **random** split instead of a
chronological one:

| Metric | Time-based (reported) | Random split | Difference |
|---|---|---|---|
| Precision | 0.337 | 0.299 | -0.038 |
| Recall | 0.343 | 0.410 | +0.067 |
| F1 | 0.340 | 0.346 | +0.006 |
| PR-AUC | 0.289 | 0.258 | -0.031 |

The random split comes out **-0.031 PR-AUC** — no material inflation on this dataset, so the conservative split costs us nothing here. We still use it. It is the only split that survives contact with drifting data, and picking a split after seeing which one scores better is the thing this audit exists to prevent.

Run it yourself: the audit is part of every training run, not a one-off.

## False-positive cost

The model's action is **enhanced evidence capture**, not declining the order —
a signature on delivery, a listing snapshot, a retained session log. So a false
positive costs ₹120. Nobody is refused service and no sale
is lost.

That asymmetry sets the threshold. Purely on net value the model would flag far
more, because being wrong is so cheap; the ceiling below is what a merchant can
actually operate.

| | |
|---|---|
| Threshold | 0.18 |
| Orders flagged | 7.5% (ceiling 15%) |
| Net value on the test slice | ₹212,890 |
| False positives | 1,494 at ₹120 each |
| Assumed win-rate uplift from evidence | 15% |
| Calibration error (mean abs) | 0.0156 |

Probabilities are isotonic-calibrated on a slice held out between train and
test, because the triage rule multiplies them by rupee amounts — expected value
computed from uncalibrated scores is arithmetic on noise.

The three economic assumptions above are **stated, not learned**. Change them
with `--capture-cost`, `--win-uplift`, `--max-flag-rate` and the threshold
moves accordingly.

## Top features

| Feature | mean \|SHAP\| |
|---|---|
| `txns_card_24h` | 0.8564 |
| `three_ds_authenticated` | 0.5256 |
| `cvv_match` | 0.3366 |
| `account_age_days` | 0.3210 |
| `txns_card_7d` | 0.1764 |
| `email_domain_risk` | 0.1509 |
| `avs_match` | 0.1220 |
| `prior_disputes` | 0.1009 |

SHAP values are merchant-internal only — they are a map of which signals raise
a score, and so are never returned on a customer-facing surface. See
`docs/DEFENSE_ONLY.md`.
