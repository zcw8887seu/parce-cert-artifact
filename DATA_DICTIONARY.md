# Aggregate Data Dictionary

All files in `data/` are curated publication tables. They do not contain raw events or machine-specific metadata.

## H3R1 reduced reproducibility release

`h3r1_reduced_runs.csv` and `h3r1_reconstruction_config.json` are the smallest released H3R1 inputs needed to recompute the selected calibration order statistics, session-aware run-order diagnostics, nominal validation rows, candidate Gate bound, and the fail-closed H3 disposition. They are derived data, not a raw-event archive.

### `h3r1_reduced_runs.csv`

There are 32,500 rows: 15,000 calibration rows (3,000 selected run vectors × 5 coordinates) and 17,500 validation rows (3,500 × 5). Each row is one coordinate from the pre-specified selected analysis instance of a complete fresh-process run.

| Column | Meaning |
|---|---|
| `split` | `calibration` or `validation`; the two datasets remain disjoint. |
| `session_label` | Neutral split-local session label (`CAL-01`--`CAL-20` or `VAL-01`--`VAL-20`), not the original session identifier. |
| `run_order_in_session` | Ordinal statistical-unit order within the neutral session; it is sufficient to form within-session lag pairs and never encodes a timestamp. |
| `statistical_unit` | Fixed value `selected_analysis_instance`, identifying the one pre-specified analyzed instance per complete run. |
| `coordinate_id` | `X1`, `X2`, `X3`, `X4`, or `E2E`. |
| `value_ns` | Derived duration in nanoseconds. It is a difference, not an absolute timestamp. Blank only when the corresponding value is infinite. |
| `is_infinite` | `true` for a noncompletion/loss-derived infinite coordinate, otherwise `false`. |

### `h3r1_reconstruction_config.json`

Frozen non-identifying configuration and expected numerical outputs: sample and family sizes, selected risks/envelopes, periodic-window semantics, the temporal-guard formula, the candidate bound, and expected diagnostic/validation checks. It contains no run plan, execution timestamps, host identity, integrity manifest, or hash.

The released data intentionally exclude absolute timestamps; original session IDs; global run IDs; host, user, process, path, IP/MAC, CPU, kernel, and hardware identifiers; raw event records; experiment plans and launch material; hashes, manifests, and security/integrity records. Consequently, the reconstruction independently recomputes coordinate-level event inclusion, but it does not replay the withheld raw-timestamp reference-mismatch comparison. Its frozen checked count is stated transparently in the configuration.

## `phase_boundary_summary.csv`

One row per fine-sweep phase.

| Column | Meaning |
|---|---|
| `phase_ns` | Frozen phase offset in nanoseconds |
| `n_instances` | Number of analyzed instances at that phase |
| `median_observed_wait_ns` | Median observed gate wait |
| `median_phase_bound_ns` | Median exact phase-aware bound |
| `median_maxwait_bound_ns` | Median legal maximum-wait surrogate |

## `selection_validity.csv`

The 18 main distribution × gate × procedure cells. Confidence limits are one-sided Clopper--Pearson limits for the repeated-procedure family failure rate.

## `h3_validation.csv`

Five nominal validation items for the independently frozen H3R1 design. `status=SUPPORTED` is conditional on the iid interpretation and is not the final certificate-object state.

## `h3_diagnostics.csv`

Frozen lag-one diagnostic summaries for calibration and validation. `correlation_triggered` controls the fail-closed acquisition disposition.

## `allocator_exactness.csv`

Counts of exhaustive allocator comparisons and mismatches.

## `allocator_scaling_summary.csv`

Aggregate runtime/frontier statistics grouped by stage count, menu size, and gate count after the exact-decimal correction. The corrected rerun used CPython 3.12 on a 24-logical-CPU x86-64 Windows environment. These environment-specific rows describe finite implementation behavior and do not prove complexity or determine allocator exactness.

## `guard_threshold_sensitivity.csv`

Post-hoc policy sensitivity computed from the frozen TDS1 per-repetition lag-one diagnostics. Each row reports one base threshold and DGP, the number/rate of repetitions in which any split/coordinate triggered, unsafe nominal passes, and the unsafe upgrades that would remain after the guard. The actual threshold remains `max(base_threshold, 3/sqrt(n_lag_pairs))`. The file does not alter TD3, estimate a universal operating characteristic, or reclassify physical H3.

## `tds1_summary.csv`

The 17 frozen outer decision cells for iid acquisition, temporal sensitivity, guard detection, and unsafe-upgrade behavior. The result is simulation-specific.
