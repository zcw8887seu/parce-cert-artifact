# Aggregate Data Dictionary

All files in `data/` are curated publication tables. They do not contain raw events or machine-specific metadata.

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

Aggregate runtime/frontier statistics grouped by stage count, menu size, and gate count. These rows describe finite implementation behavior and do not prove complexity.

## `tds1_summary.csv`

The 17 frozen outer decision cells for iid acquisition, temporal sensitivity, guard detection, and unsafe-upgrade behavior. The result is simulation-specific.
