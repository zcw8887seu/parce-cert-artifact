"""10,000 random calendar property tests (spec 7-P0).

Properties per case: output feasible, output earliest (vs independent oracle),
output non-decreasing when eligible increases, Python == C++.
"""

import numpy as np

from oracle import oracle_next_feasible_start
from parce_exp.calendar_ref import next_feasible_start

N_CASES = 10_000
SEED = np.random.SeedSequence(entropy=20260816, spawn_key=(0,))


def _gen_case(rng):
    period = int(rng.integers(200, 5000)) * 1000
    demand = int(rng.integers(1, 400_000))
    n_windows = int(rng.integers(1, 4))
    windows = []
    for _ in range(n_windows):
        open_ns = int(rng.integers(-period, 2 * period))
        width = int(rng.integers(1000, period + 100_000))
        windows.append((open_ns, open_ns + width))
    u = rng.random()
    if u < 0.6:
        eligible = int(rng.integers(-2 * period, 3 * period))
    elif u < 0.8:
        eligible = int(rng.integers(0, 10**15))
    else:
        open_ns, close_ns = windows[int(rng.integers(0, len(windows)))]
        anchor = int(rng.choice([open_ns, close_ns, close_ns - demand]))
        eligible = anchor + int(rng.integers(-5, 6))
    return eligible, demand, period, windows


def test_calendar_properties(calcheck, rec):
    rng = np.random.default_rng(SEED)
    py_results = []
    cases = []
    feasible_violations = 0
    oracle_mismatches = 0
    monotonic_violations = 0
    infeasible_cases = 0

    for _ in range(N_CASES):
        eligible, demand, period, windows = _gen_case(rng)
        res = next_feasible_start(eligible, demand, period, windows)

        if res[3] == 0:
            start, cycle, window, _ = res
            open_ns, close_ns = windows[window]
            wopen = cycle * period + open_ns
            ok = (
                start >= eligible
                and start >= wopen
                and start + demand <= wopen + (close_ns - open_ns)
            )
            if not ok:
                feasible_violations += 1
        else:
            infeasible_cases += 1
            if any(close_ns - open_ns >= demand for open_ns, close_ns in windows):
                feasible_violations += 1

        if res != oracle_next_feasible_start(eligible, demand, period, windows):
            oracle_mismatches += 1

        eligible2 = eligible + int(rng.integers(0, 3 * period))
        res2 = next_feasible_start(eligible2, demand, period, windows)
        if res2[0] < res[0]:
            monotonic_violations += 1

        py_results.extend([res, res2])
        cases.append((eligible, demand, period, windows))
        cases.append((eligible2, demand, period, windows))

    cpp_rows = calcheck(cases)
    reference_mismatches = sum(1 for a, b in zip(py_results, cpp_rows) if a != b)

    rec(
        "calendar_property",
        feasible_violations == 0
        and oracle_mismatches == 0
        and monotonic_violations == 0
        and reference_mismatches == 0,
        cases=N_CASES,
        reference_mismatch_count=reference_mismatches,
        oracle_mismatch_count=oracle_mismatches,
        feasibility_violation_count=feasible_violations,
        monotonicity_violation_count=monotonic_violations,
        infeasible_case_count=infeasible_cases,
    )
    assert feasible_violations == 0
    assert oracle_mismatches == 0
    assert monotonic_violations == 0
    assert reference_mismatches == 0
