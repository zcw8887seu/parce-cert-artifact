"""Calendar boundary tests (spec 7-P0): exact boundary points for calendar C1."""

from oracle import oracle_next_feasible_start
from parce_exp.calendar_ref import next_feasible_start, REASON_OK

P = 1_000_000
W1 = [(400_000, 700_000)]  # window [open, close), width 300000

# (eligible, demand, windows, expected (start, cycle, window, reason))
CASES = [
    (0, 50_000, W1, (400_000, 0, 0, 0)),
    (399_999, 50_000, W1, (400_000, 0, 0, 0)),          # open - 1
    (400_000, 50_000, W1, (400_000, 0, 0, 0)),          # open
    (350_000, 50_000, W1, (400_000, 0, 0, 0)),          # gap before window
    (649_999, 50_000, W1, (649_999, 0, 0, 0)),          # close - d - 1
    (650_000, 50_000, W1, (650_000, 0, 0, 0)),          # close - d (last feasible)
    (650_001, 50_000, W1, (1_400_000, 1, 0, 0)),        # close - d + 1
    (699_999, 50_000, W1, (1_400_000, 1, 0, 0)),        # close - 1
    (700_000, 50_000, W1, (1_400_000, 1, 0, 0)),        # close
    (1_000_000, 50_000, W1, (1_400_000, 1, 0, 0)),      # period boundary
    (1_400_000, 50_000, W1, (1_400_000, 1, 0, 0)),      # next window open
    (-1, 50_000, W1, (400_000, 0, 0, 0)),               # before first window
    # demand == window width: only s == open fits
    (0, 300_000, W1, (400_000, 0, 0, 0)),
    (400_000, 300_000, W1, (400_000, 0, 0, 0)),
    (400_001, 300_000, W1, (1_400_000, 1, 0, 0)),
    (399_999, 300_000, W1, (400_000, 0, 0, 0)),
    # demand > window width: explicitly infeasible
    (0, 300_001, W1, (0, 0, 0, 1)),
    (400_000, 300_001, W1, (0, 0, 0, 1)),
    (10**15, 300_001, W1, (0, 0, 0, 1)),
    # tie-break: two windows with the same open, identical start
    (0, 50_000, [(400_000, 700_000), (400_000, 800_000)], (400_000, 0, 0, 0)),
    # second window strictly earlier
    (0, 50_000, [(900_000, 950_000), (100_000, 200_000)], (100_000, 0, 1, 0)),
    # large absolute time (checked against the oracle below)
    (10**15, 50_000, W1, None),
    (999_999_999_999_999, 50_000, W1, None),
]


def _expected(eligible, demand, windows):
    for case_elig, case_demand, case_wins, expected in CASES:
        if case_elig == eligible and case_demand == demand and case_wins == windows:
            if expected is None:
                return oracle_next_feasible_start(eligible, demand, P, windows)
            return expected
    raise KeyError((eligible, demand, windows))


def test_boundary_python(rec):
    mismatches = []
    for eligible, demand, windows, _ in CASES:
        got = next_feasible_start(eligible, demand, P, windows)
        expected = _expected(eligible, demand, windows)
        if got != expected:
            mismatches.append((eligible, demand, windows, got, expected))
    rec(
        "calendar_boundary_python",
        not mismatches,
        cases=len(CASES),
        reference_mismatch_count=0,
        oracle_mismatch_count=len(mismatches),
    )
    assert not mismatches, mismatches[:5]


def test_boundary_vs_oracle(rec):
    bad = []
    for eligible, demand, windows, _ in CASES:
        got = next_feasible_start(eligible, demand, P, windows)
        exp = oracle_next_feasible_start(eligible, demand, P, windows)
        if got != exp:
            bad.append((eligible, demand, windows, got, exp))
    rec(
        "calendar_boundary_oracle",
        not bad,
        cases=len(CASES),
        oracle_mismatch_count=len(bad),
    )
    assert not bad, bad[:5]


def test_boundary_cpp(calcheck, rec):
    cases = [(eligible, demand, P, windows) for eligible, demand, windows, _ in CASES]
    rows = calcheck(cases)
    mismatches = []
    for (eligible, demand, windows, _), cpp in zip(CASES, rows):
        py = next_feasible_start(eligible, demand, P, windows)
        if py != cpp:
            mismatches.append((eligible, demand, py, cpp))
    rec(
        "calendar_boundary_cpp",
        not mismatches,
        cases=len(CASES),
        reference_mismatch_count=len(mismatches),
    )
    assert not mismatches, mismatches[:5]
