"""Python reference operator for the periodic service calendar (spec 5.2).

Must stay bit-identical to src/common/calendar.hpp (checked by P0 tests):
same integer arithmetic, same tie-break (start, window_id, cycle_id).

Window instances repeat at k*period_ns + open_ns for integer k, half-open
[open, close). The calendar's eligibility_offset_ns / phase_ns are applied by
the caller by shifting the window bounds before calling this function.
"""

REASON_OK = 0
REASON_INFEASIBLE_NO_WINDOW = 1


def next_feasible_start(eligible_ns, demand_ns, period_ns, windows):
    """Return (start_ns, cycle_id, window_id, reason_code).

    windows: sequence of (open_ns, close_ns) pairs.
    """
    best = None  # (start, window_id, cycle_id)
    for w, (open_ns, close_ns) in enumerate(windows):
        width = close_ns - open_ns
        if width < demand_ns:
            continue
        k = (eligible_ns - open_ns) // period_ns
        wopen = k * period_ns + open_ns
        cand = eligible_ns if eligible_ns > wopen else wopen
        if cand + demand_ns > wopen + width:
            k += 1
            cand = k * period_ns + open_ns
        key = (cand, w, k)
        if best is None or key < best:
            best = key
    if best is None:
        return (0, 0, 0, REASON_INFEASIBLE_NO_WINDOW)
    start_ns, w, k = best
    return (start_ns, k, w, REASON_OK)


def load_calendar(path):
    """Load configs/calendar.yaml; return dict with windows as (open, close) pairs."""
    import yaml

    with open(path) as f:
        cal = yaml.safe_load(f)
    cal["windows"] = [(w["open_ns"], w["close_ns"]) for w in cal["windows"]]
    return cal
