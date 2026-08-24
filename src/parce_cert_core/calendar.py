"""Integer reference operator for fit-within-one-window periodic service."""

REASON_OK = 0
REASON_INFEASIBLE_NO_WINDOW = 1


def next_feasible_start(eligible_ns, demand_ns, period_ns, windows):
    """Return ``(start_ns, cycle_id, window_id, reason_code)``.

    ``windows`` is a sequence of half-open ``(open_ns, close_ns)`` pairs.
    A demand is admitted only when it fits completely in one window.
    """
    if period_ns <= 0 or demand_ns < 0:
        raise ValueError("period_ns must be positive and demand_ns nonnegative")

    best = None
    for window_id, (open_ns, close_ns) in enumerate(windows):
        width = close_ns - open_ns
        if width < demand_ns:
            continue
        cycle_id = (eligible_ns - open_ns) // period_ns
        window_open = cycle_id * period_ns + open_ns
        candidate = max(eligible_ns, window_open)
        if candidate + demand_ns > window_open + width:
            cycle_id += 1
            candidate = cycle_id * period_ns + open_ns
        key = (candidate, window_id, cycle_id)
        if best is None or key < best:
            best = key

    if best is None:
        return 0, 0, 0, REASON_INFEASIBLE_NO_WINDOW

    start_ns, window_id, cycle_id = best
    return start_ns, cycle_id, window_id, REASON_OK
