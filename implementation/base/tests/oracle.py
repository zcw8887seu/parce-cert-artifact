"""Independent slow oracle for next_feasible_start (test-side only).

Scans a candidate set instead of using the closed-form jump, so it is a
genuinely separate implementation from both calendar_ref.py and calendar.hpp.
"""


def _fits(s, demand, period, windows):
    for w, (open_ns, close_ns) in enumerate(windows):
        if close_ns - open_ns < demand:
            continue
        k = (s - open_ns) // period
        if k * period + open_ns <= s and s + demand <= k * period + close_ns:
            return w, k
    return None


def oracle_next_feasible_start(eligible, demand, period, windows):
    if all(close_ns - open_ns < demand for open_ns, close_ns in windows):
        return (0, 0, 0, 1)
    candidates = {eligible}
    for open_ns, close_ns in windows:
        k_lo = (eligible - close_ns + demand) // period - 1
        k_hi = (eligible - open_ns) // period + 2
        for k in range(k_lo, k_hi + 1):
            candidates.add(k * period + open_ns)
    feasible = []
    for s in sorted(candidates):
        if s < eligible:
            continue
        hit = _fits(s, demand, period, windows)
        if hit is not None:
            w, k = hit
            feasible.append((s, w, k))
    s, w, k = min(feasible)
    return (s, k, w, 0)
