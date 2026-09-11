"""Test-side exhaustive allocator oracle with independent exact arithmetic.

This module intentionally does not import allocator risk-normalization or gate
vectorization helpers.  It parses finite decimals, creates its own integer
scale, enumerates every menu combination, and applies an independent periodic
window routine.  The production DP and this oracle therefore share the public
problem contract, not a binary-float feasibility implementation.
"""

from decimal import Decimal

import numpy as np


def _decimal(value):
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("oracle risks must be nonnegative finite decimals")
    return result


def _integerize(menus, budget):
    budget_decimal = _decimal(budget)
    risks = [[_decimal(item.risk) for item in menu] for menu in menus]
    values = [budget_decimal]
    values.extend(value for menu in risks for value in menu)
    places = max(0, max(-value.as_tuple().exponent for value in values))

    def as_int(value):
        return int(value.scaleb(places).to_integral_exact())

    return [[as_int(value) for value in menu] for menu in risks], \
        as_int(budget_decimal), places


def _gate_vec(eligible, demand_ns, period_ns, windows, base_ns):
    absolute = np.asarray(eligible, dtype=np.int64) + np.int64(base_ns)
    candidates = []
    for open_ns, close_ns in windows:
        if close_ns - open_ns < demand_ns:
            continue
        cycle = (absolute - open_ns) // period_ns
        window_open = cycle * period_ns + open_ns
        start = np.maximum(absolute, window_open)
        late = start + demand_ns > cycle * period_ns + close_ns
        start = np.where(late, (cycle + 1) * period_ns + open_ns, start)
        candidates.append(start)
    if not candidates:
        raise ValueError("demand fits no periodic window")
    return np.minimum.reduce(candidates) - np.int64(base_ns) + demand_ns


def exhaustive_exact_selection(
    menus,
    gate_after,
    demand_ns,
    period_ns,
    windows,
    risk_budget,
    base_ns=0,
    gate_ops=None,
):
    """Return ``(bound_ns, exact_risk_decimal, menu_ids)`` or ``None``."""
    if len(menus) != len(gate_after) or any(not menu for menu in menus):
        raise ValueError("invalid finite serial-chain instance")
    risk_ticks, budget_ticks, places = _integerize(menus, risk_budget)
    sizes = tuple(len(menu) for menu in menus)
    indices = np.indices(sizes, dtype=np.int64).reshape(len(menus), -1)
    combinations = indices.shape[1]
    completion = np.zeros(combinations, dtype=np.int64)
    max_risk = sum(max(stage) for stage in risk_ticks)
    risk_dtype = (np.int64 if max(max_risk, budget_ticks)
                  <= np.iinfo(np.int64).max else object)
    risk = np.zeros(combinations, dtype=risk_dtype)
    lex_code = np.zeros(combinations, dtype=np.int64)

    for stage, menu in enumerate(menus):
        order = sorted(range(len(menu)), key=lambda item: menu[item].menu_id)
        rank = np.empty(len(menu), dtype=np.int64)
        for position, item in enumerate(order):
            rank[item] = position
        selected = indices[stage]
        completion += np.array([item.q_ns for item in menu],
                               dtype=np.int64)[selected]
        risk += np.array(risk_ticks[stage], dtype=risk_dtype)[selected]
        lex_code = lex_code * len(menu) + rank[selected]
        if gate_after[stage]:
            gate_fn = (gate_ops[stage] if gate_ops is not None
                       and stage < len(gate_ops) else None)
            if gate_fn is None:
                completion = _gate_vec(completion, demand_ns, period_ns,
                                       windows, base_ns)
            else:
                completion = np.array(
                    [int(gate_fn(int(value))) for value in completion],
                    dtype=np.int64,
                )

    feasible = np.flatnonzero(risk <= budget_ticks)
    if not len(feasible):
        return None
    if risk_dtype is object:
        selected_combo = min(
            (int(index) for index in feasible),
            key=lambda index: (
                int(completion[index]), int(risk[index]), int(lex_code[index])
            ),
        )
    else:
        selected_combo = int(feasible[int(np.lexsort((
            lex_code[feasible], risk[feasible], completion[feasible]
        ))[0])])
    ids = tuple(
        menus[stage][int(indices[stage, selected_combo])].menu_id
        for stage in range(len(menus))
    )
    risk_decimal = Decimal(int(risk[selected_combo])).scaleb(-places)
    return int(completion[selected_combo]), risk_decimal, ids
