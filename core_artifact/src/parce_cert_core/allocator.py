"""Exact finite-decimal Pareto-frontier allocator for a serial chain."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import product

from .calendar import REASON_OK, next_feasible_start


class GateInfeasible(Exception):
    """Raised when the demand fits no periodic window."""


@dataclass(frozen=True)
class MenuItem:
    menu_id: str
    risk: object
    q_ns: int


@dataclass(frozen=True)
class Selection:
    bound_ns: int
    risk_sum: float
    menu_ids: tuple[str, ...]


def _risk_decimal(value):
    """Decode a public risk value according to finite-decimal semantics."""
    try:
        value_decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"risk must be a finite decimal, got {value!r}") from exc
    if not value_decimal.is_finite() or value_decimal < 0:
        raise ValueError(f"risk must be a nonnegative finite decimal, got {value!r}")
    return value_decimal


def _exact_risk_ticks(menus, risk_budget):
    """Map every input decimal to an instance-derived common integer scale."""
    budget_decimal = _risk_decimal(risk_budget)
    risk_decimals = [[_risk_decimal(item.risk) for item in menu]
                     for menu in menus]
    values = [budget_decimal]
    values.extend(value for menu in risk_decimals for value in menu)
    decimal_places = max(0, max(-value.as_tuple().exponent for value in values))

    def ticks(value):
        scaled = value.scaleb(decimal_places)
        integral = scaled.to_integral_value()
        if scaled != integral:
            raise ValueError(f"cannot encode risk exactly: {value}")
        return int(integral)

    return (
        [[ticks(value) for value in menu] for menu in risk_decimals],
        ticks(budget_decimal),
        decimal_places,
    )


def _ticks_to_float(ticks, decimal_places):
    return float(Decimal(ticks).scaleb(-decimal_places))


def _validate_allocator_inputs(menus, gate_after, coordinate_ids):
    if len(menus) != len(gate_after):
        raise ValueError("gate_after must contain one Boolean per stage")
    if any(not menu for menu in menus):
        raise ValueError("every stage must have at least one menu item")
    if coordinate_ids is None:
        return
    if len(coordinate_ids) != len(menus):
        raise ValueError("coordinate_ids must contain one identifier per stage")
    if len(set(coordinate_ids)) != len(coordinate_ids):
        raise ValueError(
            "repeated/shared coordinates require an augmented retained-state "
            "allocator; the memoryless serial-chain DP rejects them"
        )


def gate_completion_offset(arrival_offset, demand_ns, period_ns, windows, base_ns=0):
    start, _cycle, _window, reason = next_feasible_start(
        base_ns + arrival_offset, demand_ns, period_ns, windows
    )
    if reason != REASON_OK:
        raise GateInfeasible("demand does not fit any periodic window")
    return start - base_ns + demand_ns


def chain_completion(qs, gate_after, demand_ns, period_ns, windows, base_ns=0):
    completion = 0
    for stage, q_ns in enumerate(qs):
        completion += int(q_ns)
        if gate_after[stage]:
            completion = gate_completion_offset(
                completion, demand_ns, period_ns, windows, base_ns
            )
    return completion


def _prune_states(states):
    states = sorted(set(states), key=lambda state: (state[0], state[1], state[2]))
    kept = []
    best_time_at_lower_risk = None
    index = 0
    while index < len(states):
        risk = states[index][0]
        end = index
        while end < len(states) and states[end][0] == risk:
            end += 1
        best_ids_at_same_risk = None
        minimum_kept_time = None
        for risk_b, time_b, ids_b in states[index:end]:
            lower_risk_dominates = (
                best_time_at_lower_risk is not None
                and best_time_at_lower_risk <= time_b
            )
            same_risk_dominates = (
                best_ids_at_same_risk is not None
                and best_ids_at_same_risk <= ids_b
            )
            if lower_risk_dominates or same_risk_dominates:
                continue
            kept.append((risk_b, time_b, ids_b))
            if best_ids_at_same_risk is None or ids_b < best_ids_at_same_risk:
                best_ids_at_same_risk = ids_b
            if minimum_kept_time is None or time_b < minimum_kept_time:
                minimum_kept_time = time_b
        if minimum_kept_time is not None and (
            best_time_at_lower_risk is None
            or minimum_kept_time < best_time_at_lower_risk
        ):
            best_time_at_lower_risk = minimum_kept_time
        index = end
    return kept


def pareto_dp(menus, gate_after, demand_ns, period_ns, windows, risk_budget,
              base_ns=0, coordinate_ids=None):
    """Minimize (bound, exact risk sum, menu IDs) for a memoryless chain.

    Floats are interpreted through their shortest decimal spelling; callers
    may pass strings or ``Decimal`` values to preserve source tokens exactly.
    The integer scale is inferred from each instance, not fixed in advance.
    Explicit repeated coordinate IDs are rejected because once-only charging
    needs an augmented retained-state allocator.
    """
    _validate_allocator_inputs(menus, gate_after, coordinate_ids)
    risk_ticks, budget_ticks, decimal_places = _exact_risk_ticks(
        menus, risk_budget)
    states = [(0, 0, ())]
    for stage, menu in enumerate(menus):
        candidates = []
        for risk, completion, ids in states:
            for item_index, item in enumerate(menu):
                new_risk = risk + risk_ticks[stage][item_index]
                if new_risk > budget_ticks:
                    continue
                new_completion = completion + item.q_ns
                if gate_after[stage]:
                    new_completion = gate_completion_offset(
                        new_completion, demand_ns, period_ns, windows, base_ns
                    )
                candidates.append(
                    (new_risk, new_completion, ids + (item.menu_id,))
                )
        states = _prune_states(candidates)
        if not states:
            return None
    risk, completion, ids = min(states, key=lambda state: (state[1], state[0], state[2]))
    return Selection(int(completion),
                     _ticks_to_float(risk, decimal_places), ids)


def brute_force_selection(
    menus, gate_after, demand_ns, period_ns, windows, risk_budget, base_ns=0,
    coordinate_ids=None,
):
    """Independent Decimal exhaustive oracle for small allocator instances."""
    _validate_allocator_inputs(menus, gate_after, coordinate_ids)
    budget_decimal = _risk_decimal(risk_budget)
    feasible = []
    for choice in product(*menus):
        # Deliberately use Decimal addition here rather than the DP's integer
        # tick encoder, so exhaustive checks do not share feasibility arithmetic.
        risk = sum((_risk_decimal(item.risk) for item in choice), Decimal(0))
        if risk > budget_decimal:
            continue
        bound = chain_completion(
            [item.q_ns for item in choice],
            gate_after,
            demand_ns,
            period_ns,
            windows,
            base_ns,
        )
        ids = tuple(item.menu_id for item in choice)
        feasible.append((bound, risk, ids))
    if not feasible:
        return None
    bound, risk, ids = min(feasible, key=lambda item: (item[0], item[1], item[2]))
    return Selection(int(bound), float(risk), ids)
