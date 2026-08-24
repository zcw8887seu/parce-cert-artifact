"""Minimal exact Pareto-frontier allocator for a serial PARCE-Cert chain."""

from dataclasses import dataclass
from itertools import product

from .calendar import REASON_OK, next_feasible_start


class GateInfeasible(Exception):
    """Raised when the demand fits no periodic window."""


@dataclass(frozen=True)
class MenuItem:
    menu_id: str
    risk: float
    q_ns: int


@dataclass(frozen=True)
class Selection:
    bound_ns: int
    risk_sum: float
    menu_ids: tuple[str, ...]


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
    for risk_b, time_b, ids_b in states:
        dominated = any(
            risk_a <= risk_b
            and time_a <= time_b
            and (risk_a < risk_b or ids_a < ids_b)
            for risk_a, time_a, ids_a in kept
        )
        if not dominated:
            kept.append((risk_b, time_b, ids_b))
    return kept


def pareto_dp(menus, gate_after, demand_ns, period_ns, windows, risk_budget, base_ns=0):
    """Select one row per stage and minimize bound, risk, then menu-id order."""
    if len(menus) != len(gate_after):
        raise ValueError("gate_after must contain one Boolean per stage")
    states = [(0.0, 0, ())]
    for stage, menu in enumerate(menus):
        candidates = []
        for risk, completion, ids in states:
            for item in menu:
                new_risk = risk + item.risk
                if new_risk > risk_budget:
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
    return Selection(int(completion), float(risk), ids)


def brute_force_selection(
    menus, gate_after, demand_ns, period_ns, windows, risk_budget, base_ns=0
):
    """Small-instance oracle used to check the exact frontier allocator."""
    feasible = []
    for choice in product(*menus):
        risk = sum(item.risk for item in choice)
        if risk > risk_budget:
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
        feasible.append(Selection(bound, float(risk), ids))
    if not feasible:
        return None
    return min(feasible, key=lambda item: (item.bound_ns, item.risk_sum, item.menu_ids))
