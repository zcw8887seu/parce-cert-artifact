"""1,000 random serial chains: Pareto DP vs exhaustive enumeration (spec 7-P0).

Match must be exact on feasibility, optimal bound, risk sum and the full
tie-break (min bound -> min risk sum -> lexicographic menu ids). Half the
instances are tie-heavy (duplicated risks and q values) to stress the
tie-break; the rest use continuous random risks and larger sizes.
"""

from decimal import Decimal

import numpy as np

from allocator_exact_oracle import exhaustive_exact_selection
from parce_exp.allocator import (
    MenuItem,
    chain_completion,
    gate_completion_offset,
    gate_completion_offset_vec,
    pareto_dp,
)

N_CHAINS = 1000
SEED = np.random.SeedSequence(entropy=20260816, spawn_key=(2,))
P = 1_000_000
W1 = [(400_000, 700_000)]
DEMAND = 50_000


def _gen_instance(rng, tie_heavy):
    if tie_heavy:
        n_stages = int(rng.integers(2, 5))
        menu_size = int(rng.integers(2, 4))
        risk_pool = [0.001, 0.002, 0.003]
        q_pool = [30_000, 60_000, 90_000]
        budget = 0.006
    else:
        n_stages = int(rng.integers(2, 9))
        menu_size = int(rng.integers(2, 7))
        risk_pool = None
        q_pool = None
        budget = 0.010 if rng.random() < 0.5 else float(rng.uniform(0.0008 * n_stages, 0.004 * n_stages))
    menus = []
    for j in range(n_stages):
        items = []
        for i in range(menu_size):
            risk = float(rng.choice(risk_pool)) if risk_pool else float(rng.uniform(0.0005, 0.004))
            q = int(rng.choice(q_pool)) if q_pool else int(rng.integers(1, 201)) * 1000
            items.append(MenuItem(menu_id=f"m{j}_{i}", risk=risk, q_ns=q))
        menus.append(items)
    n_gates = int(rng.integers(0, min(3, n_stages - 1) + 1))
    gate_positions = set(rng.choice(n_stages - 1, size=n_gates, replace=False).tolist())
    gate_after = [j in gate_positions for j in range(n_stages)]
    gate_after[-1] = False
    return menus, gate_after, budget


def _sel_tuple(sel):
    if sel is None:
        return None
    return (sel.bound_ns, sel.menu_ids)


def test_gate_vec_matches_scalar():
    rng = np.random.default_rng(np.random.SeedSequence(entropy=20260816, spawn_key=(3,)))
    arrivals = rng.integers(0, 4_000_000, size=2000, dtype=np.int64)
    vec = gate_completion_offset_vec(arrivals, DEMAND, P, W1, 0)
    scal = np.array([gate_completion_offset(int(a), DEMAND, P, W1, 0) for a in arrivals])
    assert bool((vec == scal).all())


def test_dp_vs_brute(rec):
    rng = np.random.default_rng(SEED)
    mismatches = []
    tie_instances = 0
    infeasible_instances = 0
    feasible_instances = 0

    for i in range(N_CHAINS):
        tie_heavy = i < 150
        menus, gate_after, budget = _gen_instance(rng, tie_heavy)
        if tie_heavy:
            tie_instances += 1
        dp = pareto_dp(menus, gate_after, DEMAND, P, W1, budget, 0)
        bf = exhaustive_exact_selection(
            menus, gate_after, DEMAND, P, W1, budget, 0)
        bf_key = None if bf is None else (bf[0], bf[2])

        if _sel_tuple(dp) != bf_key:
            mismatches.append((i, _sel_tuple(dp), bf_key))
            continue
        if dp is not None:
            feasible_instances += 1
            q_by_id = [
                {it.menu_id: it for it in menu} for menu in menus
            ]
            qs = [q_by_id[j][mid].q_ns for j, mid in enumerate(dp.menu_ids)]
            exact_risk = sum(
                (Decimal(str(q_by_id[j][mid].risk))
                 for j, mid in enumerate(dp.menu_ids)),
                Decimal(0),
            )
            if exact_risk != bf[1]:
                mismatches.append((i, "exact-risk", exact_risk, bf[1]))
            recomputed = chain_completion(qs, gate_after, DEMAND, P, W1, 0)
            if recomputed != dp.bound_ns:
                mismatches.append((i, "recompute", recomputed, dp.bound_ns))
        else:
            infeasible_instances += 1

    rec(
        "dp_vs_brute",
        not mismatches,
        cases=N_CHAINS,
        dp_brute_mismatch_count=len(mismatches),
        tie_instances=tie_instances,
        infeasible_instances=infeasible_instances,
        feasible_instances=feasible_instances,
    )
    assert not mismatches, mismatches[:3]
