"""SG-01 regressions for exact risk semantics and DP state sufficiency."""

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allocator_exact_oracle import exhaustive_exact_selection
from parce_exp.allocator import (
    MenuItem,
    _prune_states,
    brute_force_selection,
    pareto_dp,
    spec56_gate_ops,
)


PERIOD = 1_000_000
WINDOWS = [(400_000, 700_000)]
DEMAND = 50_000
ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parents[3] / "data/e01/derived"


def _key(selection):
    return None if selection is None else (selection.bound_ns, selection.menu_ids)


def test_frontier_pruning_matches_tie_safe_dominance_definition():
    rng = np.random.default_rng(20260826)
    for _ in range(200):
        states = [
            (
                int(rng.integers(0, 6)),
                int(rng.integers(0, 12)),
                tuple(chr(97 + int(value)) for value in rng.integers(0, 5, 3)),
            )
            for _ in range(80)
        ]
        unique = sorted(set(states), key=lambda state: (
            state[0], state[1], state[2]))
        expected = []
        for candidate in unique:
            dominated = any(
                other != candidate
                and other[0] <= candidate[0]
                and other[1] <= candidate[1]
                and (other[0] < candidate[0] or other[2] <= candidate[2])
                and (
                    other[0] < candidate[0]
                    or other[1] < candidate[1]
                    or other[2] < candidate[2]
                )
                for other in unique
            )
            if not dominated:
                expected.append(candidate)
        assert _prune_states(states) == expected


def test_decimal_boundary_0_1_plus_0_2_is_feasible():
    menus = [
        [MenuItem("a", 0.1, 10)],
        [MenuItem("b", 0.2, 20)],
    ]
    gates = [False, False]
    expected = (30, ("a", "b"))
    assert _key(pareto_dp(
        menus, gates, DEMAND, PERIOD, WINDOWS, 0.3)) == expected
    assert _key(brute_force_selection(
        menus, gates, DEMAND, PERIOD, WINDOWS, 0.3)) == expected


def test_instance_derived_ticks_accept_arbitrary_finite_decimal_precision():
    menus = [
        [MenuItem("a", "0.10000000000000000001", 10)],
        [MenuItem("b", Decimal("0.20000000000000000002"), 20)],
    ]
    budget = "0.30000000000000000003"
    gates = [False, False]
    expected = (30, ("a", "b"))
    assert _key(pareto_dp(
        menus, gates, DEMAND, PERIOD, WINDOWS, budget)) == expected
    assert _key(brute_force_selection(
        menus, gates, DEMAND, PERIOD, WINDOWS, budget)) == expected


def test_gate_plateau_preserves_lexicographically_preferred_prefix():
    menus = [
        [
            MenuItem("z", "0.1", 0),
            MenuItem("a", "0.1", 1),
        ],
        [MenuItem("x", "0.0", 0)],
    ]
    gates = [False, True]
    dp = pareto_dp(menus, gates, DEMAND, PERIOD, WINDOWS, "0.1")
    oracle = exhaustive_exact_selection(
        menus, gates, DEMAND, PERIOD, WINDOWS, "0.1")
    assert dp is not None
    assert (dp.bound_ns, dp.menu_ids) == (450_000, ("a", "x"))
    assert (dp.bound_ns, dp.menu_ids) == (oracle[0], oracle[2])


@pytest.mark.parametrize("allocator", [pareto_dp, brute_force_selection])
def test_repeated_coordinate_requires_augmented_state(allocator):
    menus = [
        [MenuItem("x-low", "0.1", 1)],
        [MenuItem("x-reuse", "0.0", 1)],
    ]
    with pytest.raises(ValueError, match="repeated/shared coordinates"):
        allocator(
            menus,
            [False, False],
            DEMAND,
            PERIOD,
            WINDOWS,
            "0.1",
            coordinate_ids=("X", "X"),
        )


def _random_instance(rng, tie_heavy):
    if tie_heavy:
        stages = int(rng.integers(2, 5))
        menu_size = int(rng.integers(2, 4))
        risk_pool = ["0.001", "0.002", "0.003"]
        q_pool = [30_000, 60_000, 90_000]
        budget = "0.006"
    else:
        stages = int(rng.integers(2, 9))
        menu_size = int(rng.integers(2, 7))
        risk_pool = None
        q_pool = None
        budget = ("0.010" if rng.random() < 0.5 else
                  str(float(rng.uniform(0.0008 * stages, 0.004 * stages))))
    menus = []
    for stage in range(stages):
        items = []
        for row in range(menu_size):
            risk = (str(rng.choice(risk_pool)) if risk_pool else
                    str(float(rng.uniform(0.0005, 0.004))))
            q_ns = (int(rng.choice(q_pool)) if q_pool else
                    int(rng.integers(1, 201)) * 1000)
            items.append(MenuItem(f"m{stage}_{row}", risk, q_ns))
        menus.append(items)
    gate_count = int(rng.integers(0, min(3, stages - 1) + 1))
    positions = set(rng.choice(
        stages - 1, size=gate_count, replace=False).tolist())
    gates = [stage in positions for stage in range(stages)]
    gates[-1] = False
    return menus, gates, budget


def test_1000_random_chains_against_independent_exact_oracle():
    rng = np.random.default_rng(np.random.SeedSequence(
        entropy=20260816, spawn_key=(2,)))
    for case in range(1000):
        menus, gates, budget = _random_instance(rng, case < 150)
        dp = pareto_dp(menus, gates, DEMAND, PERIOD, WINDOWS, budget)
        oracle = exhaustive_exact_selection(
            menus, gates, DEMAND, PERIOD, WINDOWS, budget)
        oracle_key = None if oracle is None else (oracle[0], oracle[2])
        assert _key(dp) == oracle_key, case


def test_81_real_menu_combinations_against_independent_exact_oracle():
    menu_frame = pd.read_parquet(RESULTS_ROOT / "menu.parquet")
    design = json.loads(
        (RESULTS_ROOT / "selected_design.json").read_text())
    coordinates = ["X1", "X2", "X3", "X4"]
    menus = []
    for coordinate in coordinates:
        rows = menu_frame[
            (menu_frame["coordinate_id"] == coordinate)
            & (menu_frame["status"] == "CERTIFIED")
            & (~menu_frame["q_is_infinite"].astype(bool))
        ]
        menus.append([
            MenuItem(str(row.menu_id), str(row.risk), int(row.q_ns))
            for row in rows.itertuples()
        ])
    assert np.prod([len(menu) for menu in menus]) == 81
    gates = [False, True, True, False]
    phase = int(design["phase_ns"])
    gate_ops = spec56_gate_ops(
        len(menus), gates, DEMAND, PERIOD, WINDOWS, phase)
    dp = pareto_dp(
        menus, gates, DEMAND, PERIOD, WINDOWS, "0.010",
        base_ns=phase, gate_ops=gate_ops)
    oracle = exhaustive_exact_selection(
        menus, gates, DEMAND, PERIOD, WINDOWS, "0.010",
        base_ns=phase, gate_ops=gate_ops)
    assert _key(dp) == (oracle[0], oracle[2])
    assert _key(dp) == (
        int(design["bound_ns"]),
        tuple(design["selected_menu_ids"][coordinate]
              for coordinate in coordinates),
    )
