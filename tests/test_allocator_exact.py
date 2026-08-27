from decimal import Decimal

import pytest

from parce_cert_core import MenuItem, brute_force_selection, pareto_dp


PERIOD = 1_000_000
WINDOWS = [(400_000, 700_000)]
DEMAND = 50_000


def test_decimal_budget_boundary_is_exact():
    menus = [
        [MenuItem("a", 0.1, 10)],
        [MenuItem("b", 0.2, 20)],
    ]
    gates = [False, False]
    dp = pareto_dp(menus, gates, DEMAND, PERIOD, WINDOWS, 0.3)
    oracle = brute_force_selection(
        menus, gates, DEMAND, PERIOD, WINDOWS, 0.3)
    assert dp == oracle
    assert dp is not None
    assert (dp.bound_ns, dp.risk_sum, dp.menu_ids) == (
        30, 0.3, ("a", "b"))


def test_scale_is_derived_from_all_decimal_tokens():
    menus = [
        [MenuItem("a", "0.10000000000000000001", 10)],
        [MenuItem("b", Decimal("0.20000000000000000002"), 20)],
    ]
    budget = "0.30000000000000000003"
    dp = pareto_dp(
        menus, [False, False], DEMAND, PERIOD, WINDOWS, budget)
    oracle = brute_force_selection(
        menus, [False, False], DEMAND, PERIOD, WINDOWS, budget)
    assert dp == oracle
    assert dp is not None


def test_gate_plateau_keeps_lexicographic_winner():
    menus = [
        [MenuItem("z", "0.1", 0), MenuItem("a", "0.1", 1)],
        [MenuItem("x", "0", 0)],
    ]
    gates = [False, True]
    dp = pareto_dp(menus, gates, DEMAND, PERIOD, WINDOWS, "0.1")
    oracle = brute_force_selection(
        menus, gates, DEMAND, PERIOD, WINDOWS, "0.1")
    assert dp == oracle
    assert dp is not None
    assert (dp.bound_ns, dp.menu_ids) == (450_000, ("a", "x"))


@pytest.mark.parametrize("allocator", [pareto_dp, brute_force_selection])
def test_repeated_coordinate_is_rejected_fail_closed(allocator):
    menus = [
        [MenuItem("x", "0.1", 1)],
        [MenuItem("x-reuse", "0", 1)],
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
