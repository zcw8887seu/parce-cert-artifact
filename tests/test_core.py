from parce_cert_core import (
    InfeasibleSampleSize,
    MenuItem,
    brute_force_selection,
    clopper_pearson,
    next_feasible_start,
    pareto_dp,
    required_order_statistic,
    validation_verdict,
)


PERIOD = 1_000_000
WINDOWS = [(400_000, 700_000)]
DEMAND = 50_000


def test_gate_last_feasible_start():
    assert next_feasible_start(650_000, DEMAND, PERIOD, WINDOWS) == (650_000, 0, 0, 0)
    assert next_feasible_start(650_001, DEMAND, PERIOD, WINDOWS) == (1_400_000, 1, 0, 0)


def test_order_statistic_table():
    alpha_row = 0.05 / 12
    assert required_order_statistic(3000, 0.002, alpha_row) == 3000
    assert required_order_statistic(3000, 0.003, alpha_row) == 2999
    assert required_order_statistic(3000, 0.004, alpha_row) == 2997
    try:
        required_order_statistic(10, 0.002, alpha_row)
        raise AssertionError("expected InfeasibleSampleSize")
    except InfeasibleSampleSize:
        pass


def test_validation_outcomes():
    lower, upper = clopper_pearson(0, 3500, 0.01)
    assert validation_verdict(lower, upper, 0.002) == "SUPPORTED"
    assert validation_verdict(0.005, 0.006, 0.004) == "NOT_SUPPORTED"
    assert validation_verdict(0.001, 0.005, 0.004) == "INCONCLUSIVE"


def test_pareto_matches_small_oracle():
    menus = [
        [MenuItem("a0", 0.002, 100_000), MenuItem("a1", 0.004, 70_000)],
        [MenuItem("b0", 0.002, 620_000), MenuItem("b1", 0.004, 500_000)],
        [MenuItem("c0", 0.002, 90_000), MenuItem("c1", 0.004, 60_000)],
    ]
    gates = [False, True, False]
    dp = pareto_dp(menus, gates, DEMAND, PERIOD, WINDOWS, 0.010)
    oracle = brute_force_selection(menus, gates, DEMAND, PERIOD, WINDOWS, 0.010)
    assert dp == oracle
