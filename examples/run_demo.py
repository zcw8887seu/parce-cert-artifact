"""Run the three compact PARCE-Cert demonstrations."""

from parce_cert_core import (
    MenuItem,
    brute_force_selection,
    clopper_pearson,
    next_feasible_start,
    pareto_dp,
    required_order_statistic,
    validation_verdict,
)


def main():
    period = 1_000_000
    windows = [(400_000, 700_000)]
    demand = 50_000

    before = next_feasible_start(650_000, demand, period, windows)
    after = next_feasible_start(650_001, demand, period, windows)
    print("Gate boundary:", before, "->", after)

    alpha_row = 0.05 / 12
    orders = {
        epsilon: required_order_statistic(3000, epsilon, alpha_row)
        for epsilon in (0.002, 0.003, 0.004)
    }
    print("Order-statistic menu:", orders)

    lower, upper = clopper_pearson(1, 3500, 0.01)
    print("Validation interval:", (lower, upper), validation_verdict(lower, upper, 0.003))

    menus = [
        [MenuItem("X1@strict", 0.002, 100_000), MenuItem("X1@loose", 0.004, 70_000)],
        [MenuItem("X2@strict", 0.002, 620_000), MenuItem("X2@loose", 0.004, 500_000)],
        [MenuItem("X3@strict", 0.002, 90_000), MenuItem("X3@loose", 0.004, 60_000)],
    ]
    gate_after = [False, True, False]
    dp = pareto_dp(menus, gate_after, demand, period, windows, 0.010)
    oracle = brute_force_selection(menus, gate_after, demand, period, windows, 0.010)
    print("Exact allocation:", dp)
    assert dp == oracle


if __name__ == "__main__":
    main()
