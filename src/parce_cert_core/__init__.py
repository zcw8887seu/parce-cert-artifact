"""Core, host-independent PARCE-Cert reference functions."""

from .allocator import MenuItem, Selection, brute_force_selection, pareto_dp
from .calendar import REASON_INFEASIBLE_NO_WINDOW, REASON_OK, next_feasible_start
from .statistics import (
    InfeasibleSampleSize,
    clopper_pearson,
    required_order_statistic,
    validation_verdict,
)

__all__ = [
    "InfeasibleSampleSize",
    "MenuItem",
    "REASON_INFEASIBLE_NO_WINDOW",
    "REASON_OK",
    "Selection",
    "brute_force_selection",
    "clopper_pearson",
    "next_feasible_start",
    "pareto_dp",
    "required_order_statistic",
    "validation_verdict",
]
