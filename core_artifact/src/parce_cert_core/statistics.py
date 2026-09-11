"""Finite-sample acquisition and validation utilities used by PARCE-Cert."""

from scipy.stats import beta, binom


class InfeasibleSampleSize(Exception):
    """Raised when the maximum order statistic cannot meet a row error."""


def required_order_statistic(n, epsilon, alpha_row):
    """Return the smallest one-based order ``k`` satisfying the tail rule."""
    if n < 1 or not 0.0 < epsilon < 1.0 or not 0.0 < alpha_row < 1.0:
        raise ValueError("invalid n, epsilon, or alpha_row")
    if binom.sf(n - 1, n, 1.0 - epsilon) > alpha_row:
        raise InfeasibleSampleSize(
            f"n={n} is insufficient for epsilon={epsilon} at alpha={alpha_row}"
        )
    low, high = 1, n
    while low < high:
        mid = (low + high) // 2
        if binom.sf(mid - 1, n, 1.0 - epsilon) <= alpha_row:
            high = mid
        else:
            low = mid + 1
    return low


def clopper_pearson(failures, n, alpha):
    """Return one-sided lower and upper Clopper--Pearson limits."""
    if n < 1 or failures < 0 or failures > n or not 0.0 < alpha < 1.0:
        raise ValueError("invalid failures, n, or alpha")
    lower = 0.0 if failures == 0 else float(beta.ppf(alpha, failures, n - failures + 1))
    upper = 1.0 if failures == n else float(beta.ppf(1.0 - alpha, failures + 1, n - failures))
    return lower, upper


def validation_verdict(lower, upper, target_risk):
    """Map a fixed-item interval to the three nominal statistical outcomes."""
    if upper <= target_risk:
        return "SUPPORTED"
    if lower > target_risk:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"
