"""Numerical summaries for selection validity, confidence levels and runtime."""

def h4_main_cells(selval):
    """12 confirmatory main cells from corrected selection-validity data."""
    from .selection_sim import CONFIRMATORY_METHODS
    from .validate import clopper_pearson, verdict

    main = selval[(selval["experiment_scope"] == "main")
                  & (selval["J"] == 4) & (selval["R"] == 3)
                  & (selval["n"] == 3000)
                  & (selval["procedure"].isin(CONFIRMATORY_METHODS))]
    cells = []
    for key, g in main.groupby(
            ["distribution_id", "gate_regime", "procedure"]):
        fam = g["family_failure"].dropna().astype(bool)
        m, N = int(fam.sum()), len(fam)
        lo = hi = float("nan")
        if N:
            lo, hi = clopper_pearson(m, N, 0.05)
        cells.append({
            "d": key[0], "rg": key[1], "m": key[2], "failures": m,
            "rate": m / N if N else float("nan"), "lo": lo, "hi": hi,
            "reps": len(g), "n_completed": N,
            "status": verdict(lo, hi, 0.05) if N else "INCONCLUSIVE"})
    return cells

def h4_status_from_data(selval):
    """Compute the H4 disposition from the confirmatory cells."""
    from .selection_sim import h4_disposition

    cells = h4_main_cells(selval)
    return h4_disposition([c["status"] for c in cells]), cells

def pointwise_family_rate_range(selval):
    """Min/max pointwise family-failure rate over the 6 main cells."""
    pw = selval[(selval["experiment_scope"] == "main") & (selval["J"] == 4)
                & (selval["R"] == 3) & (selval["n"] == 3000)
                & (selval["procedure"] == "pointwise_95_then_select")]
    rates = pw.groupby(["distribution_id", "gate_regime"])["family_failure"].apply(
        lambda s: float(s.dropna().astype(bool).mean()))
    if not len(rates):
        return None, None, 0
    return float(rates.min()), float(rates.max()), int(len(rates))

def format_runtime(ns):
    """Convert runtime_ns to a correctly labelled unit string."""
    ns = float(ns)
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.4g} us"
    if ns < 1e9:
        return f"{ns / 1e6:.4g} ms"
    return f"{ns / 1e9:.4g} s"

def cp_confidence_wording(alpha_row_):
    """Confidence wording that never mislabels alpha_row=0.01 as 95%."""
    per_item = 1.0 - float(alpha_row_)
    if abs(per_item - 0.99) < 1e-9:
        return ("one-sided 99% per-item CP (alpha_row=0.01; Bonferroni "
                "family-wise 95% over the 5-item validation family)")
    if abs(per_item - 0.95) < 1e-9 and float(alpha_row_) == 0.05:
        return "one-sided 95% CP"
    return (f"one-sided {per_item * 100:.4g}% CP "
            f"(alpha_row={float(alpha_row_):g})")
