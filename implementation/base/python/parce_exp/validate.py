"""Independent validation statistics (spec 6.4)."""

from scipy.stats import beta


def clopper_pearson(m, n, alpha):
    """One-sided-pair Clopper-Pearson interval for m failures out of n."""
    lower = 0.0 if m == 0 else float(beta.ppf(alpha, m, n - m + 1))
    upper = 1.0 if m == n else float(beta.ppf(1.0 - alpha, m + 1, n - m))
    return lower, upper


def verdict(cp_lower, cp_upper, target_risk):
    if cp_upper <= target_risk:
        return "SUPPORTED"
    if cp_lower > target_risk:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------ P3
def analyze(args):
    import hashlib
    import json
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    from . import plots

    run_root = Path(args.run_root)
    root = Path(__file__).resolve().parents[2]
    risk_cfg = yaml.safe_load((root / "configs" / "risk.yaml").read_text())
    design = json.loads((run_root / "derived" / "selected_design.json").read_text())
    if design.get("status") != "DESIGN_FROZEN":
        raise SystemExit("no frozen design; physical validation is skipped "
                         "per spec 7-P3 (NOT_SUPPORTED recorded)")
    baselines = json.loads((run_root / "derived" / "baselines.json").read_text())

    samples = pd.read_parquet(run_root / "derived" / "stage_samples.parquet")
    val = samples[samples["split"] == "validation"]
    coords = COORDS = ["X1", "X2", "X3", "X4"]
    alpha_row_val = 0.05 / 5

    # per-run joint vectors
    per_run = {}
    for coord in coords + ["E2E"]:
        col = val[val["coordinate_id"] == coord]
        for r in col.itertuples():
            d = per_run.setdefault(int(r.run_id), {})
            d[coord] = (float("inf") if r.is_infinite or pd.isna(r.value_ns)
                        else float(r.value_ns))

    q_sel = {c: int(design["selected_q_ns"][c]) for c in coords}
    risk_sel = {c: float(design["selected_risks"][c]) for c in coords}
    B = int(design["bound_ns"])
    risk_sum = float(design["risk_sum"])

    rows = []
    for coord in coords:
        vals = [per_run[run][coord] for run in per_run]
        n = len(vals)
        m = sum(1 for v in vals if v == float("inf") or v > q_sel[coord])
        lo, hi = clopper_pearson(m, n, alpha_row_val)
        rows.append({
            "item_id": coord, "target_risk": risk_sel[coord], "n_units": n,
            "failures": m, "alpha_row": alpha_row_val, "cp_lower": lo,
            "cp_upper": hi, "status": verdict(lo, hi, risk_sel[coord])})
    e2e_vals = [per_run[run]["E2E"] for run in per_run]
    n_e2e = len(e2e_vals)
    m_e2e = sum(1 for v in e2e_vals if v == float("inf") or v > B)
    lo, hi = clopper_pearson(m_e2e, n_e2e, alpha_row_val)
    rows.append({
        "item_id": "E2E", "target_risk": risk_sum, "n_units": n_e2e,
        "failures": m_e2e, "alpha_row": alpha_row_val, "cp_lower": lo,
        "cp_upper": hi, "status": verdict(lo, hi, risk_sum)})

    results = pd.DataFrame(rows, columns=[
        "item_id", "target_risk", "n_units", "failures", "alpha_row",
        "cp_lower", "cp_upper", "status"])
    out = run_root / "derived" / "validation_results.parquet"
    results.to_parquet(out, index=False)

    # event inclusion (spec 5.6): E2E > B requires some Xj > qj or infinite
    inclusion_violations = 0
    for run, d in per_run.items():
        if d["E2E"] == float("inf") or d["E2E"] > B:
            if not any(d[c] == float("inf") or d[c] > q_sel[c] for c in coords):
                inclusion_violations += 1

    finite_e2e = sorted(v for v in e2e_vals if v != float("inf"))
    slack_min = (B - finite_e2e[-1]) if finite_e2e else None
    slack_median = (B - float(np.median(finite_e2e))) if finite_e2e else None

    # charged cycles / reschedules on validation selected instances
    ev_path = run_root / "raw" / "events"
    frames = [pd.read_parquet(p) for p in sorted(ev_path.glob("batch_*.parquet"))]
    events = pd.concat(frames, ignore_index=True)
    vs = events[(events["split"] == "validation")
                & events["selected_single"].astype(bool)
                & events["valid_event"].astype(bool)]
    reschedules = int(vs["reschedule_count"].sum())
    extra_cycle = int((vs["service_cycle_id"] != vs["target_cycle_id"]).sum())
    gate_sig = hashlib.sha256(json.dumps(
        design["calendar_signature"], separators=(",", ":")).encode()).hexdigest()

    fig02 = plots.fig02_stage_validation(
        {c: [per_run[r][c] for r in per_run] for c in coords},
        q_sel, {c: dict(results[results.item_id == c].iloc[0]) for c in coords},
        run_root / "figures" / "fig02_stage_validation.svg")
    fig03 = plots.fig03_e2e_validation(
        e2e_vals, B, baselines, risk_sum,
        run_root / "figures" / "fig03_e2e_validation.svg")

    lines = [
        "# 04 Certification and Validation (P3)",
        "",
        f"- generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- design: procedure={design['procedure']}, phase_ns="
        f"{design['phase_ns']}, risk_sum={risk_sum:.4f}, bound_ns={B},",
        f"  replay_bound_ns={design['replay_bound_ns']}, "
        f"dp_matches_brute_force={design['dp_matches_brute_force']}",
        f"- gate signature (sha256 of calendar signature): {gate_sig}",
        f"- calendar signature: {design['calendar_signature']}",
        "",
        "## Validation results (spec 6.4; family alpha 0.05, size 5)",
        "",
        "| item | target risk | n | failures | CP lower | CP upper | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['item_id']} | {r['target_risk']:.4f} | {r['n_units']} | "
            f"{r['failures']} | {r['cp_lower']:.6f} | {r['cp_upper']:.6f} | "
            f"{r['status']} |")
    lines += [
        "",
        f"- event inclusion violations (E2E > B without any Xj > qj): "
        f"{inclusion_violations} (required: 0)",
        f"- packet loss in validation selected instances: "
        f"{sum(1 for v in e2e_vals if v == float('inf'))}",
        f"- slack: B - max(E2E) = {slack_min} ns; B - median(E2E) = "
        f"{slack_median:.0f} ns",
        f"- charged cycles: instances with service_cycle != target_cycle = "
        f"{extra_cycle}; total reschedules = {reschedules}",
        "",
        "## Baselines (frozen before validation data was read)",
        "",
        "| baseline | description | value |",
        "|---|---|---|",
        f"| B1 | equal-risk phase-aware (risk<=0.0025 per stage, risk_sum "
        f"{baselines['B1_equal_risk_phase_aware']['risk_sum']:.4f}) | bound "
        f"{baselines['B1_equal_risk_phase_aware']['bound_ns']} ns |",
        f"| B2 | same-q legal max-wait | bound "
        f"{baselines['B2_same_q_max_wait']['bound_ns']} ns |",
        f"| B3 | calibration E2E order statistic (eps={baselines['B3_calibration_e2e_order_statistic']['epsilon']:.4f}, "
        f"k={baselines['B3_calibration_e2e_order_statistic']['order_k']}) | "
        f"bound {baselines['B3_calibration_e2e_order_statistic']['bound_ns']} ns |",
        f"| B4 | marginal resampling ({baselines['B4_marginal_resampling']['flag']}) | "
        f"P(E2E>B)~{baselines['B4_marginal_resampling']['exceedance_probability_estimate']:.5f}, "
        f"q={baselines['B4_marginal_resampling']['quantile_bound_ns']} ns |",
        "",
        f"- B4 frozen_at {baselines['frozen_at']} (before validation analysis)",
        "",
        "Figures: figures/fig02_stage_validation.svg, "
        "figures/fig03_e2e_validation.svg; data: derived/validation_results.parquet.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "source .venv/bin/activate",
        "bash scripts/run_calibration.sh && python -m parce_exp.extract --run-root runs/e01 --split calibration",
        "python -m parce_exp.tolerance calibrate --run-root runs/e01",
        "python -m parce_exp.select_design --run-root runs/e01",
        "bash scripts/run_validation.sh && python -m parce_exp.extract --run-root runs/e01 --split validation",
        "python -m parce_exp.validate --run-root runs/e01",
        "```",
    ]
    (run_root / "reports" / "04_CERTIFICATION_VALIDATION.md").write_text(
        "\n".join(lines) + "\n")
    print(results.to_string(index=False))
    print(f"event inclusion violations: {inclusion_violations}")


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="parce_exp.validate")
    p.add_argument("--run-root", default="runs/e01")
    args = p.parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
