"""Order-statistic tolerance construction (spec 6.2)."""

from scipy.stats import binom


class InfeasibleSampleSize(Exception):
    pass


def alpha_row(family_size, family_alpha):
    return family_alpha / family_size


def required_order_statistic(n, epsilon, alpha_row_):
    """Smallest k such that BinomialSurvival(k-1; n, 1-eps) <= alpha_row.

    Y_(k) is the k-th smallest of n samples, so k = n selects the maximum.
    Raises InfeasibleSampleSize when even the maximum does not certify.
    """
    if n < 1 or not (0.0 < epsilon < 1.0):
        raise ValueError("invalid n or epsilon")
    if binom.sf(n - 1, n, 1.0 - epsilon) > alpha_row_:
        raise InfeasibleSampleSize(
            f"n={n} insufficient for epsilon={epsilon} at alpha={alpha_row_}"
        )
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if binom.sf(mid - 1, n, 1.0 - epsilon) <= alpha_row_:
            hi = mid
        else:
            lo = mid + 1
    return lo


MENU_COLUMNS = [
    "menu_id", "coordinate_id", "risk", "alpha_family", "alpha_row",
    "n_units", "order_k", "q_ns", "q_is_infinite", "status",
]
STAGE_COORDS = ["X1", "X2", "X3", "X4"]


def calibrate_menu(samples, risk_grid, family_alpha, family_size=12):
    """Build the simultaneous menu from calibration stage samples (spec 6.2).

    samples: derived/stage_samples.parquet rows for split=calibration.
    Returns (menu_dataframe, pruned_list).
    """
    import pandas as pd

    a_row = alpha_row(family_size, family_alpha)
    rows = []
    pruned = []
    for coord in STAGE_COORDS:
        cs = samples[(samples["coordinate_id"] == coord)]
        values = sorted(
            float(v) if not inf else float("inf")
            for v, inf in zip(cs["value_ns"], cs["is_infinite"])
        )
        n = len(values)
        points = []
        for eps in risk_grid:
            menu_id = f"{coord}@{eps:g}"
            try:
                k = required_order_statistic(n, float(eps), a_row)
            except InfeasibleSampleSize:
                rows.append({
                    "menu_id": menu_id, "coordinate_id": coord,
                    "risk": float(eps), "alpha_family": float(family_alpha),
                    "alpha_row": a_row, "n_units": n, "order_k": None,
                    "q_ns": None, "q_is_infinite": False,
                    "status": "INFEASIBLE_SAMPLE_SIZE"})
                continue
            q = values[k - 1]
            q_inf = q == float("inf")
            points.append({
                "menu_id": menu_id, "coordinate_id": coord,
                "risk": float(eps), "alpha_family": float(family_alpha),
                "alpha_row": a_row, "n_units": n, "order_k": int(k),
                "q_ns": None if q_inf else int(q),
                "q_is_infinite": q_inf,
                "status": "INFEASIBLE_PACKET_LOSS" if q_inf else "CERTIFIED"})
        # drop points dominated in BOTH risk and q (ties preserved, spec 6.2)
        for a in points:
            if a["status"] != "CERTIFIED":
                rows.append(a)
                continue
            dominated = False
            for b in points:
                if b is a or b["status"] != "CERTIFIED":
                    continue
                if (b["risk"] <= a["risk"] and b["q_ns"] <= a["q_ns"]
                        and (b["risk"] < a["risk"] or b["q_ns"] < a["q_ns"])):
                    dominated = True
                    break
            if dominated:
                pruned.append(a["menu_id"])
            else:
                rows.append(a)
    return pd.DataFrame(rows, columns=MENU_COLUMNS), pruned


def main(argv=None):
    import argparse
    import json
    import os
    from pathlib import Path

    import pandas as pd
    import yaml

    p = argparse.ArgumentParser(prog="parce_exp.tolerance")
    sub = p.add_subparsers(dest="cmd", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--run-root", default="runs/e01")
    args = p.parse_args(argv)
    run_root = Path(args.run_root)
    root = Path(__file__).resolve().parents[2]
    risk = yaml.safe_load((root / "configs" / "risk.yaml").read_text())
    samples = pd.read_parquet(run_root / "derived" / "stage_samples.parquet")
    samples = samples[samples["split"] == "calibration"]
    menu, pruned = calibrate_menu(
        samples, risk["stage_risk_grid"], risk["calibration_family_alpha"])
    out = run_root / "derived" / "menu.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    menu.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    (run_root / "derived" / "menu_pruned.json").write_text(
        json.dumps({"pruned_dominated_menu_ids": pruned}, indent=2))
    print(menu.to_string(index=False))
    print(f"pruned (dominated): {pruned}")


if __name__ == "__main__":
    main()
