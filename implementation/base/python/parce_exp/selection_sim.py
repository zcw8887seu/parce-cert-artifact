"""Post-selection validity simulation (spec 7-P4.1; supplement e01-s1).

Three procedures (pointwise, Bonferroni-simultaneous, split-select-certify)
select stage risks from order-statistic menus built on synthetic samples
from known distributions; content validity is judged exactly via the known
CDFs. Uses the same risk grid, system budget and spec-5.6 gate model as the
physical experiment.

Supplement e01-s1 corrections (supplement.MD S1/S2):
- the split branch executes via the SPLIT_METHOD constant (the legacy
  ``method == "split"`` test never matched ``split_select_certify``);
- selection and certification samples come from independent per-repetition
  seed streams; certification data only re-certifies the frozen risk
  indices at alpha_row = 0.05 / J and never re-selects them;
- sample-cost bookkeeping (n selection units, n certification units, 2n
  total for split; n for the other procedures);
- resume support with atomic 100-repetition batches and unique-key checks;
- H4 disposition computed dynamically from the 12 confirmatory main cells.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .allocator import MenuItem, pareto_dp
from .calendar_ref import load_calendar, next_feasible_start
from .tolerance import required_order_statistic
from .validate import clopper_pearson, verdict

POINTWISE_METHOD = "pointwise_95_then_select"
SIMULTANEOUS_METHOD = "bonferroni_simultaneous"
SPLIT_METHOD = "split_select_certify"
ALL_METHODS = [POINTWISE_METHOD, SIMULTANEOUS_METHOD, SPLIT_METHOD]
CONFIRMATORY_METHODS = [SIMULTANEOUS_METHOD, SPLIT_METHOD]
MASTER_SEED_DEFAULT = 20260816
BATCH_REPS = 100
TARGET_FAMILY_RISK = 0.05

STAGE_SCALES_NS = [100_000, 25_000, 60_000, 90_000]

ROW_COLUMNS = [
    "supplement_id", "experiment_scope", "distribution_id", "gate_regime",
    "J", "R", "n", "repetition", "procedure",
    "selected_content_failure", "family_failure",
    "selected_risk_sum", "bound_ns", "selected_menu_ids",
    "master_seed", "selection_seed", "certification_seed",
    "selection_n_units", "certification_n_units", "sample_cost_units",
    "gate_signature", "status", "infeasible_reason",
]

SUMMARY_COLUMNS = [
    "supplement_id", "experiment_scope", "distribution_id", "gate_regime",
    "J", "R", "n", "procedure", "repetitions",
    "family_failures", "family_failure_rate",
    "selected_failures", "selected_failure_rate",
    "alpha", "cp_lower", "cp_upper", "target", "status",
    "sample_cost_per_repetition", "total_sample_cost", "infeasible_reason",
]

UNIQUE_KEY = ["supplement_id", "experiment_scope", "distribution_id",
              "gate_regime", "J", "R", "n", "repetition", "procedure"]

PAIRING_FIELDS = ["selected_content_failure", "family_failure",
                  "selected_risk_sum", "bound_ns", "selected_menu_ids"]


def tag_int(*parts):
    """Deterministic integer tag from arbitrary (string/int) parts."""
    h = hashlib.sha256("|".join(str(x) for x in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


SEED_MASK = (1 << 62) - 1  # keep derived seeds inside int64 across parquet I/O


def derive_seed(master_seed, cid, repetition, purpose):
    """Per-repetition seed stream tag (supplement.MD 6.3)."""
    return tag_int(master_seed, cid, purpose, repetition) & SEED_MASK


def cell_id(scope, dist_name, regime, J, R, n):
    return f"{scope}/{dist_name}/{regime}/J{J}/R{R}/n{n}"


def cell_gate_signature(calendar_id, period_ns, windows, demand_ns, phase_ns):
    payload = json.dumps({
        "calendar_id": calendar_id, "period_ns": int(period_ns),
        "windows": [[int(o), int(c)] for o, c in windows],
        "demand_ns": int(demand_ns), "phase_ns": int(phase_ns),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _expand_scales(J):
    base = len(STAGE_SCALES_NS)
    return [STAGE_SCALES_NS[j % base] for j in range(J)]


class ShiftedLognormal:
    """X = shift + LogNormal(mu, sigma)."""

    def __init__(self, scale_ns):
        self.shift = 0.6 * scale_ns
        self.mu = float(np.log(0.4 * scale_ns))
        self.sigma = 0.45

    def rvs(self, rng, size):
        return rng.lognormal(self.mu, self.sigma, size=size) + self.shift

    def sf(self, x):
        return stats.lognorm.sf(np.asarray(x) - self.shift,
                                s=self.sigma, scale=np.exp(self.mu))


class GammaDist:
    def __init__(self, scale_ns):
        self.k = 4.0
        self.theta = scale_ns / self.k

    def rvs(self, rng, size):
        return rng.gamma(self.k, self.theta, size=size)

    def sf(self, x):
        return stats.gamma.sf(np.asarray(x), a=self.k, scale=self.theta)


class LognormalMixture:
    """0.7 * LN(m1, 0.35) + 0.3 * LN(m2, 0.7)."""

    def __init__(self, scale_ns):
        self.m1 = float(np.log(0.7 * scale_ns))
        self.m2 = float(np.log(2.0 * scale_ns))
        self.s1, self.s2 = 0.35, 0.7
        self.w = 0.7

    def rvs(self, rng, size):
        comp = rng.random(size=size) < self.w
        a = rng.lognormal(self.m1, self.s1, size=size)
        b = rng.lognormal(self.m2, self.s2, size=size)
        return np.where(comp, a, b)

    def sf(self, x):
        x = np.asarray(x)
        return self.w * stats.lognorm.sf(x, s=self.s1, scale=np.exp(self.m1)) \
            + (1 - self.w) * stats.lognorm.sf(x, s=self.s2, scale=np.exp(self.m2))


DISTRIBUTIONS = {
    "shifted_lognormal": ShiftedLognormal,
    "gamma": GammaDist,
    "lognormal_mixture": LognormalMixture,
}


def gates_for(J):
    """Spec-5.6 pattern repeated: gates after stages (j mod 4) in {1, 2}."""
    return [j % 4 in (1, 2) for j in range(J)]


def make_gate_ops(J, demand, period, windows, phase):
    if not any(gates_for(J)):
        return None
    last_gate = max(j for j, g in enumerate(gates_for(J)) if g)

    def op_for(j):
        def op(r):
            start, _c, _w, reason = next_feasible_start(
                phase + r, demand, period, windows)
            if reason != 0:
                raise RuntimeError("infeasible admission")
            out = start - phase
            if j == last_gate:
                out += demand
            return out
        return op

    return [op_for(j) if gates_for(J)[j] else None for j in range(J)]


def chain_bound_sim(qs, J, demand, period, windows, phase):
    if not any(gates_for(J)):
        return int(sum(qs))
    last_gate = max(j for j, g in enumerate(gates_for(J)) if g)
    r = 0
    for j, q in enumerate(qs):
        r += int(q)
        if gates_for(J)[j]:
            start, _c, _w, reason = next_feasible_start(
                phase + r, demand, period, windows)
            if reason != 0:
                raise RuntimeError("infeasible admission")
            r = start - phase
        if j == last_gate:
            r += demand
    return r


def regime_phase(dist_name, regime, period, windows, demand, seed=20260816):
    """far: median arrival at window open; near: at close-d (frozen-like).

    Always computed from the base 4-stage scale structure (stages 1-2 drive
    the admission position), independent of the cell's J.
    """
    rng = np.random.default_rng(np.random.SeedSequence(entropy=(seed, 6)))
    d1 = DISTRIBUTIONS[dist_name](STAGE_SCALES_NS[0])
    d2 = DISTRIBUTIONS[dist_name](STAGE_SCALES_NS[1])
    m12 = float(np.median(d1.rvs(rng, 200_000) + d2.rvs(rng, 200_000)))
    anchor = windows[0][0] if regime == "far" else windows[0][1] - demand
    return int(round(anchor - m12)) % period


def menu_from_samples(samples_sorted, risks, ks, J):
    """Build per-stage MenuItem lists; ks[r] None marks an infeasible point."""
    menus = []
    for j in range(J):
        items = []
        for r_idx, eps in enumerate(risks):
            k = ks[r_idx]
            if k is None:
                continue
            items.append(MenuItem(menu_id=f"X{j+1}@{eps:g}", risk=float(eps),
                                  q_ns=int(samples_sorted[j][k - 1])))
        menus.append(items)
    return menus


def family_invalid(menus, dists):
    for j, menu in enumerate(menus):
        if not menu:
            continue  # empty stage = infeasible selection, not a content failure
        q = np.array([it.q_ns for it in menu], dtype=float)
        p = dists[j].sf(q)
        eps = np.array([it.risk for it in menu])
        if bool((p > eps + 1e-12).any()):
            return True
    return False


def selected_invalid(sel_menus, sel, dists):
    if sel is None:
        return False
    for j, menu in enumerate(sel_menus):
        it = next(m for m in menu if m.menu_id == sel.menu_ids[j])
        if float(dists[j].sf(it.q_ns)) > it.risk + 1e-12:
            return True
    return False


def _order_stats(n, risks, alpha):
    ks = []
    for eps in risks:
        try:
            ks.append(required_order_statistic(n, float(eps), alpha))
        except Exception:
            ks.append(None)
    return ks


def _finish(base, sel_fail, fam_fail, risk_sum, bound, menu_ids, reason):
    row = dict(base)
    row["selected_content_failure"] = sel_fail
    row["family_failure"] = fam_fail
    row["selected_risk_sum"] = None if risk_sum is None else float(risk_sum)
    row["bound_ns"] = None if bound is None else int(bound)
    row["selected_menu_ids"] = (None if menu_ids is None
                                else json.dumps(list(menu_ids)))
    row["status"] = "INFEASIBLE" if reason else "COMPLETED"
    row["infeasible_reason"] = reason
    return row


def certify_frozen(sel, cert_sorted, n, J):
    """Re-certify frozen risk indices on independent certification data.

    Only q is recomputed (at alpha_row = 0.05/J); the risk indices selected
    on selection data are never revisited (supplement.MD 6.2).
    """
    eps_sel = [float(mid.split("@")[1]) for mid in sel.menu_ids]
    ks_cert = [required_order_statistic(n, eps, 0.05 / J) for eps in eps_sel]
    cert_menus = [[MenuItem(menu_id=sel.menu_ids[j], risk=eps_sel[j],
                            q_ns=int(cert_sorted[j][ks_cert[j] - 1]))]
                  for j in range(J)]
    return cert_menus, eps_sel


def _run_method(method, dists, sel_samples, cert_sorted, dist_name, regime,
                J, R, n, rep, risks, budget, demand, period, windows, phase,
                gate_after, gate_ops, master_seed, selection_seed,
                certification_seed, gsig, supplement_id, scope):
    base = {
        "supplement_id": supplement_id, "experiment_scope": scope,
        "distribution_id": dist_name, "gate_regime": regime,
        "J": J, "R": R, "n": n, "repetition": rep, "procedure": method,
        "master_seed": master_seed, "selection_seed": int(selection_seed),
        "certification_seed": (int(certification_seed)
                               if method == SPLIT_METHOD else None),
        "selection_n_units": n,
        "certification_n_units": n if method == SPLIT_METHOD else 0,
        "sample_cost_units": 2 * n if method == SPLIT_METHOD else n,
        "gate_signature": gsig,
    }
    if method == SPLIT_METHOD:
        a_sel = 0.05  # selection stage picks indices only (supplement.MD 6.2)
    elif method == POINTWISE_METHOD:
        a_sel = 0.05
    elif method == SIMULTANEOUS_METHOD:
        a_sel = 0.05 / (J * R)
    else:
        raise ValueError(f"unknown method: {method}")

    ks = _order_stats(n, risks, a_sel)
    menus = menu_from_samples(sel_samples, risks, ks, J)
    sel = pareto_dp(menus, gate_after, demand, period, windows, budget,
                    base_ns=phase, gate_ops=gate_ops)
    if sel is None:
        return _finish(base, None, None, None, None, None,
                       "NO_FEASIBLE_SELECTION")
    if method != SPLIT_METHOD:
        fam_fail = family_invalid(menus, dists)
        sel_fail = selected_invalid(menus, sel, dists)
        return _finish(base, sel_fail, fam_fail, sel.risk_sum, sel.bound_ns,
                       sel.menu_ids, None)

    # split: risk indices are frozen by the selection DP above; independent
    # certification data only re-computes q for those frozen indices.
    try:
        cert_menus, eps_sel = certify_frozen(sel, cert_sorted, n, J)
    except Exception:
        return _finish(base, None, None, None, None, None,
                       "CERTIFICATION_ORDER_STATISTIC_INFEASIBLE")
    fam_fail = family_invalid(cert_menus, dists)
    sel_fail = selected_invalid(cert_menus, sel, dists)
    bound = chain_bound_sim([m[0].q_ns for m in cert_menus], J, demand,
                            period, windows, phase)
    return _finish(base, sel_fail, fam_fail, sum(eps_sel), bound,
                   sel.menu_ids, None)


def run_cell(dist_name, regime, J, R, n, reps, risks, budget, demand, period,
             windows, phase, methods, master_seed, scope, supplement_id,
             start_rep=0, calendar_id="C1"):
    """Run repetitions [start_rep, start_rep+reps) of one simulation cell."""
    dists = [DISTRIBUTIONS[dist_name](s) for s in _expand_scales(J)]
    gate_after = gates_for(J)
    gate_ops = make_gate_ops(J, demand, period, windows, phase)
    cid = cell_id(scope, dist_name, regime, J, R, n)
    gsig = cell_gate_signature(calendar_id, period, windows, demand, phase)
    rows = []
    for rep in range(start_rep, start_rep + reps):
        selection_seed = derive_seed(master_seed, cid, rep, "selection")
        certification_seed = derive_seed(master_seed, cid, rep, "certification")
        rng_sel = np.random.default_rng(selection_seed)
        sel_samples = np.sort(
            np.stack([dists[j].rvs(rng_sel, n) for j in range(J)]), axis=1)
        cert_sorted = None
        if SPLIT_METHOD in methods:
            rng_cert = np.random.default_rng(certification_seed)
            cert_sorted = np.sort(
                np.stack([dists[j].rvs(rng_cert, n) for j in range(J)]),
                axis=1)
        for method in methods:
            rows.append(_run_method(
                method, dists, sel_samples, cert_sorted, dist_name, regime,
                J, R, n, rep, risks, budget, demand, period, windows, phase,
                gate_after, gate_ops, master_seed, selection_seed,
                certification_seed, gsig, supplement_id, scope))
    return rows


def write_stop_report(run_root, supplement_id, reason, last_step):
    """Supplement.MD 13: stop report + STOPPED state (matching id only)."""
    supp = Path(run_root) / "supplement"
    (supp / "reports").mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    (supp / "reports" / "STOP_REPORT.md").write_text(
        f"# e01-s1 STOP_REPORT\n\n"
        f"- stopped at: {now}\n- supplement_id: {supplement_id}\n"
        f"- last_completed_step: {last_step}\n\n## Reason\n\n{reason}\n")
    state_path = supp / "state" / "supplement_state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        if st.get("supplement_id") != supplement_id:
            return
        st.update({"status": "STOPPED", "stop_reason": reason,
                   "last_completed_step": last_step})
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, indent=2) + "\n")
        os.replace(tmp, state_path)


def update_state(run_root, supplement_id, **updates):
    """Update the supplement state file; no-op for a foreign supplement id."""
    state_path = (Path(run_root) / "supplement" / "state"
                  / "supplement_state.json")
    if not state_path.exists():
        print(f"[state] no state file at {state_path}; skipping update")
        return None
    st = json.loads(state_path.read_text())
    if st.get("supplement_id") != supplement_id:
        print(f"[state] state belongs to {st.get('supplement_id')}; "
              f"not updating for {supplement_id}")
        return None
    st.update(updates)
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=2) + "\n")
    os.replace(tmp, state_path)
    return st


def _batch_name(scope, dist_name, regime, J, R, n, start, end):
    return (f"{scope}__{dist_name}__{regime}__J{J}_R{R}_n{n}"
            f"__r{start:06d}-{end:06d}.parquet")


def _completed_reps(batch_dir, scope, dist_name, regime, J, R, n):
    prefix = f"{scope}__{dist_name}__{regime}__J{J}_R{R}_n{n}__r"
    done = set()
    for p in sorted(batch_dir.glob(prefix + "*.parquet")):
        try:
            s, e = p.stem.split("__r")[-1].split("-")
            s, e = int(s), int(e)
        except ValueError:
            continue
        if e > s:
            done.update(range(s, e))
    return done


def _write_batch(batch_dir, name, rows):
    batch_dir.mkdir(parents=True, exist_ok=True)
    tmp = batch_dir / (name + ".tmp")
    pd.DataFrame(rows, columns=ROW_COLUMNS).to_parquet(tmp, index=False)
    os.replace(tmp, batch_dir / name)


def _run_cell_batched(cell, batch_dir, resume, log_every=10):
    scope, dist_name, regime = (cell["scope"], cell["dist"], cell["regime"])
    J, R, n, total = cell["J"], cell["R"], cell["n"], cell["reps"]
    done = (_completed_reps(batch_dir, scope, dist_name, regime, J, R, n)
            if resume else set())
    n_batches = 0
    for w0 in range(0, total, BATCH_REPS):
        w1 = min(w0 + BATCH_REPS, total)
        if resume and all(r in done for r in range(w0, w1)):
            continue
        rows = run_cell(
            dist_name, regime, J, R, n, w1 - w0, cell["risks"],
            cell["budget"], cell["demand"], cell["period"], cell["windows"],
            cell["phase"], cell["methods"], cell["master_seed"], scope,
            cell["supplement_id"], start_rep=w0,
            calendar_id=cell["calendar_id"])
        _write_batch(batch_dir, _batch_name(scope, dist_name, regime, J, R, n,
                                            w0, w1), rows)
        n_batches += 1
        if n_batches % log_every == 0:
            print(f"[sim] {scope}/{dist_name}/{regime} J{J} R{R} n{n}: "
                  f"reps {w1}/{total} done", flush=True)
    return n_batches


def h4_disposition(cell_statuses):
    """Supplement.MD 7.5: 12 confirmatory main cells decide H4."""
    if any(s == "NOT_SUPPORTED" for s in cell_statuses):
        return "NOT_SUPPORTED"
    if cell_statuses and all(s == "SUPPORTED" for s in cell_statuses):
        return "SUPPORTED"
    return "INCONCLUSIVE"


def build_summary(df, infeasible_cells, supplement_id):
    rows = []
    keys = ["experiment_scope", "distribution_id", "gate_regime", "J", "R",
            "n", "procedure"]
    for key, g in df.groupby(keys, sort=True):
        scope, dist_name, regime, J, R, n, procedure = key
        completed = g[g["status"] == "COMPLETED"]
        fam = completed["family_failure"].dropna().astype(bool)
        sel = completed["selected_content_failure"].dropna().astype(bool)
        reps = int(len(g))
        rec = {
            "supplement_id": supplement_id, "experiment_scope": scope,
            "distribution_id": dist_name, "gate_regime": regime,
            "J": J, "R": R, "n": n, "procedure": procedure,
            "repetitions": reps,
            "family_failures": int(fam.sum()) if len(fam) else 0,
            "family_failure_rate": (float(fam.mean()) if len(fam) else None),
            "selected_failures": int(sel.sum()) if len(sel) else 0,
            "selected_failure_rate": (float(sel.mean()) if len(sel) else None),
            "alpha": 0.05, "target": TARGET_FAMILY_RISK,
            "sample_cost_per_repetition": (
                2 * int(n) if procedure == SPLIT_METHOD else int(n)),
            "total_sample_cost": (
                (2 * int(n) if procedure == SPLIT_METHOD else int(n)) * reps),
            "infeasible_reason": None,
        }
        if len(fam):
            m, N = int(fam.sum()), len(fam)
            lo, hi = clopper_pearson(m, N, 0.05)
            rec["cp_lower"], rec["cp_upper"] = lo, hi
            rec["status"] = (verdict(lo, hi, TARGET_FAMILY_RISK)
                             if procedure in CONFIRMATORY_METHODS
                             else "REPORTED_ONLY")
        else:
            rec["cp_lower"] = rec["cp_upper"] = None
            rec["status"] = "INFEASIBLE"
            rec["infeasible_reason"] = "NO_COMPLETED_REPETITIONS"
        rows.append(rec)
    for cell in infeasible_cells:
        rows.append({
            "supplement_id": supplement_id,
            "experiment_scope": cell["scope"],
            "distribution_id": cell["dist"], "gate_regime": cell["regime"],
            "J": cell["J"], "R": cell["R"], "n": cell["n"],
            "procedure": cell["procedure"], "repetitions": 0,
            "family_failures": None, "family_failure_rate": None,
            "selected_failures": None, "selected_failure_rate": None,
            "alpha": 0.05, "cp_lower": None, "cp_upper": None,
            "target": TARGET_FAMILY_RISK, "status": "INFEASIBLE",
            "sample_cost_per_repetition": (
                2 * int(cell["n"]) if cell["procedure"] == SPLIT_METHOD
                else int(cell["n"])),
            "total_sample_cost": 0,
            "infeasible_reason": cell["reason"],
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["experiment_scope", "distribution_id", "gate_regime", "J", "R", "n",
         "procedure"], ignore_index=True)


def check_merged(df, main_reps, sens_reps):
    """Supplement.MD 7.6 independence/pairing acceptance; returns H4 info or
    raises StopSupplement."""
    errors = []
    dup = df.duplicated(UNIQUE_KEY)
    if dup.any():
        errors.append(f"unique-key duplicates: {int(dup.sum())}")

    main_df = df[df["experiment_scope"] == "main"]
    expected_main = 6 * len(ALL_METHODS) * main_reps
    if len(main_df) != expected_main:
        errors.append(f"main rows {len(main_df)} != {expected_main}")
    for key, g in main_df.groupby(["distribution_id", "gate_regime",
                                   "procedure"]):
        reps = set(g["repetition"])
        if reps != set(range(main_reps)):
            errors.append(f"main cell {key}: missing/duplicated repetitions")

    split = df[df["procedure"] == SPLIT_METHOD]
    if len(split) and (split["selection_seed"].astype("int64")
                       == split["certification_seed"].astype("int64")).any():
        errors.append("selection_seed == certification_seed in split rows")
    if len(split) and ((split["certification_n_units"] != split["n"]) |
                       (split["sample_cost_units"] != 2 * split["n"])).any():
        errors.append("split sample-cost bookkeeping mismatch")
    other = df[df["procedure"] != SPLIT_METHOD]
    if len(other) and ((other["certification_n_units"] != 0) |
                       (other["sample_cost_units"] != other["n"])).any():
        errors.append("non-split sample-cost bookkeeping mismatch")

    identical_cells = []
    main_key = ["distribution_id", "gate_regime"]
    for key in main_df.groupby(main_key).groups:
        cell = main_df[main_df["distribution_id"] == key[0]]
        cell = cell[cell["gate_regime"] == key[1]]
        sim = cell[cell["procedure"] == SIMULTANEOUS_METHOD].sort_values(
            "repetition")
        spl = cell[cell["procedure"] == SPLIT_METHOD].sort_values("repetition")
        if len(sim) != len(spl):
            identical_cells.append(f"{key}: row-count mismatch")
            continue
        same = all(
            sim[f].reset_index(drop=True).equals(
                spl[f].reset_index(drop=True))
            for f in PAIRING_FIELDS)
        if same:
            identical_cells.append(str(key))
    if identical_cells:
        errors.append("simultaneous == split on all five fields in cells: "
                      + ", ".join(identical_cells))
    if errors:
        raise StopSupplement("; ".join(errors))


class StopSupplement(Exception):
    pass


def make_fig04(summary, out_path):
    """Main simulation only: family failure rate + one-sided 95% CP upper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = summary[(summary["experiment_scope"] == "main")
                   & (summary["J"] == 4) & (summary["R"] == 3)
                   & (summary["n"] == 3000)]
    groups = [(d, rg) for d in DISTRIBUTIONS for rg in ("far", "near")]
    methods = [SIMULTANEOUS_METHOD, SPLIT_METHOD, POINTWISE_METHOD]
    labels = {"far": "far", "near": "near"}
    x = np.arange(len(groups))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for m_i, method in enumerate(methods):
        rates, uppers = [], []
        for d, rg in groups:
            row = main[(main["distribution_id"] == d)
                       & (main["gate_regime"] == rg)
                       & (main["procedure"] == method)]
            rates.append(float(row["family_failure_rate"].iloc[0])
                         if len(row) else 0.0)
            uppers.append(float(row["cp_upper"].iloc[0])
                          if len(row) and row["cp_upper"].notna().iloc[0]
                          else np.nan)
        pos = x + (m_i - 1) * width
        short = {"bonferroni_simultaneous": "simultaneous",
                 "split_select_certify": "split (cost 2n)",
                 "pointwise_95_then_select": "pointwise (report only)"}
        ax.bar(pos, rates, width, label=short[method])
        ax.scatter(pos, uppers, marker="_", s=220, linewidths=1.8,
                   color="black", zorder=3)
    ax.axhline(TARGET_FAMILY_RISK, color="red", linestyle="--", linewidth=1.2,
               label="0.05 target")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{labels[rg]}" for d, rg in groups], fontsize=8)
    ax.set_ylabel("family failure rate")
    ax.set_ylim(0, max(0.08, float(np.nanmax(
        main["cp_upper"].astype(float)) * 1.15) if len(main) else 0.08))
    ax.set_title("Post-selection validity, main simulation "
                 "(bars: rate; dashes: one-sided 95% CP upper)")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, "split sample cost = 2n (n selection + n "
             "independent certification units); simultaneous and pointwise "
             "cost n. Pointwise has no confirmatory requirement.",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_report_02(out_path, summary, df, phases, h4_status, main_reps,
                    sens_reps, infeasible_cells, parquet_path):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    main = summary[(summary["experiment_scope"] == "main")
                   & (summary["J"] == 4) & (summary["R"] == 3)
                   & (summary["n"] == 3000)]
    conf = main[main["procedure"].isin(CONFIRMATORY_METHODS)]
    pw = main[main["procedure"] == POINTWISE_METHOD]
    sens = summary[summary["experiment_scope"] == "sensitivity"]

    L = ["# 02 Corrected Selection-Validity Simulation (supplement e01-s1, S2)",
         "",
         f"- generated: {now}",
         f"- main: 3 distributions x 2 regimes x 3 procedures x "
         f"{main_reps} repetitions (J=4, R=3, n=3000; risks 0.002/0.003/"
         "0.004; budget 0.010) = "
         f"{6 * 3 * main_reps} rows",
         f"- sensitivity: shifted_lognormal, J in {{1,8}}, R in {{1,6}}, "
         f"n in {{1000,3000}}, 2 regimes x 2 confirmatory procedures x "
         f"{sens_reps} repetitions; 32 method-cells recorded",
         f"- gate phases (regime_phase): {phases}",
         "",
         "## Main confirmatory cells (one-sided 95% CP, target 0.05)",
         "",
         "| distribution | regime | procedure | reps | failures | rate | "
         "CP lower | CP upper | verdict |", "|---|---|---|---:|---:|---:|"
         "---:|---:|---|"]
    for _, r in conf.iterrows():
        L.append(
            f"| {r['distribution_id']} | {r['gate_regime']} | "
            f"{r['procedure']} | {int(r['repetitions'])} | "
            f"{int(r['family_failures'])} | {r['family_failure_rate']:.4f} | "
            f"{r['cp_lower']:.4f} | {r['cp_upper']:.4f} | {r['status']} |")
    L += ["",
          f"**H4 disposition (supplement.MD 7.5, dynamic): {h4_status}**",
          "", "## Pointwise (reported only; no confirmatory requirement)", "",
          "| distribution | regime | reps | family failures | family rate | "
          "selected failures | selected rate | CP upper |", "|---|---|---:|"
          "---:|---:|---:|---:|---:|"]
    for _, r in pw.iterrows():
        L.append(
            f"| {r['distribution_id']} | {r['gate_regime']} | "
            f"{int(r['repetitions'])} | {int(r['family_failures'])} | "
            f"{r['family_failure_rate']:.4f} | {int(r['selected_failures'])} | "
            f"{r['selected_failure_rate']:.4f} | {r['cp_upper']:.4f} |")
    L += ["", "## Sensitivity cells (all 32 method-cells; INFEASIBLE kept)",
          "",
          "| J | R | n | regime | procedure | reps | family rate | CP upper "
          "| verdict | reason |", "|---:|---:|---:|---|---|---:|---:|---:|"
          "---|---|"]
    for _, r in sens.iterrows():
        rate = (f"{r['family_failure_rate']:.4f}"
                if r["family_failure_rate"] is not None
                and not pd.isna(r["family_failure_rate"]) else "-")
        cpu = (f"{r['cp_upper']:.4f}"
               if r["cp_upper"] is not None and not pd.isna(r["cp_upper"])
               else "-")
        L.append(f"| {int(r['J'])} | {int(r['R'])} | {int(r['n'])} | "
                 f"{r['gate_regime']} | {r['procedure']} | "
                 f"{int(r['repetitions'])} | {rate} | {cpu} | {r['status']} "
                 f"| {r['infeasible_reason'] or ''} |")
    L += ["",
          "## Conventions",
          "",
          "- split_select_certify per repetition: draw J*n selection samples, "
          "build menus at alpha 0.05, run Pareto DP to freeze the per-stage "
          "risk indices; draw J*n independent certification samples from a "
          "different derived seed; re-compute q only for the frozen indices "
          "at alpha_row = 0.05/J; certification data never re-selects "
          "indices.",
          "- seeds: selection_seed = derive(master_seed, cell_id, rep, "
          "\"selection\"), certification_seed = derive(master_seed, cell_id, "
          "rep, \"certification\"); simultaneous and split share selection "
          "samples (common random numbers); split certification samples are "
          "used by no other procedure.",
          "- sample costs (per repetition): pointwise/simultaneous n; split "
          "2n (n selection + n certification).",
          "- rows with status=INFEASIBLE carry no failure judgement (the "
          "procedure produced no certified design); they are excluded from "
          "CP denominators and reported separately.",
          f"- pairing check (supplement.MD 7.6): simultaneous and split are "
          "not identical on all five tracked fields in any cell (verified at "
          "merge time).",
          f"- superseded pre-fix P4.1 data: runs/e01/superseded/"
          "e01-s1-pre-fix/.",
          "",
          "## Reproduction",
          "",
          "```bash",
          "source .venv/bin/activate",
          "python -m parce_exp.selection_sim --run-root runs/e01 "
          "--supplement-id e01-s1 --main-reps "
          f"{main_reps} --sens-reps {sens_reps} "
          "--output-root runs/e01/supplement --resume",
          "```",
          "",
          f"Data: {parquet_path}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")


def _sensitivity_grid(R, n, a_row, eps_budget_stage):
    """R risk levels spanning the certifiable range; (grid, reason)."""
    eps_min_cert = 1.0 - a_row ** (1.0 / n)  # smallest certifiable epsilon
    lo = max(eps_min_cert * 1.05, 0.0005)
    hi = eps_budget_stage * 2.0
    if R == 1:
        grid = [eps_budget_stage]
    else:
        grid = list(np.linspace(lo, hi, R))
    for eps in grid:
        try:
            required_order_statistic(n, float(eps), a_row)
        except Exception:
            return None, (f"ORDER_STATISTIC_INFEASIBLE: n={n} too small to "
                          f"certify eps={eps:g} at alpha_row={a_row:g}")
    return [float(e) for e in grid], None


def main(argv=None):
    p = argparse.ArgumentParser(prog="parce_exp.selection_sim")
    p.add_argument("--run-root", default="runs/e01")
    p.add_argument("--supplement-id", default="e01-s1")
    p.add_argument("--main-reps", type=int, default=10_000)
    p.add_argument("--sens-reps", type=int, default=1000)
    p.add_argument("--output-root", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--master-seed", type=int, default=MASTER_SEED_DEFAULT)
    args = p.parse_args(argv)

    run_root = Path(args.run_root)
    root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root) if args.output_root \
        else run_root / "supplement"
    cal = load_calendar(root / "configs" / "calendar.yaml")
    risk_cfg = yaml.safe_load((root / "configs" / "risk.yaml").read_text())
    demand = int(cal["service_demand_ns"])
    period = int(cal["period_ns"])
    windows = [(int(o), int(c)) for o, c in cal["windows"]]
    calendar_id = str(cal.get("calendar_id", "C1"))
    main_risks = [float(e) for e in risk_cfg["stage_risk_grid"]]
    budget_main = float(risk_cfg["system_risk_budget"])

    batch_dir = output_root / "derived" / "sv_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    common = dict(demand=demand, period=period, windows=windows,
                  master_seed=args.master_seed,
                  supplement_id=args.supplement_id, calendar_id=calendar_id)

    phases = {}
    main_cells = []
    for dist_name in DISTRIBUTIONS:
        for regime in ("far", "near"):
            phase = regime_phase(dist_name, regime, period, windows, demand)
            phases[f"{dist_name}/{regime}"] = phase
            main_cells.append(dict(common, scope="main", dist=dist_name,
                                   regime=regime, J=4, R=3, n=3000,
                                   reps=args.main_reps, methods=ALL_METHODS,
                                   risks=main_risks, budget=budget_main,
                                   phase=phase))

    sens_cells = []
    infeasible_cells = []
    for J in (1, 8):
        for R in (1, 6):
            for n in (1000, 3000):
                for regime in ("far", "near"):
                    dist_name = "shifted_lognormal"
                    phase = regime_phase(dist_name, regime, period, windows,
                                         demand)
                    phases[f"sens/{J}/{R}/{n}/{regime}"] = phase
                    budget = max(0.010, 0.0025 * J)
                    a_row = 0.05 / (J * R)
                    grid, reason = _sensitivity_grid(R, n, a_row, budget / J)
                    if grid is None:
                        for method in CONFIRMATORY_METHODS:
                            infeasible_cells.append(dict(
                                scope="sensitivity", dist=dist_name,
                                regime=regime, J=J, R=R, n=n,
                                procedure=method, reason=reason))
                        print(f"[sim] sensitivity J{J} R{R} n{n} {regime}: "
                              f"INFEASIBLE grid ({reason})", flush=True)
                        continue
                    sens_cells.append(dict(common, scope="sensitivity",
                                           dist=dist_name, regime=regime,
                                           J=J, R=R, n=n,
                                           reps=args.sens_reps,
                                           methods=CONFIRMATORY_METHODS,
                                           risks=grid, budget=budget,
                                           phase=phase))

    for cell in main_cells + sens_cells:
        nb = _run_cell_batched(cell, batch_dir, args.resume)
        tag = (f"{cell['scope']}/{cell['dist']}/{cell['regime']}"
               f"/J{cell['J']}/R{cell['R']}/n{cell['n']}")
        print(f"[sim] cell {tag} complete ({nb} new batches)", flush=True)

    frames = [pd.read_parquet(p) for p in sorted(batch_dir.glob("*.parquet"))]
    if not frames:
        raise SystemExit("no simulation batches found")
    df = pd.concat(frames, ignore_index=True)
    try:
        check_merged(df, args.main_reps, args.sens_reps)
    except StopSupplement as exc:
        write_stop_report(run_root, args.supplement_id, str(exc), "S2")
        raise SystemExit(f"STOPPED (supplement.MD 13): {exc}")

    update_state(run_root, args.supplement_id, status="P4_RERUN_COMPLETE",
                 completed_steps=["S0", "S1", "S2"],
                 p4_main_completed_cells=[
                     f"{c['dist']}/{c['regime']}" for c in main_cells],
                 p4_sensitivity_completed_cells=[
                     f"J{c['J']}/R{c['R']}/n{c['n']}/{c['regime']}"
                     f"/{m}" for c in sens_cells
                     for m in CONFIRMATORY_METHODS] + [
                     f"J{c['J']}/R{c['R']}/n{c['n']}/{c['regime']}"
                     f"/{c['procedure']}/INFEASIBLE"
                     for c in infeasible_cells])

    summary = build_summary(df, infeasible_cells, args.supplement_id)
    conf = summary[(summary["experiment_scope"] == "main")
                   & (summary["J"] == 4) & (summary["R"] == 3)
                   & (summary["n"] == 3000)
                   & (summary["procedure"].isin(CONFIRMATORY_METHODS))]
    h4_status = h4_disposition(list(conf["status"]))

    for col in ("J", "R", "n", "repetition", "master_seed", "selection_seed",
                "certification_seed", "selection_n_units",
                "certification_n_units", "sample_cost_units", "bound_ns"):
        df[col] = df[col].astype("Int64")
    for col in ("selected_content_failure", "family_failure"):
        df[col] = df[col].astype("boolean")

    derived = output_root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    pq = derived / "selection_validity.parquet"
    tmp = pq.with_suffix(".parquet.tmp")
    df[ROW_COLUMNS].to_parquet(tmp, index=False)
    os.replace(tmp, pq)
    summary.to_csv(derived / "selection_validity_summary.csv", index=False)

    fig_path = output_root / "figures" / "fig04_selection_validity.svg"
    make_fig04(summary, fig_path)
    write_report_02(output_root / "reports" /
                    "02_CORRECTED_SELECTION_VALIDITY.md", summary, df,
                    phases, h4_status, args.main_reps, args.sens_reps,
                    infeasible_cells, pq)

    print(summary[["experiment_scope", "distribution_id", "gate_regime",
                   "J", "R", "n", "procedure", "repetitions",
                   "family_failure_rate", "cp_upper", "status"]]
          .to_string(index=False))
    print(f"H4 disposition: {h4_status}")
    print(f"rows: {len(df)} (main "
          f"{int((df['experiment_scope'] == 'main').sum())}); "
          f"sensitivity method-cells in summary: "
          f"{int((summary['experiment_scope'] == 'sensitivity').sum())}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
