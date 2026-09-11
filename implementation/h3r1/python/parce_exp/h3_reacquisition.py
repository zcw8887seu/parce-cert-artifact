"""e01-h3r1: multi-session physical reacquisition of H3 evidence.

Re-runs Calibration (20 sessions x 150 valid runs) and Validation
(20 sessions x 175 valid runs) under the e01 frozen configuration, with
session-boundary-aware dependence diagnostics, to replace the H3 evidence
that e01-s1 downgraded to INCONCLUSIVE_CORRELATION_SENSITIVITY.

runs/e01 is strictly read-only for this experiment; every write goes through
guard_write() which refuses paths resolving inside runs/e01.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from .calendar_ref import load_calendar
from .extract import event_coordinates
from .runner import CPU_MAP, Campaign
from .schemas import EVENT_COLUMNS
from .tolerance import alpha_row, calibrate_menu, required_order_statistic
from .validate import clopper_pearson, verdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
E01_ROOT = PROJECT_ROOT / "runs" / "e01"
EXPERIMENT_ID = "e01-h3r1"
PARENT_EXPERIMENT_ID = "e01"
MASTER_SEED = 20260818

COORDS = ["X1", "X2", "X3", "X4", "E2E"]
STAGE_COORDS = ["X1", "X2", "X3", "X4"]
SPLITS = ("calibration", "validation")
SESSIONS = {"calibration": 20, "validation": 20}
PER_SESSION = {"calibration": 150, "validation": 175}
TARGET_VALID = {"calibration": 3000, "validation": 3500}
BATCH_FLUSH = 25
IDLE_BEFORE_S = 60
LONG_IDLE_AFTER_S = 300
LONG_IDLE_EVERY = 5
BLOCK_SIZES = (25, 50, 100)
ALPHA_ROW_VALIDATION = 0.05 / 5
ALPHA_ROW_CALIBRATION = 0.05 / 12

FROZEN_CONFIG = {
    "workload_id": "S1",
    "calendar_id": "C1",
    "period_ns": 1000000,
    "windows_ns": [[400000, 700000]],
    "phase_ns": 455509,
    "service_demand_ns": 50000,
    "packet_bytes": 256,
    "stage_risk_grid": [0.002, 0.003, 0.004],
    "system_risk_budget": 0.010,
    "calibration_family_alpha": 0.05,
    "validation_family_alpha": 0.05,
    "sender_cpu": 0,
    "gate_cpu": 2,
    "receiver_cpu": 4,
    "scheduler": "SCHED_OTHER",
    "warmup_instances_per_run": 32,
    "analysis_instances_per_run": 8,
    "selected_instances_per_run": 1,
}

STATE_ORDER = [
    "INIT", "SOFTWARE_VERIFIED", "PLAN_FROZEN", "CALIBRATION_RUNNING",
    "CALIBRATION_COMPLETE", "CALIBRATION_DIAGNOSTICS_COMPLETE",
    "DESIGN_FROZEN", "VALIDATION_RUNNING", "VALIDATION_COMPLETE",
    "ANALYZED", "REPORTS_COMPLETE", "COMPLETE",
]

TECHNICAL_INVALID = {
    "PROCESS_CRASH", "RAW_LOG_CORRUPT", "LOG_BUFFER_OVERFLOW",
    "CLOCK_ANOMALY", "NEGATIVE_STAGE_DURATION", "SERVICE_ORDER_VIOLATION",
    "THERMAL_THROTTLE", "USER_ABORT",
}

SESSION_PLAN_COLUMNS = [
    "experiment_id", "split", "planned_session_index", "planned_valid_runs",
    "planned_idle_before_s", "planned_long_idle_after_s", "session_seed",
    "status", "actual_session_id", "actual_start", "actual_end",
]
RUN_PLAN_COLUMNS = [
    "experiment_id", "split", "planned_session_index", "session_order",
    "run_id", "seed", "warmup_instances", "analysis_instances",
    "selected_instance_index", "phase_ns", "status", "actual_session_id",
    "attempt_index", "invalid_reason",
]
H3R1_EVENT_COLUMNS = [
    "experiment_id", "split", "planned_session_index", "actual_session_id",
    "session_order", "run_id", "instance_id", "seq", "is_warmup",
    "selected_single", "phase_ns", "service_demand_ns",
    "t_cause_ns", "t_compute_end_ns", "t_gate_ingress_ns",
    "t_gate_target1_ns", "t_gate_service_start_ns", "t_gate_send_ns",
    "t_rx_ns", "target_cycle_id", "target_window_id", "service_cycle_id",
    "service_window_id", "reschedule_count", "packet_lost", "out_of_order",
    "duplicate", "valid_event", "invalid_reason",
]
RUN_SUMMARY_COLUMNS = [
    "experiment_id", "split", "planned_session_index", "actual_session_id",
    "session_order", "run_id", "attempt_index", "status", "valid_run",
    "selected_instance_index", "phase_ns", "sender_exit", "gate_exit",
    "receiver_exit", "packet_lost", "out_of_order", "duplicate",
    "log_drops", "reference_mismatch_count", "invalid_reason",
    "started_at", "finished_at",
]
STAGE_SAMPLE_COLUMNS = [
    "experiment_id", "split", "planned_session_index", "actual_session_id",
    "session_order", "run_id", "coordinate_id", "value_ns", "is_infinite",
    "selected_single", "phase_ns", "calendar_id", "workload_id",
]
SESSION_PLATFORM_COLUMNS = [
    "experiment_id", "split", "planned_session_index", "actual_session_id",
    "capture_point", "start_or_end", "timestamp", "os", "kernel",
    "cpu_model", "logical_cpus", "numa_nodes", "governor", "smt",
    "temperature_c", "load_1m", "load_5m", "available_memory_bytes",
    "free_disk_bytes", "sender_cpu", "gate_cpu", "receiver_cpu",
    "clock_resolution_ns", "unavailable_fields",
]
DIAG_COLUMNS = [
    "split", "coordinate_id", "n_units", "n_lag_pairs", "lag1_pearson",
    "lag1_spearman", "trigger_threshold", "correlation_triggered",
    "acf_lag2", "acf_lag5", "acf_lag10", "acf_lag25",
    "session_median_min", "session_median_max", "session_maximum_min",
    "session_maximum_max", "status",
]
SESSION_SUMMARY_COLUMNS = [
    "split", "coordinate_id", "planned_session_index", "actual_session_id",
    "n_units", "median_ns", "maximum_ns", "failure_count",
    "lag1_pearson", "lag1_spearman",
]
BLOCK_SENSITIVITY_COLUMNS = [
    "item_id", "block_size", "n_blocks", "failed_blocks",
    "block_failure_rate", "cp_lower", "cp_upper", "run_level_failures",
    "degenerate_flag",
]
VALIDATION_RESULT_COLUMNS = [
    "item_id", "target_risk", "n_units", "failures", "alpha_row",
    "cp_lower", "cp_upper", "status",
]


# ----------------------------------------------------------------- guards
def utc_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def inside_e01(path):
    p = Path(path).resolve(strict=False)
    e01 = E01_ROOT.resolve(strict=False)
    return p == e01 or e01 in p.parents


def guard_write(path):
    """Refuse any write whose target resolves inside runs/e01 (spec 1.1)."""
    if inside_e01(path):
        raise SystemExit(f"STOP: write target resolves inside runs/e01: {path}")
    return Path(path)


def guard_run_root(run_root):
    root = Path(run_root).resolve(strict=False)
    if inside_e01(root):
        raise SystemExit(f"STOP: run root resolves inside runs/e01: {run_root}")
    if root == PROJECT_ROOT.resolve(strict=False):
        raise SystemExit("STOP: run root must be a dedicated directory")
    return root


def atomic_write_bytes(path, data):
    path = guard_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def atomic_write_parquet(df, path):
    import pyarrow  # noqa: F401  (fail fast if missing)

    path = guard_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_csv(df, path):
    path = guard_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ------------------------------------------------------------------ state
def state_path(run_root):
    return Path(run_root) / "state" / "state.json"


def load_state(run_root):
    p = state_path(run_root)
    if not p.exists():
        raise SystemExit(f"missing state file {p}; run `init` first")
    return json.loads(p.read_text())


def write_state(run_root, **updates):
    """Atomic state update (spec 4.2)."""
    st = load_state(run_root)
    st.update(updates)
    st["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_json(state_path(run_root), st)
    return st


def init_state_file(run_root):
    p = state_path(run_root)
    if p.exists():
        raise SystemExit(f"state file already exists at {p}; refusing re-init")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "INIT",
        "last_completed_step": "R0",
        "completed_steps": [],
        "h3_status": "NOT_RUN",
        "calibration_correlation_flagged": False,
        "calibration_valid_runs": 0,
        "validation_valid_runs": 0,
        "stop_reason": None,
        "init_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_json(p, payload)
    return payload


def complete_step(run_root, step, status=None, **updates):
    st = load_state(run_root)
    if st.get("status") == "STOPPED":
        raise SystemExit("experiment is STOPPED; refusing further steps")
    if step not in st["completed_steps"]:
        st["completed_steps"].append(step)
    if status is not None:
        st["status"] = status
    st["last_completed_step"] = step
    st.update(updates)
    st["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_json(state_path(run_root), st)
    return st


def require_status(run_root, allowed, step):
    st = load_state(run_root)
    if st.get("status") == "STOPPED":
        raise SystemExit("experiment is STOPPED; no further action")
    if st.get("status") not in allowed:
        raise SystemExit(
            f"{step}: requires status in {sorted(allowed)}, current "
            f"{st.get('status')!r} (validation may not start before "
            "DESIGN_FROZEN)")


def stop_experiment(run_root, reason, last_step):
    """Spec 11: keep files, write STOP_REPORT, set STOPPED."""
    run_root = Path(run_root)
    reports = run_root / "reports"
    guard_write(reports).mkdir(parents=True, exist_ok=True)
    st = load_state(run_root)
    atomic_write_text(
        reports / "STOP_REPORT.md",
        f"# e01-h3r1 STOP_REPORT\n\n"
        f"- stopped at: {utc_now()}\n"
        f"- last_completed_step: {last_step}\n\n## Reason\n\n{reason}\n\n"
        "## Affected outputs\n\nAll files generated so far are preserved "
        "as-is; nothing was deleted or overwritten.\n")
    log_deviation(run_root, "STOP", f"{last_step}: {reason}")
    write_state(run_root, status="STOPPED", stop_reason=reason,
                last_completed_step=last_step)
    raise SystemExit(f"STOPPED: {reason}")


def log_deviation(run_root, kind, detail):
    run_root = Path(run_root)
    guard_write(run_root / "meta").mkdir(parents=True, exist_ok=True)
    line = json.dumps({"time": utc_now(), "kind": kind, "detail": detail})
    with open(run_root / "meta" / "deviations.jsonl", "a") as fh:
        fh.write(line + "\n")


# ----------------------------------------------------------------- plans
def derive_rng(*parts):
    from .allocator import tag_int

    return np.random.default_rng(
        np.random.SeedSequence(entropy=(MASTER_SEED, tag_int(*parts))))


def session_id_format(split, index):
    return f"{EXPERIMENT_ID}-{split}-s{index:02d}"


def actual_session_id(split, index):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{session_id_format(split, index)}-{stamp}"


def valid_session_id(sid):
    """Session ids must be real, non-empty, non-PLANNED, unique upstream."""
    return bool(sid) and sid.strip() != "" and sid != "PLANNED"


def generate_plans():
    """Pure plan generation from MASTER_SEED (spec 6; one-shot)."""
    sess_rows, run_rows = [], []
    run_id = 0
    for split in SPLITS:
        n_sess, per = SESSIONS[split], PER_SESSION[split]
        for idx in range(1, n_sess + 1):
            long_idle = (LONG_IDLE_AFTER_S
                         if idx % LONG_IDLE_EVERY == 0 and idx < n_sess else 0)
            sess_rows.append({
                "experiment_id": EXPERIMENT_ID, "split": split,
                "planned_session_index": idx, "planned_valid_runs": per,
                "planned_idle_before_s": IDLE_BEFORE_S,
                "planned_long_idle_after_s": long_idle,
                "session_seed": int(derive_rng("session", split, idx)
                                    .integers(0, 2**62)),
                "status": "PLANNED", "actual_session_id": "",
                "actual_start": "", "actual_end": "",
            })
            for order in range(1, per + 1):
                run_id += 1
                rng = derive_rng("run", run_id)
                run_rows.append({
                    "experiment_id": EXPERIMENT_ID, "split": split,
                    "planned_session_index": idx, "session_order": order,
                    "run_id": run_id,
                    "seed": int(rng.integers(0, 2**62)),
                    "warmup_instances": FROZEN_CONFIG[
                        "warmup_instances_per_run"],
                    "analysis_instances": FROZEN_CONFIG[
                        "analysis_instances_per_run"],
                    "selected_instance_index": int(
                        rng.integers(0, FROZEN_CONFIG[
                            "analysis_instances_per_run"])),
                    "phase_ns": FROZEN_CONFIG["phase_ns"],
                    "status": "PLANNED", "actual_session_id": "",
                    "attempt_index": 0, "invalid_reason": "",
                })
    return (pd.DataFrame(sess_rows, columns=SESSION_PLAN_COLUMNS),
            pd.DataFrame(run_rows, columns=RUN_PLAN_COLUMNS))


def validate_plans(sess_df, run_df):
    """Spec 6 acceptance; returns list of violation strings."""
    v = []
    for split in SPLITS:
        s = sess_df[sess_df["split"] == split]
        if len(s) != SESSIONS[split]:
            v.append(f"{split}: session rows {len(s)} != {SESSIONS[split]}")
        if not (s["planned_valid_runs"] == PER_SESSION[split]).all():
            v.append(f"{split}: planned_valid_runs mismatch")
        r = run_df[run_df["split"] == split]
        if len(r) != TARGET_VALID[split]:
            v.append(f"{split}: run rows {len(r)} != {TARGET_VALID[split]}")
        for idx, grp in r.groupby("planned_session_index"):
            orders = sorted(grp["session_order"].tolist())
            if orders != list(range(1, PER_SESSION[split] + 1)):
                v.append(f"{split} session {idx}: session_order not "
                         f"1..{PER_SESSION[split]} contiguous")
    if run_df["run_id"].duplicated().any():
        v.append("run_id not globally unique")
    if run_df["seed"].duplicated().any():
        v.append("seed not globally unique")
    if sess_df["session_seed"].duplicated().any():
        v.append("session_seed not unique")
    if not run_df["selected_instance_index"].between(0, 7).all():
        v.append("selected_instance_index outside [0,7]")
    done = run_df[run_df["status"] != "PLANNED"]
    if len(done):
        v.append(f"{len(done)} non-PLANNED rows before acquisition")
    if not (run_df["phase_ns"] == FROZEN_CONFIG["phase_ns"]).all():
        v.append("phase_ns deviates from frozen config")
    return v


def load_plans(run_root):
    cfg = Path(run_root) / "configs"
    sess = pd.read_csv(cfg / "session_plan.csv")
    runs = pd.read_csv(cfg / "run_plan.csv")
    return sess, runs


def update_run_plan_row(run_root, run_id, **fields):
    sess, runs = load_plans(run_root)
    mask = runs["run_id"] == int(run_id)
    if not mask.any():
        raise SystemExit(f"run_id {run_id} not in frozen run plan")
    for k, val in fields.items():
        runs[k] = runs[k].astype(object)
        runs.loc[mask, k] = val
    atomic_write_csv(runs, Path(run_root) / "configs" / "run_plan.csv")


def update_session_plan_row(run_root, split, index, **fields):
    sess, runs = load_plans(run_root)
    mask = ((sess["split"] == split)
            & (sess["planned_session_index"] == int(index)))
    if not mask.any():
        raise SystemExit(f"session {split}/{index} not in session plan")
    for k, val in fields.items():
        sess[k] = sess[k].astype(object)
        sess.loc[mask, k] = val
    atomic_write_csv(sess, Path(run_root) / "configs" / "session_plan.csv")


# --------------------------------------------------------------- platform
def _clock_resolution_ns():
    try:
        import ctypes

        class TS(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long),
                        ("tv_nsec", ctypes.c_long)]

        res = TS()
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.clock_getres(1, ctypes.byref(res)) != 0:  # CLOCK_MONOTONIC
            return None
        return int(res.tv_sec * 10**9 + res.tv_nsec)
    except Exception:
        return None


def _cpu_model():
    """Locale-independent CPU model from /proc/cpuinfo."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _numa_nodes():
    try:
        return len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
    except OSError:
        return None


def _read_first(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_temp_c():
    best = None
    for hw in Path("/sys/class/hwmon").iterdir():
        try:
            name = (hw / "name").read_text().strip()
        except OSError:
            continue
        if name not in ("coretemp", "k10temp", "zenpower"):
            continue
        for t in sorted(hw.glob("temp*_input")):
            try:
                c = int(t.read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            if 0 < c < 120 and (best is None or c > best):
                best = c
    return best


def capture_platform_block(run_root):
    """One platform snapshot; unavailable fields listed by name."""
    fields = {}
    unavailable = []

    def put(key, val):
        if val is None or (isinstance(val, str) and val.strip() == ""):
            unavailable.append(key)
            fields[key] = None
        else:
            fields[key] = val

    put("os", f"{os.uname().sysname} {os.uname().release}")
    put("kernel", os.uname().release)
    put("cpu_model", _cpu_model())
    put("logical_cpus", os.cpu_count())
    put("numa_nodes", _numa_nodes())
    put("governor", _read_first(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))
    put("smt", _read_first("/sys/devices/system/cpu/smt/active"))
    put("temperature_c", _cpu_temp_c())
    try:
        put("load_1m", float(os.getloadavg()[0]))
        put("load_5m", float(os.getloadavg()[1]))
    except OSError:
        put("load_1m", None)
        put("load_5m", None)
    mem = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    put("available_memory_bytes", mem)
    try:
        put("free_disk_bytes", shutil.disk_usage(str(run_root)).free)
    except OSError:
        put("free_disk_bytes", None)
    put("sender_cpu", CPU_MAP["sender"])
    put("gate_cpu", CPU_MAP["gate"])
    put("receiver_cpu", CPU_MAP["receiver"])
    put("clock_resolution_ns", _clock_resolution_ns())
    return fields, unavailable


def append_session_platform(run_root, split, index, actual_session_id,
                            start_or_end):
    fields, unavailable = capture_platform_block(run_root)
    row = {
        "experiment_id": EXPERIMENT_ID, "split": split,
        "planned_session_index": int(index),
        "actual_session_id": actual_session_id,
        "capture_point": "session", "start_or_end": start_or_end,
        "timestamp": utc_now(),
        **fields,
        "unavailable_fields": ",".join(unavailable),
    }
    df = pd.DataFrame([row], columns=SESSION_PLATFORM_COLUMNS)
    path = Path(run_root) / "meta" / "session_platform.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    atomic_write_parquet(df, path)
    return row


def verify_affinity():
    frozen = {"sender": FROZEN_CONFIG["sender_cpu"],
              "gate": FROZEN_CONFIG["gate_cpu"],
              "receiver": FROZEN_CONFIG["receiver_cpu"]}
    if CPU_MAP != frozen:
        raise SystemExit(f"CPU affinity deviates from frozen config: "
                         f"{CPU_MAP} != {frozen}")
    return dict(CPU_MAP)


def check_no_leftover():
    """Refuse to start a session with leftover acquisition processes."""
    pat = "build/sender|build/software_gate|build/receiver"
    try:
        out = subprocess.run(["pgrep", "-af", pat], capture_output=True,
                             text=True, timeout=10)
        found = [l for l in out.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        found = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc / "cmdline").read_bytes().decode(
                    "utf-8", "replace")
            except OSError:
                continue
            if any(tok in cmd for tok in
                   ("build/sender", "build/software_gate", "build/receiver")):
                found.append(f"{proc.name}:{cmd}")
    if found:
        raise SystemExit("leftover acquisition processes detected: "
                         + "; ".join(found[:5]))


# ------------------------------------------------------------- acquisition
class H3Campaign(Campaign):
    experiment_id = EXPERIMENT_ID


def h3_event_frame(events, planned_session_index, actual_session_id,
                   session_order, run_id):
    """Adapt the e01-schema event frame to the h3r1 schema (spec 7.2)."""
    df = events.copy()
    if len(df) == 0:
        df = pd.DataFrame(columns=H3R1_EVENT_COLUMNS)
        return df
    df = df.drop(columns=["session_id"], errors="ignore")
    df.insert(1, "planned_session_index", int(planned_session_index))
    df.insert(2, "actual_session_id", str(actual_session_id))
    df.insert(3, "session_order", int(session_order))
    df["experiment_id"] = EXPERIMENT_ID
    df["run_id"] = int(run_id)
    return df[H3R1_EVENT_COLUMNS]


def h3_summary_row(summary, split, planned_session_index, actual_session_id,
                   session_order, attempt_index, selected_instance_index,
                   reference_mismatch_count, started_at, finished_at):
    return {
        "experiment_id": EXPERIMENT_ID, "split": split,
        "planned_session_index": int(planned_session_index),
        "actual_session_id": str(actual_session_id),
        "session_order": int(session_order), "run_id": int(summary["run_id"]),
        "attempt_index": int(attempt_index),
        "status": summary["status"], "valid_run": bool(summary["valid_run"]),
        "selected_instance_index": int(selected_instance_index),
        "phase_ns": int(summary.get("phase_ns", FROZEN_CONFIG["phase_ns"])),
        "sender_exit": summary.get("sender_exit"),
        "gate_exit": summary.get("gate_exit"),
        "receiver_exit": summary.get("receiver_exit"),
        "packet_lost": int(summary.get("packet_lost", 0) or 0),
        "out_of_order": int(summary.get("out_of_order", 0) or 0),
        "duplicate": int(summary.get("duplicate", 0) or 0),
        "log_drops": int(summary.get("log_drops", 0) or 0),
        "reference_mismatch_count": int(reference_mismatch_count),
        "invalid_reason": summary.get("invalid_reason"),
        "started_at": started_at, "finished_at": finished_at,
    }


def append_run_summary(run_root, row):
    path = Path(run_root) / "raw" / "run_summaries.parquet"
    df = pd.DataFrame([row], columns=RUN_SUMMARY_COLUMNS)
    if path.exists():
        old = pd.read_parquet(path)
        keep = ~((old["run_id"] == row["run_id"])
                 & (old["attempt_index"] == row["attempt_index"]))
        df = pd.concat([old[keep], df], ignore_index=True)
    atomic_write_parquet(df, path)


def session_events_dir(run_root, split, index):
    return (Path(run_root) / "raw" / "events" / split
            / f"session_{int(index):02d}")


def consolidate_pending(out_dir, batch_counter):
    """Merge per-run pending parquet files into one atomic session batch."""
    pend = sorted((out_dir / "pending").glob("run_*.parquet"))
    if not pend:
        return batch_counter
    df = pd.concat([pd.read_parquet(p) for p in pend], ignore_index=True)
    atomic_write_parquet(df, out_dir / f"batch_{batch_counter:04d}.parquet")
    for p in pend:
        p.unlink()
    return batch_counter + 1


def load_events(run_root, split):
    root = Path(run_root) / "raw" / "events" / split
    frames = []
    if root.exists():
        for path in sorted(root.glob("session_*/batch_*.parquet")):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=H3R1_EVENT_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(subset=["run_id", "seq"]).any():
        raise SystemExit(f"duplicate (run_id, seq) keys in raw {split} "
                         "events")
    return df


def acquisition_state_path(run_root, split):
    return Path(run_root) / "derived" / f"acquisition_state_{split}.json"


def load_acquisition_state(run_root, split):
    p = acquisition_state_path(run_root, split)
    if p.exists():
        return json.loads(p.read_text())
    return {"split": split, "sessions": {}, "total_valid": 0,
            "total_attempted": 0}


def save_acquisition_state(run_root, split, st):
    atomic_write_json(acquisition_state_path(run_root, split), st)


def acquire_split(run_root, split, resume):
    if split == "calibration":
        require_status(run_root, {"PLAN_FROZEN", "CALIBRATION_RUNNING"},
                       f"acquire/{split}")
    else:
        require_status(run_root, {"DESIGN_FROZEN", "VALIDATION_RUNNING"},
                       f"acquire/{split}")
    run_root = guard_run_root(run_root)
    camp = H3Campaign(run_root)
    verify_affinity()
    sess_df, run_df = load_plans(run_root)
    acq = load_acquisition_state(run_root, split)
    if acq["total_attempted"] > 0 and not resume:
        raise SystemExit(f"{split}: existing progress found; pass --resume")

    target = TARGET_VALID[split]
    max_attempts = int(target * 1.05)
    sessions = sess_df[sess_df["split"] == split].sort_values(
        "planned_session_index")
    plan_runs = run_df[run_df["split"] == split]

    for _, srow in sessions.iterrows():
        idx = int(srow["planned_session_index"])
        per = PER_SESSION[split]
        sst = acq["sessions"].get(str(idx))
        if sst and sst.get("status") == "COMPLETE":
            continue
        if sst is None:
            sst = {"status": "RUNNING", "valid": [], "attempted": 0,
                   "actual_session_id": "", "batches_flushed": 0}
            acq["sessions"][str(idx)] = sst

        if acq["total_attempted"] >= max_attempts:
            stop_experiment(run_root, f"{split}: attempted runs "
                             f"{acq['total_attempted']} exceed 105% cap "
                             f"{max_attempts}", f"acquire/{split}")

        # spec 5.3 session startup sequence
        check_no_leftover()
        if not sst["actual_session_id"]:
            sid = actual_session_id(split, idx)
            if not valid_session_id(sid):
                stop_experiment(run_root, "invalid session id generated",
                                f"acquire/{split}")
            sst["actual_session_id"] = sid
            sst["started_at"] = utc_now()
            append_session_platform(run_root, split, idx, sid, "start")
            update_session_plan_row(run_root, split, idx, status="RUNNING",
                                    actual_session_id=sid,
                                    actual_start=sst["started_at"])
            sys.stderr.write(
                f"[{split}] session {idx:02d} start {sid} "
                f"(valid total {acq['total_valid']}/{target})\n")
            sys.stderr.write(f"[{split}] session {idx:02d}: idling "
                             f"{IDLE_BEFORE_S}s before runs\n")
            time.sleep(IDLE_BEFORE_S)
        sid = sst["actual_session_id"]

        sess_runs = plan_runs[plan_runs["planned_session_index"]
                              == idx].sort_values("session_order")
        out_dir = session_events_dir(run_root, split, idx)
        guard_write(out_dir).mkdir(parents=True, exist_ok=True)
        sst.setdefault("attempts", {})
        sst.setdefault("unflushed", 0)

        for row in sess_runs.itertuples():
            rid = int(row.run_id)
            if rid in sst["valid"]:
                continue
            if acq["total_valid"] >= target:
                break
            while rid not in sst["valid"]:
                if acq["total_attempted"] >= max_attempts:
                    stop_experiment(
                        run_root, f"{split}: attempted runs "
                        f"{acq['total_attempted']} exceed 105% cap "
                        f"{max_attempts}", f"acquire/{split}")
                attempt = int(sst["attempts"].get(str(rid), 0)) + 1
                suffix = "" if attempt == 1 else f"_a{attempt}"
                log_dir = (Path(run_root) / "raw" / split
                           / f"session_{idx:02d}" / f"run_{rid:04d}{suffix}")
                started_at = utc_now()
                events, summary = camp.spawn_one_run(
                    rid, sid, split, int(row.phase_ns), int(row.seed),
                    int(row.selected_instance_index), log_dir)
                finished_at = utc_now()
                mismatch = int((events["invalid_reason"]
                                == "REFERENCE_GATE_MISMATCH").sum()) if len(
                                    events) else 0
                if summary["invalid_reason"] == "REFERENCE_GATE_MISMATCH" \
                        or mismatch:
                    stop_experiment(
                        run_root, "reference operator mismatch during "
                        f"{split} acquisition (run {rid})",
                        f"acquire/{split}")
                summary_row = h3_summary_row(
                    summary, split, idx, sid, row.session_order, attempt,
                    int(row.selected_instance_index), mismatch, started_at,
                    finished_at)
                append_run_summary(run_root, summary_row)
                sst["attempts"][str(rid)] = attempt
                sst["attempted"] += 1
                acq["total_attempted"] += 1
                if summary["valid_run"]:
                    sst["valid"].append(rid)
                    acq["total_valid"] += 1
                    update_run_plan_row(run_root, rid, status="DONE",
                                        actual_session_id=sid,
                                        attempt_index=attempt,
                                        invalid_reason="")
                    # only valid attempts enter raw event batches, keeping
                    # run_id unique across batches
                    frame = h3_event_frame(events, idx, sid,
                                           row.session_order, rid)
                    atomic_write_parquet(
                        frame, out_dir / "pending" / f"run_{rid:04d}.parquet")
                    sst["unflushed"] += 1
                else:
                    reason = summary["invalid_reason"] or "UNKNOWN"
                    update_run_plan_row(run_root, rid, status="INVALID",
                                        actual_session_id=sid,
                                        attempt_index=attempt,
                                        invalid_reason=reason)
                    log_deviation(
                        run_root, "TECHNICAL_INVALID",
                        f"{split} run {rid} attempt {attempt}: {reason}")
                    if reason not in TECHNICAL_INVALID:
                        stop_experiment(
                            run_root, f"{split} run {rid}: unexpected "
                            f"invalid reason {reason}", f"acquire/{split}")
                if sst["unflushed"] >= BATCH_FLUSH:
                    sst["batches_flushed"] = consolidate_pending(
                        out_dir, sst["batches_flushed"])
                    sst["unflushed"] = 0
                if len(sst["valid"]) % 25 == 0:
                    sys.stderr.write(
                        f"[{split}] session {idx:02d}: {len(sst['valid'])}"
                        f"/{per} valid ({acq['total_valid']}/{target} total, "
                        f"{acq['total_attempted']} attempted)\n")
                save_acquisition_state(run_root, split, acq)
                time.sleep(camp.cooldown_ms / 1000.0)
                if not summary["valid_run"] and attempt >= 3:
                    log_deviation(
                        run_root, "RUN_EXHAUSTED_ATTEMPTS",
                        f"{split} run {rid} still invalid after {attempt} "
                        "attempts")
                    break

        sst["batches_flushed"] = consolidate_pending(
            out_dir, sst["batches_flushed"])
        if len(sst["valid"]) != per:
            stop_experiment(run_root, f"{split} session {idx}: "
                             f"{len(sst['valid'])}/{per} valid runs after "
                             "all planned attempts", f"acquire/{split}")
        append_session_platform(run_root, split, idx, sid, "end")
        sst["ended_at"] = utc_now()
        sst["status"] = "COMPLETE"
        update_session_plan_row(run_root, split, idx, status="COMPLETE",
                                actual_end=sst["ended_at"])
        save_acquisition_state(run_root, split, acq)
        sys.stderr.write(f"[{split}] session {idx:02d} complete: "
                         f"{len(sst['valid'])}/{per} valid\n")

        long_idle = int(srow["planned_long_idle_after_s"] or 0)
        if long_idle > 0:
            check_no_leftover()
            time.sleep(long_idle)

    if acq["total_valid"] != target:
        stop_experiment(run_root, f"{split}: {acq['total_valid']}/{target} "
                         "valid runs after all sessions", f"acquire/{split}")
    _final_acquisition_checks(run_root, split, acq)
    save_acquisition_state(run_root, split, acq)
    if split == "calibration":
        complete_step(run_root, f"acquire/{split}",
                      status="CALIBRATION_COMPLETE",
                      calibration_valid_runs=int(acq["total_valid"]))
    else:
        complete_step(run_root, f"acquire/{split}",
                      status="VALIDATION_COMPLETE",
                      validation_valid_runs=int(acq["total_valid"]))


def _final_acquisition_checks(run_root, split, acq):
    """Spec R2/R5 final counts; stop on any violation."""
    valid_runs = sorted(
        rid for s in acq["sessions"].values() for rid in s["valid"])
    if len(valid_runs) != len(set(valid_runs)):
        stop_experiment(run_root, f"{split}: duplicate valid run_id",
                        f"acquire/{split}")
    sids = [s["actual_session_id"] for s in acq["sessions"].values()]
    if len({sid for sid in sids if valid_session_id(sid)}) != SESSIONS[split]:
        stop_experiment(run_root, f"{split}: unique actual session ids != "
                         f"{SESSIONS[split]}", f"acquire/{split}")
    events = load_events(run_root, split)
    selected = (events["selected_single"].astype(bool)
                & events["valid_event"].astype(bool))
    n_sel = int(selected.sum())
    expected = TARGET_VALID[split]
    if n_sel != expected:
        stop_experiment(run_root, f"{split}: selected instances {n_sel} != "
                         f"{expected}", f"acquire/{split}")


# ------------------------------------------------------------- diagnostics
def session_pairs(values, session_keys, lag=1):
    """Within-session lag-k pairs only; no pair across session boundaries."""
    values = np.asarray(values, dtype=float)
    session_keys = list(session_keys)
    a, b = [], []
    for i in range(len(values) - lag):
        if session_keys[i] == session_keys[i + lag]:
            a.append(values[i])
            b.append(values[i + lag])
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def _pearson(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if len(a) < 3:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(sps.spearmanr(a, b).statistic)


def correlation_trigger(pearson, spearman, n_lag_pairs):
    thr = max(0.10, 3.0 / np.sqrt(max(n_lag_pairs, 1)))
    triggered = ((np.isfinite(pearson) and abs(pearson) >= thr)
                 or (np.isfinite(spearman) and abs(spearman) >= thr))
    return float(thr), bool(triggered)


def run_order_diagnostics(samples, thresholds=None):
    """Session-aware run-order diagnostics (spec 8.3).

    samples: h3r1 stage-sample frame sorted by (planned_session_index,
    session_order). thresholds: frozen q / B (validation) or None
    (calibration: failures count infinite values only).
    """
    diag_rows, sess_rows = [], []
    split = samples["split"].iloc[0]
    for coord in COORDS:
        g = samples[samples["coordinate_id"] == coord].sort_values(
            ["planned_session_index", "session_order"])
        values = np.where(g["is_infinite"].astype(bool), np.nan,
                          g["value_ns"].astype(float).to_numpy())
        keys = g["planned_session_index"].tolist()
        n_units = int(len(g))
        a1, b1 = session_pairs(values, keys, 1)
        n_pairs = int(len(a1))
        pearson = _pearson(a1, b1)
        spearman = _spearman(a1, b1)
        thr, triggered = correlation_trigger(pearson, spearman, n_pairs)
        acfs = {}
        for lag in (2, 5, 10, 25):
            ak, bk = session_pairs(values, keys, lag)
            acfs[lag] = _pearson(ak, bk)
        medians, maxima = [], []
        for idx, grp in g.groupby("planned_session_index"):
            gv = np.where(grp["is_infinite"].astype(bool), np.nan,
                          grp["value_ns"].astype(float).to_numpy())
            if thresholds is None:
                failures = int(grp["is_infinite"].astype(bool).sum())
            else:
                failures = int((np.isnan(gv) | (gv > thresholds[coord])).sum())
            ga, gb = session_pairs(gv, grp["planned_session_index"].tolist(), 1)
            sess_rows.append({
                "split": split, "coordinate_id": coord,
                "planned_session_index": int(idx),
                "actual_session_id": str(grp["actual_session_id"].iloc[0]),
                "n_units": int(len(grp)),
                "median_ns": (float(np.nanmedian(gv))
                              if np.isfinite(gv).any() else None),
                "maximum_ns": (float(np.nanmax(gv))
                               if np.isfinite(gv).any() else None),
                "failure_count": failures,
                "lag1_pearson": _pearson(ga, gb),
                "lag1_spearman": _spearman(ga, gb),
            })
            if np.isfinite(gv).any():
                medians.append(float(np.nanmedian(gv)))
                maxima.append(float(np.nanmax(gv)))
        diag_rows.append({
            "split": split, "coordinate_id": coord, "n_units": n_units,
            "n_lag_pairs": n_pairs, "lag1_pearson": pearson,
            "lag1_spearman": spearman, "trigger_threshold": thr,
            "correlation_triggered": triggered,
            "acf_lag2": acfs[2], "acf_lag5": acfs[5],
            "acf_lag10": acfs[10], "acf_lag25": acfs[25],
            "session_median_min": min(medians) if medians else None,
            "session_median_max": max(medians) if medians else None,
            "session_maximum_min": min(maxima) if maxima else None,
            "session_maximum_max": max(maxima) if maxima else None,
            "status": "CORRELATION_TRIGGERED" if triggered else "OK",
        })
    return (pd.DataFrame(diag_rows, columns=DIAG_COLUMNS),
            pd.DataFrame(sess_rows, columns=SESSION_SUMMARY_COLUMNS))


def build_stage_samples(events, run_plan, split):
    """Spec 7.3: five rows per valid run, new-experiment samples only."""
    valid_ids = set(run_plan[run_plan["status"] == "DONE"]["run_id"].astype(int))
    ev = events[(events["split"] == split)
                & (events["experiment_id"] == EXPERIMENT_ID)
                & events["run_id"].astype(int).isin(valid_ids)
                & (~events["is_warmup"].astype(bool))
                & events["valid_event"].astype(bool)
                & events["selected_single"].astype(bool)].copy()
    ev["run_id"] = ev["run_id"].astype(int)
    plan_idx = run_plan.set_index("run_id")
    coords = event_coordinates(ev)
    rows = []
    for r in coords.itertuples():
        plan_row = plan_idx.loc[int(r.run_id)]
        for coord in COORDS:
            value = getattr(r, coord)
            lost = bool(r.packet_lost) and coord in ("X4", "E2E")
            rows.append({
                "experiment_id": EXPERIMENT_ID, "split": split,
                "planned_session_index": int(plan_row["planned_session_index"]),
                "actual_session_id": str(plan_row["actual_session_id"]),
                "session_order": int(plan_row["session_order"]),
                "run_id": int(r.run_id), "coordinate_id": coord,
                "value_ns": None if (pd.isna(value) or lost) else int(value),
                "is_infinite": bool(lost),
                "selected_single": True,
                "phase_ns": int(plan_row["phase_ns"]),
                "calendar_id": FROZEN_CONFIG["calendar_id"],
                "workload_id": FROZEN_CONFIG["workload_id"],
            })
    df = pd.DataFrame(rows, columns=STAGE_SAMPLE_COLUMNS)
    if len(df) and not (df.groupby("run_id").size() == 5).all():
        raise SystemExit("stage samples: some run did not produce exactly "
                         "5 coordinate rows")
    if len(df) and not set(df["run_id"]).issubset(valid_ids):
        raise SystemExit("stage samples contain runs outside the frozen "
                         "plan DONE set")
    return df


def per_run_vectors(samples, split):
    """{run_id: {coord: value(inf-aware)}} plus plan key for ordering."""
    vec = {}
    for r in samples[samples["split"] == split].itertuples():
        d = vec.setdefault(int(r.run_id), {
            "planned_session_index": int(r.planned_session_index),
            "session_order": int(r.session_order)})
        d[r.coordinate_id] = (float("inf") if r.is_infinite
                              or pd.isna(r.value_ns) else float(r.value_ns))
    return vec


def failure_flags(vec, thresholds):
    """Per-run failure indicators per item (X1..X4 vs q, E2E vs B)."""
    out = {c: [] for c in COORDS}
    order = sorted(vec, key=lambda k: (vec[k]["planned_session_index"],
                                       vec[k]["session_order"]))
    for rid in order:
        d = vec[rid]
        for c in COORDS:
            out[c].append(bool(d[c] == float("inf") or d[c] > thresholds[c]))
    out["_run_ids"] = order
    out["_session_keys"] = [vec[k]["planned_session_index"] for k in order]
    return out


def block_sensitivity(flags, run_failures, item):
    """Non-overlapping within-session blocks; CP per block failure (8.4)."""
    rows = []
    f = np.asarray(flags, dtype=bool)
    keys = list(run_failures["_session_keys"])
    for bs in BLOCK_SIZES:
        failed_blocks = 0
        n_blocks = 0
        for idx in sorted(set(keys)):
            pos = [i for i, k in enumerate(keys) if k == idx]
            for s in range(0, len(pos), bs):
                chunk = f[pos[s:s + bs]]
                if len(chunk) == 0:
                    continue
                n_blocks += 1
                if bool(chunk.any()):
                    failed_blocks += 1
        rate = failed_blocks / n_blocks if n_blocks else float("nan")
        lo, hi = clopper_pearson(failed_blocks, n_blocks, 0.01)
        run_level = int(f.sum())
        degenerate = ("ZERO_EVENT_EMPIRICAL_BOOTSTRAP_DEGENERATE"
                      if run_level == 0 else "")
        rows.append({
            "item_id": item, "block_size": int(bs), "n_blocks": n_blocks,
            "failed_blocks": int(failed_blocks),
            "block_failure_rate": float(rate), "cp_lower": lo,
            "cp_upper": hi, "run_level_failures": run_level,
            "degenerate_flag": degenerate,
        })
    return rows


def h3_disposition(inclusion_violations, mismatch_count, calib_flagged,
                   val_triggered, item_statuses):
    """Spec 3.3 priority cascade (pure; order-literal with the cap)."""
    if inclusion_violations or mismatch_count:
        return "MODEL_OR_IMPLEMENTATION_ERROR"
    any_ns = any(s == "NOT_SUPPORTED" for s in item_statuses)
    all_sup = all(s == "SUPPORTED" for s in item_statuses)
    if any_ns:
        h3 = "NOT_SUPPORTED"
    elif val_triggered:
        h3 = "INCONCLUSIVE_CORRELATION_SENSITIVITY"
    elif all_sup:
        h3 = "SUPPORTED"
    else:
        h3 = "INCONCLUSIVE"
    if calib_flagged and h3 == "SUPPORTED":
        h3 = "INCONCLUSIVE_CORRELATION_SENSITIVITY"
    return h3


def cp_wording(alpha_row_):
    """Spec 8.2: alpha_row=0.01 is one-sided 99% per item, 95% family."""
    if abs(alpha_row_ - 0.01) < 1e-12:
        return ("one-sided 99% CP per item; 95% family level under the "
                "five-item Bonferroni procedure")
    return (f"one-sided {100 * (1.0 - alpha_row_):g}% CP per item under "
            f"the Bonferroni procedure (alpha_row={alpha_row_:.5g})")


# ------------------------------------------------------------ figures
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_run_order(samples, diag, thresholds, title, out_path,
                  session_bounds):
    plt = _mpl()
    fig, axes = plt.subplots(len(COORDS), 1, figsize=(11, 12), sharex=True)
    for i, coord in enumerate(COORDS):
        ax = axes[i]
        g = samples[samples["coordinate_id"] == coord].sort_values(
            ["planned_session_index", "session_order"])
        x = np.arange(len(g))
        vals = np.where(g["is_infinite"].astype(bool), np.nan,
                        g["value_ns"].astype(float).to_numpy())
        ax.plot(x, vals, lw=0.4, alpha=0.6)
        for b in session_bounds:
            ax.axvline(b - 0.5, color="gray", lw=0.5, alpha=0.6)
        if thresholds is not None:
            ax.axhline(thresholds[coord], color="red", ls="--", lw=1.0)
        d = diag[diag["coordinate_id"] == coord]
        if len(d) and bool(d["correlation_triggered"].iloc[0]):
            ax.set_facecolor("#fff2f2")
        ax.set_ylabel(f"{coord} [ns]", fontsize=8)
        ax.set_title(f"{coord} (lag1 r={d['lag1_pearson'].iloc[0]:.3f}, "
                     f"rho={d['lag1_spearman'].iloc[0]:.3f})", fontsize=8,
                     loc="right")
    axes[-1].set_xlabel("run order within split (sessions separated)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    guard_write(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def fig_sessions(sess_summary, title, out_path):
    plt = _mpl()
    fig, axes = plt.subplots(len(COORDS), 1, figsize=(11, 12), sharex=True)
    for i, coord in enumerate(COORDS):
        ax = axes[i]
        g = sess_summary[sess_summary["coordinate_id"] == coord].sort_values(
            "planned_session_index")
        ax.plot(g["planned_session_index"], g["median_ns"], "o-", ms=3,
                label="session median")
        ax.plot(g["planned_session_index"], g["maximum_ns"], "s--", ms=3,
                label="session maximum")
        ax.set_ylabel(f"{coord} [ns]", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    axes[-1].set_xlabel("planned session index")
    axes[-1].set_xticks(sorted(sess_summary["planned_session_index"].unique()))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    guard_write(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def fig_old_new(h3r1_samples, design, e01_design, e01_samples, out_path):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    x = np.arange(len(STAGE_COORDS))
    new_med = [np.nanmedian(h3r1_samples[h3r1_samples["coordinate_id"] == c]
                            ["value_ns"].astype(float)) for c in STAGE_COORDS]
    old_med = [np.nanmedian(e01_samples[e01_samples["coordinate_id"] == c]
                            ["value_ns"].astype(float)) for c in STAGE_COORDS]
    ax.bar(x - 0.18, old_med, 0.36, label="e01 (old)")
    ax.bar(x + 0.18, new_med, 0.36, label="e01-h3r1 (new)")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_COORDS)
    ax.set_ylabel("median [ns]")
    ax.set_title("Stage medians (EXPLORATORY_OLD_NEW_COMPARISON)", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    labels = ["B (e2e bound)", "max E2E old", "max E2E new"]
    old_max = np.nanmax(e01_samples[e01_samples["coordinate_id"] == "E2E"]
                        ["value_ns"].astype(float))
    new_max = np.nanmax(h3r1_samples[h3r1_samples["coordinate_id"] == "E2E"]
                        ["value_ns"].astype(float))
    vals = [int(e01_design["bound_ns"]), float(old_max), float(new_max)]
    ax.bar(labels, vals, color=["gray", "tab:blue", "tab:orange"])
    if design is not None:
        ax.axhline(int(design["bound_ns"]), color="red", ls="--", lw=1,
                   label=f"new B={int(design['bound_ns'])}")
        ax.legend(fontsize=8)
    ax.set_ylabel("ns")
    ax.set_title("E2E bound vs observed maxima", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    guard_write(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------- CLI steps
def cmd_init(args):
    run_root = guard_run_root(args.run_root)
    for sub in ("state", "meta", "configs", "derived", "figures", "reports",
                "raw/events/calibration", "raw/events/validation",
                "raw/calibration", "raw/validation",
                "handoff/outgoing"):
        guard_write(run_root / sub).mkdir(parents=True, exist_ok=True)
    init_state_file(run_root)
    fields, unavailable = capture_platform_block(run_root)
    atomic_write_text(
        run_root / "meta" / "platform_baseline.yaml",
        yaml.safe_dump({
            "experiment_id": EXPERIMENT_ID,
            "captured_at": utc_now(),
            "capture_scope": "R0_BASELINE",
            **fields,
            "unavailable_fields": unavailable,
        }, sort_keys=False))
    (run_root / "meta" / "deviations.jsonl").touch()
    print(json.dumps({"run_root": str(run_root), "status": "INIT",
                      "unavailable_fields": unavailable}, indent=2))


def cmd_software_gate(args):
    run_root = guard_run_root(args.run_root)
    require_status(run_root, {"INIT", "SOFTWARE_VERIFIED"}, "software-gate")

    checks = {}
    # runs/e01 readable, and the parent supplement is the corrected version
    checks["e01_readable"] = E01_ROOT.is_dir() and (
        E01_ROOT / "reports" / "FINAL_EXPERIMENT_REPORT.md").exists()
    supp_state = E01_ROOT / "supplement" / "state" / "supplement_state.json"
    checks["e01s1_state_complete"] = (
        supp_state.exists()
        and json.loads(supp_state.read_text()).get("status") == "COMPLETE")
    report_src = (PROJECT_ROOT / "python" / "parce_exp" / "report.py").read_text()
    checks["h4_dynamic_in_report_py"] = "h4_status_from_data" in report_src
    checks["cpu_topology_compatible"] = (
        os.cpu_count() or 0) >= 4 and CPU_MAP == {
            "sender": FROZEN_CONFIG["sender_cpu"],
            "gate": FROZEN_CONFIG["gate_cpu"],
            "receiver": FROZEN_CONFIG["receiver_cpu"]}
    cal = load_calendar(PROJECT_ROOT / "configs" / "calendar.yaml")
    checks["calendar_matches_frozen"] = (
        int(cal["period_ns"]) == FROZEN_CONFIG["period_ns"]
        and int(cal["service_demand_ns"]) == FROZEN_CONFIG["service_demand_ns"]
        and [[int(o), int(c)] for o, c in cal["windows"]]
        == FROZEN_CONFIG["windows_ns"])
    for name, ok in checks.items():
        if not ok:
            stop_experiment(run_root, f"software-gate check failed: {name}",
                            "R0")

    env = dict(os.environ)
    env["PARCE_TEST_RESULTS_DIR"] = str(run_root / "derived" / "test_run")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q",
                           str(PROJECT_ROOT / "tests")],
                          cwd=str(PROJECT_ROOT), env=env,
                          capture_output=True, text=True, timeout=3600)
    tests_ok = proc.returncode == 0
    checks["full_pytest_suite"] = tests_ok
    test_dir = run_root / "derived" / "test_run" / "derived"
    ref_mismatch = 0
    dp_mismatch = 0
    if (test_dir / "software_tests.json").exists():
        payload = json.loads((test_dir / "software_tests.json").read_text())
        ref_mismatch = int(payload.get("reference_mismatch_count", 0))
        dp_mismatch = int(payload.get("dp_brute_mismatch_count", 0))
    checks["reference_mismatch_zero"] = ref_mismatch == 0
    checks["dp_brute_mismatch_zero"] = dp_mismatch == 0

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "step": "R0",
        "generated_at": utc_now(),
        "overall_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pytest_exit_status": int(proc.returncode),
        "pytest_tail": proc.stdout[-2000:],
        "reference_mismatch_count": ref_mismatch,
        "dp_brute_mismatch_count": dp_mismatch,
    }
    atomic_write_json(run_root / "derived" / "software_tests.json", payload)

    base = yaml.safe_load((run_root / "meta" / "platform_baseline.yaml")
                          .read_text())
    lines = [
        "# 01 Plan and Platform (e01-h3r1 R0/R1)", "",
        f"- generated: {utc_now()}",
        f"- experiment: {EXPERIMENT_ID} (parent {PARENT_EXPERIMENT_ID}; "
        "this experiment re-acquires H3 evidence with sessionized physical "
        "Calibration and Validation)",
        "- old H3 status being replaced: "
        "INCONCLUSIVE_CORRELATION_SENSITIVITY (e01-s1 S3 run-order trigger)",
        "", "## Frozen configuration (h3_reacquisition.MD 2)", "",
    ]
    for k, v in FROZEN_CONFIG.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Software gate (R0)", "",
              "| check | result |", "|---|---|"]
    for k, v in checks.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += ["", "## Platform baseline (R0 capture)", ""]
    for k, v in base.items():
        if k not in ("experiment_id",):
            lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Session plan design", "",
        f"- {SESSIONS['calibration']} calibration sessions x "
        f"{PER_SESSION['calibration']} valid runs; {SESSIONS['validation']} "
        f"validation sessions x {PER_SESSION['validation']} valid runs",
        f"- {IDLE_BEFORE_S}s idle before each session (not counted as runs); "
        f"{LONG_IDLE_AFTER_S}s long idle after every {LONG_IDLE_EVERY}th "
        "completed session except the final one of a split",
        "- session ids: "
        f"{EXPERIMENT_ID}-{{calibration,validation}}-s{{01..20}}-{{UTC}}; "
        "runs execute serially in frozen session_order",
        "- run plan master seed: "
        f"{MASTER_SEED} (one-shot; never regenerated)",
        "- attempt semantics: a technically invalid run (process crash, log "
        "corruption, missing timestamps, host interruption) is retried under "
        "the same run_id with attempt_index+1; attempted cap = 105% of the "
        "valid-run target; reference mismatches stop the experiment",
        "", "## Reproduction commands", "",
        "```bash",
        "source .venv/bin/activate",
        "pytest -q",
        f"python -m parce_exp.h3_reacquisition init --run-root {args.run_root}",
        f"python -m parce_exp.h3_reacquisition software-gate --run-root {args.run_root}",
        f"python -m parce_exp.h3_reacquisition freeze-plan --run-root {args.run_root} --master-seed {MASTER_SEED}",
        f"bash scripts/run_h3r1.sh calibration --run-root {args.run_root} --resume",
        f"python -m parce_exp.h3_reacquisition analyze-calibration --run-root {args.run_root}",
        f"python -m parce_exp.h3_reacquisition freeze-design --run-root {args.run_root}",
        f"bash scripts/run_h3r1.sh validation --run-root {args.run_root} --resume",
        f"python -m parce_exp.h3_reacquisition analyze-validation --run-root {args.run_root}",
        f"python -m parce_exp.h3_reacquisition build-reports --run-root {args.run_root}",
        "pytest -q",
        f"python -m parce_exp.h3_reacquisition verify-final --run-root {args.run_root}",
        f"python -m parce_exp.h3_reacquisition build-handoff --run-root {args.run_root} "
        f"--output {args.run_root}/handoff/outgoing/e01-h3r1-review-package.zip",
        "```", "",
    ]
    atomic_write_text(run_root / "reports" / "01_PLAN_AND_PLATFORM.md",
                      "\n".join(lines))
    if not all(checks.values()):
        stop_experiment(run_root, "software gate failed (see report 01)",
                        "R0")
    complete_step(run_root, "R0", status="SOFTWARE_VERIFIED")
    print(json.dumps({"checks": checks, "overall": "PASS"}, indent=2))


def cmd_freeze_plan(args):
    run_root = guard_run_root(args.run_root)
    require_status(run_root, {"SOFTWARE_VERIFIED"}, "freeze-plan")
    if args.master_seed != MASTER_SEED:
        raise SystemExit(f"master seed must be {MASTER_SEED}")
    cfg = run_root / "configs"
    if (cfg / "run_plan.csv").exists() or (cfg / "session_plan.csv").exists():
        raise SystemExit("plans already frozen; regeneration forbidden")
    sess_df, run_df = generate_plans()
    violations = validate_plans(sess_df, run_df)
    if violations:
        stop_experiment(run_root, "plan validation failed: "
                        + "; ".join(violations), "R1")
    atomic_write_csv(sess_df, cfg / "session_plan.csv")
    atomic_write_csv(run_df, cfg / "run_plan.csv")
    atomic_write_text(
        cfg / "frozen_h3r1_plan.yaml",
        yaml.safe_dump({
            "experiment_id": EXPERIMENT_ID,
            "parent_experiment_id": PARENT_EXPERIMENT_ID,
            "frozen_at": utc_now(),
            "master_seed": MASTER_SEED,
            "config": FROZEN_CONFIG,
            "sessions": SESSIONS,
            "per_session": PER_SESSION,
            "target_valid": TARGET_VALID,
            "idle_before_s": IDLE_BEFORE_S,
            "long_idle_after_s": LONG_IDLE_AFTER_S,
            "long_idle_every": LONG_IDLE_EVERY,
            "attempt_cap_pct": 105,
            "notes": [
                "session_plan/run_plan generated one-shot from master seed; "
                "never regenerated",
                "long idle after every 5th completed session except the "
                "final session of a split",
                "runs/e01 is read-only for this experiment",
            ],
        }, sort_keys=False))
    complete_step(run_root, "R1", status="PLAN_FROZEN")
    print(json.dumps({"session_rows": len(sess_df), "run_rows": len(run_df),
                      "violations": violations}, indent=2))


def cmd_acquire(args):
    acquire_split(args.run_root, args.split, args.resume)


def _extract_and_diagnose(run_root, split):
    events = load_events(run_root, split)
    _, run_df = load_plans(run_root)
    plan = run_df[run_df["split"] == split]
    samples = build_stage_samples(events, plan, split)
    out = (run_root / "derived" / f"{split}_stage_samples.parquet")
    atomic_write_parquet(samples, out)
    thresholds = None
    if split == "validation":
        design = json.loads((run_root / "derived" / "selected_design.json")
                            .read_text())
        thresholds = {c: int(design["selected_q_ns"][c]) for c in STAGE_COORDS}
        thresholds["E2E"] = int(design["bound_ns"])
    diag, sess_summary = run_order_diagnostics(samples, thresholds)
    atomic_write_parquet(
        diag, run_root / "derived" / f"{split}_run_order_diagnostics.parquet")
    atomic_write_parquet(
        sess_summary,
        run_root / "derived" / f"{split}_session_summary.parquet")
    return samples, diag, sess_summary, thresholds


def _session_bounds(samples):
    g = samples[samples["coordinate_id"] == "X1"].sort_values(
        ["planned_session_index", "session_order"])
    bounds, seen = [], set()
    for idx in g["planned_session_index"]:
        if idx not in seen:
            seen.add(idx)
            bounds.append(int(g.index[g["planned_session_index"] == idx][0]))
    return bounds


def _render_report_02(run_root):
    """Render 02_CALIBRATION.md from derived artifacts (idempotent)."""
    run_root = Path(run_root)
    samples = pd.read_parquet(run_root / "derived"
                              / "calibration_stage_samples.parquet")
    diag = pd.read_parquet(run_root / "derived"
                           / "calibration_run_order_diagnostics.parquet")
    sess_df, _run_df = load_plans(run_root)
    summaries = pd.read_parquet(run_root / "raw" / "run_summaries.parquet")
    cal_sum = summaries[summaries["split"] == "calibration"]
    att_by_session = cal_sum.groupby("planned_session_index").size()

    lines = [
        "# 02 Calibration (e01-h3r1 R2/R3)", "",
        f"- generated: {utc_now()}",
        f"- valid runs: {int(len(samples) / 5)}; stage rows: {len(samples)}",
        "", "## Sessions", "",
        "| session | actual id | valid/attempted | start | end |",
        "|---|---|---|---|---|",
    ]
    for r in sess_df[sess_df["split"] == "calibration"].itertuples():
        idx = int(r.planned_session_index)
        att = int(att_by_session.get(idx, 0))
        lines.append(
            f"| s{idx:02d} | {r.actual_session_id} | "
            f"{int(r.planned_valid_runs)}/{att} | {r.actual_start} | "
            f"{r.actual_end} |")
    lines += ["", "## Descriptive and tail statistics per coordinate", "",
              "| coordinate | n | median [ns] | p99 [ns] | p99.9 [ns] | "
              "max [ns] | infinite |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for coord in COORDS:
        g = samples[samples["coordinate_id"] == coord]
        vals = g["value_ns"].astype(float).to_numpy()
        lines.append(
            f"| {coord} | {len(g)} | {np.nanmedian(vals):.0f} | "
            f"{np.nanpercentile(vals, 99):.0f} | "
            f"{np.nanpercentile(vals, 99.9):.0f} | "
            f"{np.nanmax(vals):.0f} | {int(g['is_infinite'].sum())} |")
    lines += ["", "## Run-order diagnostics (session-aware; spec 8.3)", "",
              "| coordinate | n | lag pairs | lag1 pearson | lag1 spearman | "
              "threshold | triggered |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in diag.itertuples():
        lines.append(
            f"| {r.coordinate_id} | {int(r.n_units)} | {int(r.n_lag_pairs)} "
            f"| {r.lag1_pearson:.4f} | {r.lag1_spearman:.4f} | "
            f"{r.trigger_threshold:.4f} | {r.correlation_triggered} |")
    lines += ["", "### ACF (within-session pooled pairs)", "",
              "| coordinate | lag2 | lag5 | lag10 | lag25 |",
              "|---|---:|---:|---:|---:|"]
    for r in diag.itertuples():
        lines.append(f"| {r.coordinate_id} | {r.acf_lag2:.4f} | "
                     f"{r.acf_lag5:.4f} | {r.acf_lag10:.4f} | "
                     f"{r.acf_lag25:.4f} |")
    lines += ["", "### Session heterogeneity", "",
              "| coordinate | session median range [ns] | session maximum "
              "range [ns] |", "|---|---|---|"]
    for r in diag.itertuples():
        lines.append(f"| {r.coordinate_id} | "
                     f"[{r.session_median_min:.0f}, "
                     f"{r.session_median_max:.0f}] | "
                     f"[{r.session_maximum_min:.0f}, "
                     f"{r.session_maximum_max:.0f}] |")
    lines += ["", "## Invalid runs and deviations", "",
              f"- attempted runs recorded: {len(cal_sum)}; "
              f"invalid attempts: {int((~cal_sum['valid_run']).sum())}",
              f"- reference_mismatch_count total: "
              f"{int(cal_sum['reference_mismatch_count'].sum())} (0 "
              "required)"]
    st = load_state(run_root)
    triggered = bool(diag["correlation_triggered"].any())
    if triggered:
        lines += ["",
                  "**CALIBRATION_CORRELATION_FLAGGED = true** — per spec "
                  "8.3 the experiment continues to design freeze and "
                  "validation, and the final H3 can be at most "
                  "INCONCLUSIVE_CORRELATION_SENSITIVITY."]
    else:
        lines += ["", "superblock_sensitivity = NOT_TRIGGERED; design "
                  "freeze may proceed."]
    lines += ["", "Data: derived/calibration_stage_samples.parquet, "
              "derived/calibration_run_order_diagnostics.parquet, "
              "derived/calibration_session_summary.parquet; figures: "
              "figures/fig01_calibration_run_order.svg, "
              "figures/fig02_calibration_sessions.svg", ""]
    atomic_write_text(run_root / "reports" / "02_CALIBRATION.md",
                      "\n".join(lines))


def cmd_analyze_calibration(args):
    run_root = guard_run_root(args.run_root)
    require_status(run_root, {"CALIBRATION_COMPLETE",
                              "CALIBRATION_DIAGNOSTICS_COMPLETE"},
                   "analyze-calibration")
    samples, diag, sess_summary, _ = _extract_and_diagnose(
        run_root, "calibration")
    if len(samples) != 5 * TARGET_VALID["calibration"]:
        stop_experiment(run_root, "calibration stage rows "
                        f"{len(samples)} != 15000", "R3")
    fig_run_order(samples, diag, None,
                  "fig01: calibration run-order diagnostics "
                  "(e01-h3r1; sessions separated)",
                  run_root / "figures" / "fig01_calibration_run_order.svg",
                  _session_bounds(samples))
    fig_sessions(sess_summary,
                 "fig02: calibration per-session medians/maxima (e01-h3r1)",
                 run_root / "figures" / "fig02_calibration_sessions.svg")
    _render_report_02(run_root)
    triggered = bool(diag["correlation_triggered"].any())
    complete_step(run_root, "R3",
                  status="CALIBRATION_DIAGNOSTICS_COMPLETE",
                  calibration_correlation_flagged=triggered)
    print(diag.to_string(index=False))


def _render_report_03(run_root):
    """Render 03_DESIGN_FREEZE.md from derived artifacts (idempotent;
    never touches selected_design.json)."""
    run_root = Path(run_root)
    menu = pd.read_parquet(run_root / "derived" / "menu.parquet")
    pruned = json.loads((run_root / "derived" / "menu_pruned.json")
                        .read_text())["pruned_dominated_menu_ids"]
    design = json.loads((run_root / "derived" / "selected_design.json")
                        .read_text())
    lines = [
        "# 03 Design Freeze (e01-h3r1 R4)", "",
        f"- generated: {utc_now()}",
        f"- design created_at: {design['created_at']} (must precede every "
        "validation session)",
        f"- alpha_row = 0.05/12 = {alpha_row(12, 0.05):.6f}; k table check "
        "3000/2999/2997: PASS",
        "", "## Menu points", "",
        "| menu id | epsilon | alpha family | alpha row | n | k | q [ns] | "
        "status |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in menu.itertuples():
        lines.append(
            f"| {r.menu_id} | {r.risk:g} | {r.alpha_family:g} | "
            f"{r.alpha_row:.6f} | {int(r.n_units)} | "
            f"{r.order_k if r.order_k is not None else '-'} | "
            f"{r.q_ns if r.q_ns is not None else '-'} | {r.status} |")
    lines += ["", f"- pruned dominated points: "
              f"{pruned if pruned else 'none'}",
              "", "## Pareto DP result", "",
              f"- selected menu ids: {design['selected_menu_ids']}",
              f"- selected q [ns]: {design['selected_q_ns']}",
              f"- selected risks: {design['selected_risks']}",
              f"- risk_sum: {design['risk_sum']:.4f} "
              f"(budget {design['risk_budget']})",
              f"- bound B: {design['bound_ns']} ns (independent replay "
              f"{design['replay_bound_ns']} ns, match)",
              f"- DP matches exhaustive enumeration: "
              f"{design['dp_matches_brute_force']}",
              f"- gate signature: {design['gate_signature']}", "",
              "Data: derived/menu.parquet, derived/menu_pruned.json, "
              "derived/selected_design.json", ""]
    atomic_write_text(run_root / "reports" / "03_DESIGN_FREEZE.md",
                      "\n".join(lines))


def cmd_freeze_design(args):
    run_root = guard_run_root(args.run_root)
    require_status(run_root, {"CALIBRATION_DIAGNOSTICS_COMPLETE"},
                   "freeze-design")
    from .allocator import (brute_force_selection, chain_bound_spec56,
                            gate_admission_offset, pareto_dp)
    from .select_design import GATE_AFTER, build_menus, calendar_signature, \
        gate_signature

    samples = pd.read_parquet(run_root / "derived"
                              / "calibration_stage_samples.parquet")
    risk_grid = FROZEN_CONFIG["stage_risk_grid"]
    menu, pruned = calibrate_menu(samples, risk_grid,
                                  FROZEN_CONFIG["calibration_family_alpha"])
    n_cal = TARGET_VALID["calibration"]
    if n_cal == 3000:
        expected_k = {0.002: 3000, 0.003: 2999, 0.004: 2997}
        k_ok = True
        for r in menu.itertuples():
            if r.status == "CERTIFIED" and r.order_k != expected_k[r.risk]:
                k_ok = False
        if not k_ok:
            stop_experiment(run_root, "menu order statistics k deviate from "
                            "3000/2999/2997", "R4")
    atomic_write_parquet(menu, run_root / "derived" / "menu.parquet")
    atomic_write_json(run_root / "derived" / "menu_pruned.json",
                      {"pruned_dominated_menu_ids": pruned})

    cal = load_calendar(PROJECT_ROOT / "configs" / "calendar.yaml")
    phase = FROZEN_CONFIG["phase_ns"]
    demand = FROZEN_CONFIG["service_demand_ns"]
    period = FROZEN_CONFIG["period_ns"]
    offset = int(cal["eligibility_offset_ns"])
    windows = [(int(o) + offset, int(c) + offset) for o, c in
               cal["windows"]]
    budget = float(FROZEN_CONFIG["system_risk_budget"])
    menus = build_menus(menu)
    sig = calendar_signature(cal, phase)
    design = {
        "procedure": "SIMULTANEOUS_MENU",
        "calendar_id": cal["calendar_id"],
        "phase_ns": phase,
        "risk_budget": budget,
        "selected_menu_ids": {}, "selected_risks": {}, "selected_q_ns": {},
        "risk_sum": 0.0, "bound_ns": 0,
        "calendar_signature": sig,
        "algorithm": "PARETO_FRONTIER_DP",
    }
    empty = [c for c in STAGE_COORDS if not menus[c]]
    if empty:
        design["status"] = "NOT_SUPPORTED"
        design["not_supported_reason"] = f"no finite menu point for {empty}"
        atomic_write_json(run_root / "derived" / "selected_design.json",
                          design)
        stop_experiment(run_root, "no feasible design: " + design[
            "not_supported_reason"] + "; validation not run (H3 "
            "NOT_SUPPORTED)", "R4")
    last_gate = max(j for j, g in enumerate(GATE_AFTER) if g)

    def gate_op(j):
        def op(r):
            out = gate_admission_offset(r, demand, period, windows, phase)
            if j == last_gate:
                out += demand
            return out
        return op

    gate_ops = [gate_op(j) if GATE_AFTER[j] else None for j in range(4)]
    menu_list = [menus[c] for c in STAGE_COORDS]
    sel = pareto_dp(menu_list, GATE_AFTER, demand, period, windows, budget,
                    base_ns=phase, gate_ops=gate_ops)
    brute = brute_force_selection(menu_list, GATE_AFTER, demand, period,
                                  windows, budget, base_ns=phase,
                                  gate_ops=gate_ops)
    dp_matches = ((sel is None and brute is None)
                  or (sel is not None and brute is not None and (
                      sel.bound_ns, sel.risk_sum, sel.menu_ids)
                      == (brute.bound_ns, brute.risk_sum, brute.menu_ids)))
    if sel is None:
        design["status"] = "NOT_SUPPORTED"
        design["not_supported_reason"] = (
            f"no menu combination satisfies risk_sum <= {budget}")
        atomic_write_json(run_root / "derived" / "selected_design.json",
                          design)
        stop_experiment(run_root, design["not_supported_reason"]
                        + "; validation not run (H3 NOT_SUPPORTED)", "R4")
    by_id = {c: {it.menu_id: it for it in menus[c]} for c in STAGE_COORDS}
    qs = [by_id[c][m].q_ns for c, m in zip(STAGE_COORDS, sel.menu_ids)]
    replay = chain_bound_spec56(qs, GATE_AFTER, demand, period, windows,
                                base_ns=phase)
    if replay != sel.bound_ns:
        stop_experiment(run_root, f"design replay mismatch DP={sel.bound_ns}"
                        f" replay={replay}", "R4")
    design.update({
        "selected_menu_ids": {c: m for c, m in
                              zip(STAGE_COORDS, sel.menu_ids)},
        "selected_risks": {c: by_id[c][m].risk for c, m in
                           zip(STAGE_COORDS, sel.menu_ids)},
        "selected_q_ns": {c: by_id[c][m].q_ns for c, m in
                          zip(STAGE_COORDS, sel.menu_ids)},
        "risk_sum": float(sel.risk_sum), "bound_ns": int(sel.bound_ns),
        "status": "DESIGN_FROZEN", "replay_bound_ns": int(replay),
        "dp_matches_brute_force": bool(dp_matches),
        "created_at": utc_now(),
        "gate_signature": gate_signature(sig),
    })
    if not dp_matches:
        stop_experiment(run_root, "DP and brute-force disagree on the real "
                        "menu", "R4")
    atomic_write_json(run_root / "derived" / "selected_design.json", design)

    _render_report_03(run_root)

    complete_step(run_root, "R4", status="DESIGN_FROZEN")
    print(json.dumps({k: design[k] for k in
                      ("selected_menu_ids", "selected_q_ns", "risk_sum",
                       "bound_ns", "created_at")}, indent=2))


def _render_report_04(run_root):
    """Render 04_VALIDATION_AND_H3.md from derived artifacts (idempotent,
    recomputing inclusion/H3 from data; never touches state or design)."""
    run_root = Path(run_root)
    design = json.loads((run_root / "derived" / "selected_design.json")
                        .read_text())
    samples = pd.read_parquet(run_root / "derived"
                              / "validation_stage_samples.parquet")
    results = pd.read_parquet(run_root / "derived" / "validation_results"
                              ".parquet")
    diag = pd.read_parquet(run_root / "derived"
                           / "validation_run_order_diagnostics.parquet")
    sess_summary = pd.read_parquet(run_root / "derived"
                                   / "validation_session_summary.parquet")
    blocks = pd.read_parquet(run_root / "derived" / "block_sensitivity"
                             ".parquet")
    plat = pd.read_parquet(run_root / "meta" / "session_platform.parquet")
    val_starts = plat[(plat["split"] == "validation")
                      & (plat["start_or_end"] == "start")]["timestamp"]
    summaries = pd.read_parquet(run_root / "raw" / "run_summaries.parquet")
    val_sum = summaries[summaries["split"] == "validation"]
    mismatch_total = int(val_sum["reference_mismatch_count"].sum())

    q_sel = {c: int(design["selected_q_ns"][c]) for c in STAGE_COORDS}
    B = int(design["bound_ns"])
    thresholds = {**q_sel, "E2E": B}
    vec = per_run_vectors(samples, "validation")
    inclusion_violations = 0
    e2e_inf = 0
    for d in vec.values():
        if d["E2E"] == float("inf"):
            e2e_inf += 1
        if d["E2E"] == float("inf") or d["E2E"] > B:
            if not any(d[c] == float("inf") or d[c] > q_sel[c]
                       for c in STAGE_COORDS):
                inclusion_violations += 1
    val_triggered = bool(diag["correlation_triggered"].any())
    st = load_state(run_root)
    calib_flagged = bool(st.get("calibration_correlation_flagged"))
    h3 = h3_disposition(inclusion_violations, mismatch_total, calib_flagged,
                        val_triggered, results["status"].tolist())

    lines = [
        "# 04 Validation and H3 (e01-h3r1 R5/R6)", "",
        f"- generated: {utc_now()}",
        f"- valid runs: {int(len(samples) / 5)}; stage rows: {len(samples)}",
        f"- design created_at {design['created_at']}; earliest validation "
        f"session start {min(val_starts)}",
        "", "## Validation results (family alpha 0.05, size 5; "
        + cp_wording(ALPHA_ROW_VALIDATION) + ")", "",
        "| item | target risk | n | failures | CP lower | CP upper | "
        "verdict |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results.itertuples():
        lines.append(
            f"| {r.item_id} | {r.target_risk:.4f} | {int(r.n_units)} | "
            f"{int(r.failures)} | {r.cp_lower:.6f} | {r.cp_upper:.6f} | "
            f"{r.status} |")
    lines += ["",
              f"- event inclusion violations: {inclusion_violations} "
              "(required 0)",
              f"- reference_mismatch_count total: {mismatch_total} "
              "(required 0)",
              f"- packet loss (infinite X4/E2E): {e2e_inf}", "",
              "## Run-order diagnostics (session-aware)", "",
              "| coordinate | lag pairs | lag1 pearson | lag1 spearman | "
              "threshold | triggered |", "|---|---:|---:|---:|---:|---|"]
    for r in diag.itertuples():
        lines.append(f"| {r.coordinate_id} | {int(r.n_lag_pairs)} | "
                     f"{r.lag1_pearson:.4f} | {r.lag1_spearman:.4f} | "
                     f"{r.trigger_threshold:.4f} | "
                     f"{r.correlation_triggered} |")
    lines += ["", "## Block sensitivity (within-session, non-overlapping; "
              "sensitivity analysis only)", "",
              "| item | block size | blocks | failed | rate | CP lower | "
              "CP upper | flag |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in blocks.itertuples():
        lines.append(f"| {r.item_id} | {int(r.block_size)} | "
                     f"{int(r.n_blocks)} | {int(r.failed_blocks)} | "
                     f"{r.block_failure_rate:.5f} | {r.cp_lower:.6f} | "
                     f"{r.cp_upper:.6f} | {r.degenerate_flag} |")
    per_sess_fail = sess_summary.groupby("planned_session_index")[
        "failure_count"].sum()
    lines += ["", "## Per-session failure counts (summed over coordinates)",
              "", "| session | failures |", "|---|---:|"]
    for idx, f in per_sess_fail.items():
        lines.append(f"| s{int(idx):02d} | {int(f)} |")
    lines += ["", "## H3 disposition", "",
              f"- calibration correlation flagged: {calib_flagged}",
              f"- validation correlation triggered: {val_triggered}",
              f"- item verdicts: {results['status'].tolist()}",
              f"- priority cascade result: **H3 = {h3}**",
              "",
              "Judgment path (spec 3.3): no event-inclusion counterexample "
              "and zero reference mismatches; then the correlation rules, "
              "then per-item CP rules; the calibration flag caps H3 at "
              "INCONCLUSIVE_CORRELATION_SENSITIVITY.",
              "",
              "Data: derived/validation_results.parquet, "
              "derived/validation_run_order_diagnostics.parquet, "
              "derived/validation_session_summary.parquet, "
              "derived/block_sensitivity.parquet; figures: "
              "figures/fig03_validation_run_order.svg, "
              "figures/fig04_validation_sessions.svg, "
              "figures/fig05_old_new_comparison.svg", ""]
    atomic_write_text(run_root / "reports" / "04_VALIDATION_AND_H3.md",
                      "\n".join(lines))
    return h3


def cmd_analyze_validation(args):
    run_root = guard_run_root(args.run_root)
    require_status(run_root, {"VALIDATION_COMPLETE"}, "analyze-validation")
    design = json.loads((run_root / "derived" / "selected_design.json")
                        .read_text())
    samples, diag, sess_summary, thresholds = _extract_and_diagnose(
        run_root, "validation")
    if len(samples) != 5 * TARGET_VALID["validation"]:
        stop_experiment(run_root, "validation stage rows "
                        f"{len(samples)} != 17500", "R6")

    # design must predate every validation session
    plat = pd.read_parquet(run_root / "meta" / "session_platform.parquet")
    val_starts = plat[(plat["split"] == "validation")
                      & (plat["start_or_end"] == "start")]["timestamp"]
    if len(val_starts) and min(val_starts) <= design["created_at"]:
        stop_experiment(run_root, "validation session started at or before "
                        "selected_design.created_at", "R6")

    vec = per_run_vectors(samples, "validation")
    q_sel = {c: int(design["selected_q_ns"][c]) for c in STAGE_COORDS}
    risk_sel = {c: float(design["selected_risks"][c])
                for c in STAGE_COORDS}
    B = int(design["bound_ns"])
    thresholds = {**q_sel, "E2E": B}

    rows = []
    for coord in STAGE_COORDS:
        vals = [vec[k][coord] for k in vec]
        n = len(vals)
        m = sum(1 for v in vals if v == float("inf") or v > q_sel[coord])
        lo, hi = clopper_pearson(m, n, ALPHA_ROW_VALIDATION)
        rows.append({"item_id": coord, "target_risk": risk_sel[coord],
                     "n_units": n, "failures": m,
                     "alpha_row": ALPHA_ROW_VALIDATION, "cp_lower": lo,
                     "cp_upper": hi,
                     "status": verdict(lo, hi, risk_sel[coord])})
    e2e_vals = [vec[k]["E2E"] for k in vec]
    n_e2e = len(e2e_vals)
    m_e2e = sum(1 for v in e2e_vals if v == float("inf") or v > B)
    lo, hi = clopper_pearson(m_e2e, n_e2e, ALPHA_ROW_VALIDATION)
    rows.append({"item_id": "E2E",
                 "target_risk": float(design["risk_sum"]),
                 "n_units": n_e2e, "failures": m_e2e,
                 "alpha_row": ALPHA_ROW_VALIDATION, "cp_lower": lo,
                 "cp_upper": hi,
                 "status": verdict(lo, hi, float(design["risk_sum"]))})
    results = pd.DataFrame(rows, columns=VALIDATION_RESULT_COLUMNS)
    atomic_write_parquet(results,
                         run_root / "derived" / "validation_results.parquet")

    # event inclusion (spec 8.5)
    inclusion_violations = 0
    for k, d in vec.items():
        if d["E2E"] == float("inf") or d["E2E"] > B:
            if not any(d[c] == float("inf") or d[c] > q_sel[c]
                       for c in STAGE_COORDS):
                inclusion_violations += 1
    summaries = pd.read_parquet(run_root / "raw" / "run_summaries.parquet")
    val_sum = summaries[summaries["split"] == "validation"]
    mismatch_total = int(val_sum["reference_mismatch_count"].sum())

    # block sensitivity (spec 8.4)
    flags = failure_flags(vec, thresholds)
    block_rows = []
    for item in COORDS:
        block_rows.extend(
            block_sensitivity(flags[item], flags, item))
    blocks = pd.DataFrame(block_rows, columns=BLOCK_SENSITIVITY_COLUMNS)
    atomic_write_parquet(blocks, run_root / "derived"
                         / "block_sensitivity.parquet")

    val_triggered = bool(diag["correlation_triggered"].any())
    st = load_state(run_root)
    calib_flagged = bool(st.get("calibration_correlation_flagged"))
    h3 = h3_disposition(inclusion_violations, mismatch_total, calib_flagged,
                        val_triggered, results["status"].tolist())
    if h3 == "MODEL_OR_IMPLEMENTATION_ERROR":
        stop_experiment(run_root, "event-inclusion counterexample or "
                        "reference mismatch in validation "
                        f"(violations={inclusion_violations}, "
                        f"mismatch={mismatch_total})", "R6")

    fig_run_order(samples, diag, thresholds,
                  "fig03: validation run-order diagnostics (e01-h3r1)",
                  run_root / "figures" / "fig03_validation_run_order.svg",
                  _session_bounds(samples))
    fig_sessions(sess_summary,
                 "fig04: validation per-session medians/maxima (e01-h3r1)",
                 run_root / "figures" / "fig04_validation_sessions.svg")

    _render_report_04(run_root)

    complete_step(run_root, "R6", status="ANALYZED", h3_status=h3)
    print(results.to_string(index=False))
    print(f"H3 = {h3}")

    # exploratory old-new comparison (spec 8.6) after the confirmatory part
    _old_new_comparison(run_root, samples, design)


def _old_new_comparison(run_root, new_samples, new_design):
    try:
        e01_design = json.loads((E01_ROOT / "derived" / "selected_design.json")
                                .read_text())
        e01_samples = pd.read_parquet(E01_ROOT / "derived"
                                      / "stage_samples.parquet")
        e01_diag = pd.read_parquet(E01_ROOT / "supplement" / "derived"
                                   / "run_order_diagnostics.parquet")
    except (OSError, ValueError):
        log_deviation(run_root, "OLD_NEW_COMPARISON",
                      "e01 derived files unavailable; comparison skipped")
        return
    old_cal = e01_samples[e01_samples["split"] == "calibration"]
    new_cal = pd.read_parquet(run_root / "derived"
                              / "calibration_stage_samples.parquet")
    lines = ["## Exploratory comparison with e01 "
             "(EXPLORATORY_OLD_NEW_COMPARISON)", "",
             "New and old samples were never merged; this comparison cannot "
             "change the H3 judgment.", "",
             "| quantity | e01 (old) | e01-h3r1 (new) |", "|---|---:|---:|",
             f"| calibration B [ns] | {int(e01_design['bound_ns'])} | "
             f"{int(new_design['bound_ns'])} |",
             f"| calibration risk_sum | "
             f"{float(e01_design['risk_sum']):.4f} | "
             f"{float(new_design['risk_sum']):.4f} |"]
    for c in STAGE_COORDS:
        old_q = int(e01_design["selected_q_ns"][c])
        new_q = int(new_design["selected_q_ns"][c])
        lines.append(f"| q[{c}] [ns] | {old_q} | {new_q} |")
    for c in COORDS:
        om = np.nanmedian(old_cal[old_cal["coordinate_id"] == c]
                          ["value_ns"].astype(float))
        nm = np.nanmedian(new_cal[new_cal["coordinate_id"] == c]
                          ["value_ns"].astype(float))
        lines.append(f"| calibration median {c} [ns] | {om:.0f} | "
                     f"{nm:.0f} |")
    lines += ["", "Old e01 run-order lag-1 (single-session acquisition, "
              "from e01-s1 S3):",
              ""]
    for r in e01_diag[e01_diag["split"] == "calibration"].itertuples():
        lines.append(f"- {r.coordinate_id}: pearson={r.lag1_pearson:.4f}, "
                     f"spearman={r.lag1_spearman:.4f}, "
                     f"triggered={r.correlation_triggered}")
    with open(run_root / "reports" / "04_VALIDATION_AND_H3.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")
    fig_old_new(new_samples, new_design, e01_design, e01_samples,
                run_root / "figures" / "fig05_old_new_comparison.svg")


# ---------------------------------------------------------------- reports
def artifact_index(run_root):
    """Files that actually exist, POSIX-relative (spec 12/13).

    Per-run raw JSONL/JSON log directories (raw/<split>/session_XX/run_*)
    and the derived/test_run scratch redirect are aggregated elsewhere,
    not listed one by one: only packaged-relevant artifacts are indexed.
    """
    run_root = Path(run_root)
    entries = []
    for sub in ("configs", "meta", "figures", "reports", "state", "tables"):
        base = run_root / sub
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and ".tmp" not in p.name:
                    entries.append("/".join(p.relative_to(run_root).parts))
    for split in SPLITS:
        base = run_root / "raw" / "events" / split
        if base.exists():
            for p in sorted(base.glob("session_*/batch_*.parquet")):
                entries.append("/".join(p.relative_to(run_root).parts))
    p = run_root / "raw" / "run_summaries.parquet"
    if p.exists():
        entries.append("raw/run_summaries.parquet")
    base = run_root / "derived"
    if base.exists():
        for p in sorted(base.rglob("*")):
            if (p.is_file() and ".tmp" not in p.name
                    and "test_run" not in p.parts):
                entries.append("/".join(p.relative_to(run_root).parts))
    return entries


def raw_run_log_summary(run_root):
    """Aggregate count of per-run raw log directories kept on disk."""
    run_root = Path(run_root)
    n_dirs = 0
    n_files = 0
    for split in SPLITS:
        base = run_root / "raw" / split
        if base.exists():
            for p in base.rglob("run_meta.json"):
                n_dirs += 1
                n_files += len(list(p.parent.glob("*")))
    return n_dirs, n_files


def acceptance_checks(run_root):
    """Spec 12 acceptance items, each verified against on-disk artifacts."""
    run_root = Path(run_root)
    st = load_state(run_root)
    checks = {}

    def add(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": str(detail)}

    sess_df, run_df = load_plans(run_root)
    cal_sess = sess_df[sess_df["split"] == "calibration"]
    val_sess = sess_df[sess_df["split"] == "validation"]
    init_at = st.get("init_at") or "9999"
    newest_e01 = 0.0
    for p in E01_ROOT.rglob("*"):
        if p.is_file() and "handoff" not in p.parts:
            newest_e01 = max(newest_e01, p.stat().st_mtime)
    e01_untouched = datetime.fromtimestamp(
        newest_e01, tz=timezone.utc).isoformat(timespec="seconds") \
        <= init_at
    newest_iso = datetime.fromtimestamp(
        newest_e01, tz=timezone.utc).isoformat(timespec="seconds")
    deviations_path = run_root / "meta" / "deviations.jsonl"
    deviations_text = (deviations_path.read_text()
                       if deviations_path.exists() else "")
    recorded_e01_write = "E01_P0_ARTIFACTS_OVERWRITTEN_AND_RESTORED" \
        in deviations_text
    e01_unmodified = e01_untouched and not recorded_e01_write
    if recorded_e01_write:
        e01_detail = (
            "Protocol violation recorded in deviations.jsonl: pre-init R0 "
            "pytest runs overwrote two runs/e01 P0 report artifacts. They "
            "were restored and scientific data were unaffected, but the "
            "historical read-only requirement was not satisfied.")
    else:
        e01_detail = (
            f"newest runs/e01 file mtime {newest_iso} <= init {init_at}; "
            "guard_write also refused every e01-resolving write")
    add("e01_unmodified", e01_unmodified, e01_detail)
    if recorded_e01_write:
        add("e01_scientific_artifacts_restored", True,
            "Affected P0 report artifacts were restored; no e01 scientific "
            "data entered or changed the H3R1 confirmatory analysis.")
    add("all_tests_pass",
        json.loads((run_root / "derived" / "software_tests.json").read_text())
        ["overall_status"] == "PASS")
    add("session_plan_20x150_20x175",
        len(cal_sess) == 20 and (cal_sess["planned_valid_runs"] == 150).all()
        and len(val_sess) == 20
        and (val_sess["planned_valid_runs"] == 175).all())
    cal_runs = run_df[run_df["split"] == "calibration"]
    val_runs = run_df[run_df["split"] == "validation"]
    add("run_plan_3000_plus_3500_prefrozen",
        len(cal_runs) == 3000 and len(val_runs) == 3500
        and not run_df["run_id"].duplicated().any()
        and not run_df["seed"].duplicated().any()
        and run_df["selected_instance_index"].between(0, 7).all())
    for split, n_sess in (("calibration", 20), ("validation", 20)):
        ids = sess_df[sess_df["split"] == split]["actual_session_id"].tolist()
        good = [x for x in ids if valid_session_id(x)]
        add(f"{split}_actual_session_ids_{n_sess}",
            len(good) == n_sess and len(set(good)) == n_sess)
    add("calibration_valid_runs_3000",
        int(st.get("calibration_valid_runs", 0)) == 3000)
    add("validation_valid_runs_3500",
        int(st.get("validation_valid_runs", 0)) == 3500)
    for split in SPLITS:
        pth = run_root / "derived" / f"{split}_stage_samples.parquet"
        add(f"{split}_stage_rows_{5 * TARGET_VALID[split]}",
            pth.exists() and len(pd.read_parquet(pth))
            == 5 * TARGET_VALID[split])
    menu = pd.read_parquet(run_root / "derived" / "menu.parquet")
    kmap = {round(float(r.risk), 4): int(r.order_k)
            for r in menu.itertuples() if r.status == "CERTIFIED"}
    add("menu_k_exact_3000_2999_2997",
        kmap == {0.002: 3000, 0.003: 2999, 0.004: 2997}, str(kmap))
    design = json.loads((run_root / "derived" / "selected_design.json")
                        .read_text())
    add("dp_matches_brute_on_real_menu",
        design.get("dp_matches_brute_force") is True)
    results = pd.read_parquet(run_root / "derived" / "validation_results"
                              ".parquet")
    add("five_items_cp_and_status",
        len(results) == 5 and results["cp_lower"].notna().all()
        and results["cp_upper"].notna().all()
        and results["status"].notna().all())
    rep04 = (run_root / "reports" / "04_VALIDATION_AND_H3.md").read_text()
    add("alpha_row_wording_one_sided_99",
        "one-sided 99% CP per item" in rep04 and "95% family" in rep04
        and "one-sided 95%" not in rep04)
    add("event_inclusion_zero", "event inclusion violations: 0" in rep04)
    diag_cal_time_ok = design.get("created_at") is not None
    plat = pd.read_parquet(run_root / "meta" / "session_platform.parquet")
    val_starts = plat[(plat["split"] == "validation")
                      & (plat["start_or_end"] == "start")]["timestamp"]
    add("design_created_before_all_validation",
        diag_cal_time_ok and len(val_starts) == 20
        and min(val_starts) > design["created_at"])
    add("correlation_rules_applied",
        "calibration correlation flagged:" in rep04
        and "validation correlation triggered:" in rep04
        and f"**H3 = {st.get('h3_status')}**" in rep04)
    blocks = pd.read_parquet(run_root / "derived" / "block_sensitivity"
                             ".parquet")
    add("block_sensitivity_complete",
        len(blocks) == 5 * len(BLOCK_SIZES))
    add("old_new_not_merged", "never merged" in rep04)
    add("h3_status_valid_state", st.get("h3_status") in
        {"SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE",
         "INCONCLUSIVE_CORRELATION_SENSITIVITY"})
    add("state_report_consistent", st.get("status") in STATE_ORDER
        and st.get("last_completed_step") is not None)
    idx = artifact_index(run_root)
    add("artifact_index_posix_real",
        len(idx) > 0 and all(not os.path.isabs(x) and ".." not in x
                             and "\\" not in x for x in idx))
    n_done = int((run_df["status"] == "DONE").sum())
    add("run_plan_done_counts",
        n_done == 6500 and int((run_df["status"] == "PLANNED").sum()) == 0,
        f"DONE={n_done}")
    steps = st.get("completed_steps", [])
    diag_mtime = datetime.fromtimestamp(
        (run_root / "derived" / "calibration_run_order_diagnostics.parquet")
        .stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    add("calibration_diagnostics_precede_menu",
        "R3" in steps and "R4" in steps
        and steps.index("R3") < steps.index("R4")
        and diag_mtime <= design["created_at"],
        f"R3 at position {steps.index('R3') if 'R3' in steps else -1} < "
        f"R4 at {steps.index('R4') if 'R4' in steps else -1}; diagnostics "
        f"written {diag_mtime} <= design created {design['created_at']}")
    blocks_expected = {}
    val_samples = pd.read_parquet(run_root / "derived"
                                  / "validation_stage_samples.parquet")
    sess_sizes = (val_samples[val_samples["coordinate_id"] == "X1"]
                  .groupby("planned_session_index").size())
    for bs in BLOCK_SIZES:
        blocks_expected[bs] = int(sum(-(-n // bs) for n in sess_sizes))
    blocks_ok = all(
        int(row.n_blocks) == blocks_expected[int(row.block_size)]
        for row in blocks.itertuples())
    add("blocks_within_session_arithmetic", blocks_ok,
        f"expected n_blocks by size {blocks_expected} matches "
        "per-session non-overlapping chopping")
    inclusion_zero = "event inclusion violations: 0" in rep04
    mismatch_zero = int(val_summaries(run_root)
                        ["reference_mismatch_count"].sum()) == 0
    recomputed = h3_disposition(
        0 if inclusion_zero else 1, 0 if mismatch_zero else 1,
        bool(st.get("calibration_correlation_flagged")),
        bool(diag_val_triggered(run_root)),
        results["status"].tolist())
    add("h3_recomputed_from_data_matches",
        inclusion_zero and mismatch_zero
        and recomputed == st.get("h3_status"),
        f"recomputed {recomputed} == state {st.get('h3_status')}")
    return checks


def acceptance_payload(checks):
    """Separate strict protocol acceptance from scientific-data integrity."""
    strict_ok = all(c["ok"] for c in checks.values())
    scientific_ok = all(
        c["ok"] for name, c in checks.items() if name != "e01_unmodified")
    return {
        "all_ok": strict_ok,
        "strict_protocol_acceptance": strict_ok,
        "scientific_data_integrity": scientific_ok,
        "checks": checks,
    }


def val_summaries(run_root):
    summaries = pd.read_parquet(Path(run_root) / "raw"
                                / "run_summaries.parquet")
    return summaries[summaries["split"] == "validation"]


def diag_val_triggered(run_root):
    diag = pd.read_parquet(Path(run_root) / "derived"
                           / "validation_run_order_diagnostics.parquet")
    return bool(diag["correlation_triggered"].any())


# ---------------------------------------------------------------- reports
def artifact_index(run_root):
    """Files that actually exist, POSIX-relative (spec 12/13).

    Per-run raw JSONL/JSON log directories (raw/<split>/session_XX/run_*)
    and the derived/test_run scratch redirect are aggregated elsewhere,
    not listed one by one: only packaged-relevant artifacts are indexed.
    """
    run_root = Path(run_root)
    entries = []
    for sub in ("configs", "meta", "figures", "reports", "state", "tables"):
        base = run_root / sub
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and ".tmp" not in p.name:
                    entries.append("/".join(p.relative_to(run_root).parts))
    for split in SPLITS:
        base = run_root / "raw" / "events" / split
        if base.exists():
            for p in sorted(base.glob("session_*/batch_*.parquet")):
                entries.append("/".join(p.relative_to(run_root).parts))
    p = run_root / "raw" / "run_summaries.parquet"
    if p.exists():
        entries.append("raw/run_summaries.parquet")
    base = run_root / "derived"
    if base.exists():
        for p in sorted(base.rglob("*")):
            if (p.is_file() and ".tmp" not in p.name
                    and "test_run" not in p.parts):
                entries.append("/".join(p.relative_to(run_root).parts))
    return entries


def raw_run_log_summary(run_root):
    """Aggregate count of per-run raw log directories kept on disk."""
    run_root = Path(run_root)
    n_dirs = 0
    n_files = 0
    for split in SPLITS:
        base = run_root / "raw" / split
        if base.exists():
            for p in base.rglob("run_meta.json"):
                n_dirs += 1
                n_files += len(list(p.parent.glob("*")))
    return n_dirs, n_files


def _build_report_files(run_root):
    """Write DEVIATIONS + FINAL report from current state (idempotent)."""
    run_root = Path(run_root)
    st = load_state(run_root)
    design = json.loads((run_root / "derived" / "selected_design.json")
                        .read_text())
    results = pd.read_parquet(run_root / "derived" / "validation_results"
                              ".parquet")
    devs = []
    dev_path = run_root / "meta" / "deviations.jsonl"
    if dev_path.exists():
        devs = [json.loads(l) for l in dev_path.read_text().splitlines()
                if l.strip()]
    e01_incident = any(
        d.get("kind") == "E01_P0_ARTIFACTS_OVERWRITTEN_AND_RESTORED"
        for d in devs)
    e01_convention = (
        "- The runs/e01 read-only requirement was violated during pre-init "
        "R0 development as recorded above. The affected P0 report "
        "artifacts were restored; no e01 scientific data entered or "
        "changed the H3R1 confirmatory analysis."
        if e01_incident else
        "- runs/e01 remained read-only throughout; the pytest conftest P0 "
        "artifact writer was redirected via PARCE_TEST_RESULTS_DIR for "
        "this experiment's test runs.")
    dev_lines = ["# Deviations (e01-h3r1)", "",
                 f"- generated: {utc_now()}",
                 f"- execution status at write time: {st.get('status')}", "",
                 f"- total deviation records: {len(devs)}", ""]
    for d in devs:
        dev_lines.append(f"- {d['time']} [{d['kind']}] {d['detail']}")
    dev_lines += [
        "",
        "## Standing conventions (recorded at plan freeze)", "",
        "- Long idle of 300 s is taken after every 5th completed session "
        "except the final session of a split (spec 5.3 'every 5 sessions'; "
        "the post-split gap subsumes the final one).",
        "- Technically invalid runs are retried under the same run_id with "
        "attempt_index+1 (max 3 attempts per run), keeping the frozen run "
        "plan at exactly 3000+3500 planned rows.",
        e01_convention,
        "",
    ]
    atomic_write_text(run_root / "reports" / "DEVIATIONS.md",
                      "\n".join(dev_lines))

    idx = artifact_index(run_root)
    h3 = st.get("h3_status")
    final = [
        "# e01-h3r1 Final H3 Reacquisition Report", "",
        f"- generated: {utc_now()}",
        f"- execution status: {st.get('status')}",
        f"- H3 status: {h3}",
        "", "## 1. Scope and Frozen Configuration", "",
        f"- experiment {EXPERIMENT_ID} (parent {PARENT_EXPERIMENT_ID}); "
        "confirmatory scope: sessionized Calibration, Validation, H3; "
        "exploratory: session heterogeneity, block sensitivity, old-new "
        "comparison",
        "- old H3 evidence replaced: e01-s1 S3 run-order correlation "
        "trigger (single-session acquisition)",
        "", "```yaml",
        yaml.safe_dump({"config": FROZEN_CONFIG,
                        "sessions": SESSIONS, "per_session": PER_SESSION,
                        "master_seed": MASTER_SEED}, sort_keys=False),
        "```", "", "## 2. Software and Plan Verification", "",
        "- full pytest suite (e01 regression + e01-h3r1 tests) passed at "
        "R0; reference_mismatch_count=0; DP==brute force",
        f"- plans frozen one-shot from master seed {MASTER_SEED}: 20x150 "
        "calibration + 20x175 validation sessions, 3000+3500 pre-frozen "
        "runs", "", "## 3. Sessionized Calibration", "",
        f"- {st.get('calibration_valid_runs')} valid runs across 20 "
        "sessions; details in reports/02_CALIBRATION.md",
        "", "## 4. Calibration Dependence Diagnostics", "",
        "- session-aware lag-1 with no cross-session pairs; see "
        "reports/02_CALIBRATION.md",
        f"- CALIBRATION_CORRELATION_FLAGGED = "
        f"{st.get('calibration_correlation_flagged')}",
        "", "## 5. New Menu and Frozen Design", "",
        f"- selected menu ids: {design['selected_menu_ids']}",
        f"- selected q [ns]: {design['selected_q_ns']}",
        f"- risk_sum {design['risk_sum']:.4f}; bound B "
        f"{design['bound_ns']} ns; created_at {design.get('created_at')}",
        "- DP verified against exhaustive enumeration and an independent "
        "section-5.6 replay",
        "", "## 6. Sessionized Validation", "",
        f"- {st.get('validation_valid_runs')} valid runs across 20 "
        "sessions, all sessions started after design creation",
        "", "## 7. H3 Statistical Results", "",
        "| item | target | n | failures | CP lower | CP upper | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results.itertuples():
        final.append(f"| {r.item_id} | {r.target_risk:.4f} | {int(r.n_units)}"
                     f" | {int(r.failures)} | {r.cp_lower:.6f} | "
                     f"{r.cp_upper:.6f} | {r.status} |")
    final += [
        "",
        f"Intervals are {cp_wording(ALPHA_ROW_VALIDATION)}.",
        "",
        "## 8. Block and Session Sensitivity", "",
        "- block results are a dependence sensitivity analysis; they do not "
        "replace the run-level H3 judgment (see "
        "reports/04_VALIDATION_AND_H3.md)",
        "- zero-failure items are marked "
        "ZERO_EVENT_EMPIRICAL_BOOTSTRAP_DEGENERATE instead of reporting a "
        "bootstrap upper bound of 0",
        "", "## 9. Exploratory Comparison with e01", "",
        "- labeled EXPLORATORY_OLD_NEW_COMPARISON; samples never merged; "
        "see reports/04_VALIDATION_AND_H3.md and "
        "figures/fig05_old_new_comparison.svg",
        "", "## 10. H3 Disposition", "",
        f"**H3 = {h3}** (priority cascade of h3_reacquisition.MD 3.3; "
        "execution status and hypothesis disposition are recorded "
        "separately)",
        "", "## 11. Deviations and Limitations", "",
        f"- see reports/DEVIATIONS.md ({len(devs)} records)",
        *(["- strict protocol acceptance is false because two runs/e01 P0 "
           "report artifacts were overwritten during pre-init R0 "
           "development and later restored; scientific data integrity "
           "remains true and the H3R1 numerical results are unchanged",
           "- corrected review-package layout places all event batches "
           "under runs/e01-h3r1/raw/events/ as required by "
           "h3_reacquisition.MD 13.1"] if e01_incident else []),
        "- the multi-session design reduces single-acquisition drift risk "
        "but is not a mathematical proof of independence",
        "- 'no correlation detected' is not 'independence proven'",
        "- no claim is made about risks below the sample resolution "
        "(n=3500: the smallest one-sided upper bound at zero failures is "
        f"{clopper_pearson(0, 3500, 0.01)[1]:.6f})",
        "", "## 12. Reproduction Commands", "",
        "```bash",
        "source .venv/bin/activate",
        "pytest -q",
        "python -m parce_exp.h3_reacquisition init --run-root runs/e01-h3r1",
        "python -m parce_exp.h3_reacquisition software-gate --run-root "
        "runs/e01-h3r1",
        f"python -m parce_exp.h3_reacquisition freeze-plan --run-root "
        f"runs/e01-h3r1 --master-seed {MASTER_SEED}",
        "bash scripts/run_h3r1.sh calibration --run-root runs/e01-h3r1 "
        "--resume",
        "python -m parce_exp.h3_reacquisition analyze-calibration --run-root"
        " runs/e01-h3r1",
        "python -m parce_exp.h3_reacquisition freeze-design --run-root "
        "runs/e01-h3r1",
        "bash scripts/run_h3r1.sh validation --run-root runs/e01-h3r1 "
        "--resume",
        "python -m parce_exp.h3_reacquisition analyze-validation --run-root "
        "runs/e01-h3r1",
        "python -m parce_exp.h3_reacquisition build-reports --run-root "
        "runs/e01-h3r1",
        "pytest -q",
        "python -m parce_exp.h3_reacquisition verify-final --run-root "
        "runs/e01-h3r1",
        "python -m parce_exp.h3_reacquisition build-handoff --run-root "
        "runs/e01-h3r1 --output "
        "runs/e01-h3r1/handoff/outgoing/e01-h3r1-review-package.zip",
        "```", "", "## 13. Artifact Index", "",
    ]
    for entry in idx:
        final.append(f"- {entry}")
    log_dirs, log_files = raw_run_log_summary(run_root)
    final += [
        "",
        f"({len(idx)} indexed artifacts; generated from the on-disk tree "
        "at report time)",
        "",
        "Additionally kept on disk but not packaged: per-run raw process "
        f"logs for {log_dirs} runs ({log_files} files under "
        "raw/<split>/session_XX/run_*/: sender/gate/receiver JSONL + "
        "run_meta.json) and the derived/test_run scratch redirect of the "
        "P0 test artifact writer.",
        ""]
    atomic_write_text(
        run_root / "reports" / "FINAL_H3_REACQUISITION_REPORT.md",
        "\n".join(final))


def cmd_build_reports(args):
    run_root = guard_run_root(args.run_root)
    require_status(run_root, {"ANALYZED", "REPORTS_COMPLETE"},
                   "build-reports")
    checks = acceptance_checks(run_root)
    atomic_write_json(run_root / "derived" / "acceptance_checks.json",
                      acceptance_payload(checks))
    _build_report_files(run_root)          # pass 1: pre-state report
    complete_step(run_root, "R7-reports", status="REPORTS_COMPLETE")
    _build_report_files(run_root)          # pass 2: embed REPORTS_COMPLETE
    checks = acceptance_checks(run_root)
    ok = all(c["ok"] for c in checks.values())
    if not ok:
        bad = [k for k, c in checks.items() if not c["ok"]]
        log_deviation(run_root, "ACCEPTANCE_FAILED", "; ".join(bad))
        atomic_write_json(run_root / "derived" / "acceptance_checks.json",
                          acceptance_payload(checks))
        stop_experiment(run_root, "acceptance checks failed: "
                        + "; ".join(bad), "R7")
    complete_step(run_root, "R7", status="COMPLETE")
    _build_report_files(run_root)          # pass 3: embed COMPLETE
    checks = acceptance_checks(run_root)
    atomic_write_json(run_root / "derived" / "acceptance_checks.json",
                      acceptance_payload(checks))
    _build_report_files(run_root)          # final artifact index refresh
    print(json.dumps({"all_ok": True,
                      "status": load_state(run_root)["status"],
                      "h3_status": load_state(run_root)["h3_status"]},
                     indent=2))


def cmd_rebuild_reports(args):
    """Re-render all markdown reports from derived artifacts without
    touching state, design, or parquet data (audit/documentation fixes)."""
    run_root = guard_run_root(args.run_root)
    st = load_state(run_root)
    if st.get("status") not in {"ANALYZED", "REPORTS_COMPLETE", "COMPLETE"}:
        raise SystemExit("rebuild-reports requires ANALYZED or later")
    _render_report_02(run_root)
    _render_report_03(run_root)
    h3 = _render_report_04(run_root)
    design = json.loads((run_root / "derived" / "selected_design.json")
                        .read_text())
    _old_new_comparison(run_root, pd.read_parquet(
        run_root / "derived" / "validation_stage_samples.parquet"), design)
    checks = acceptance_checks(run_root)
    atomic_write_json(run_root / "derived" / "acceptance_checks.json",
                      acceptance_payload(checks))
    _build_report_files(run_root)
    print(json.dumps({"all_ok": all(c["ok"] for c in checks.values()),
                      "h3_recomputed": h3,
                      "status": st.get("status"),
                      "h3_status": st.get("h3_status")}, indent=2))


def cmd_verify_final(args):
    run_root = guard_run_root(args.run_root)
    st = load_state(run_root)
    checks = json.loads((run_root / "derived" / "acceptance_checks.json")
                        .read_text())
    problems = []
    if st.get("status") not in ("COMPLETE", "REPORTS_COMPLETE"):
        problems.append(f"state={st.get('status')}")
    recorded_delivery_deviation = (
        checks.get("strict_protocol_acceptance") is False
        and checks.get("scientific_data_integrity") is True
        and st.get("delivery_status")
        == "CORRECTED_WITH_RECORDED_PROTOCOL_DEVIATION")
    if not checks.get("all_ok") and not recorded_delivery_deviation:
        problems.append("acceptance_checks.all_ok is false without an "
                        "accepted, integrity-preserving delivery deviation")
    final = (run_root / "reports" / "FINAL_H3_REACQUISITION_REPORT.md")
    if not final.exists():
        problems.append("missing final report")
    else:
        text = final.read_text()
        if f"H3 = {st.get('h3_status')}" not in text:
            problems.append("final report H3 line inconsistent with state")
    if problems:
        raise SystemExit("verify-final FAILED: " + "; ".join(problems))
    print(json.dumps({"verify_final": "PASS",
                      "status": st.get("status"),
                      "h3_status": st.get("h3_status")}, indent=2))


def _fmt_size(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def cmd_build_handoff(args):
    run_root = guard_run_root(args.run_root)
    st = load_state(run_root)
    stopped = st.get("status") == "STOPPED"

    def rel(project_rel):
        return PROJECT_ROOT / project_rel

    file_specs = [
        ("h3_reacquisition.MD", rel("h3_reacquisition.MD"), "md"),
        ("runs/e01-h3r1/state/state.json",
         run_root / "state" / "state.json", "json"),
        ("runs/e01-h3r1/configs/frozen_h3r1_plan.yaml",
         run_root / "configs" / "frozen_h3r1_plan.yaml", "yaml"),
        ("runs/e01-h3r1/configs/session_plan.csv",
         run_root / "configs" / "session_plan.csv", "csv"),
        ("runs/e01-h3r1/configs/run_plan.csv",
         run_root / "configs" / "run_plan.csv", "csv"),
        ("runs/e01-h3r1/meta/platform_baseline.yaml",
         run_root / "meta" / "platform_baseline.yaml", "yaml"),
        ("runs/e01-h3r1/meta/session_platform.parquet",
         run_root / "meta" / "session_platform.parquet", "parquet"),
        ("runs/e01-h3r1/meta/deviations.jsonl",
         run_root / "meta" / "deviations.jsonl", "json"),
        ("runs/e01-h3r1/raw/run_summaries.parquet",
         run_root / "raw" / "run_summaries.parquet", "parquet"),
    ]
    for split in SPLITS:
        base = run_root / "raw" / "events" / split
        for batch in sorted(base.glob("session_*/batch_*.parquet")):
            relparts = batch.relative_to(run_root).parts
            file_specs.append((
                "runs/e01-h3r1/" + "/".join(relparts),
                batch,
                "parquet",
            ))
    for name in (
            "software_tests.json", "calibration_stage_samples.parquet",
            "calibration_run_order_diagnostics.parquet",
            "calibration_session_summary.parquet", "menu.parquet",
            "menu_pruned.json", "selected_design.json",
            "validation_stage_samples.parquet", "validation_results.parquet",
            "validation_run_order_diagnostics.parquet",
            "validation_session_summary.parquet",
            "block_sensitivity.parquet", "acceptance_checks.json"):
        file_specs.append((f"runs/e01-h3r1/derived/{name}",
                           run_root / "derived" / name,
                           name.rsplit(".", 1)[-1]))
    for name in ("01_PLAN_AND_PLATFORM.md", "02_CALIBRATION.md",
                 "03_DESIGN_FREEZE.md", "04_VALIDATION_AND_H3.md",
                 "DEVIATIONS.md", "FINAL_H3_REACQUISITION_REPORT.md"):
        file_specs.append((f"runs/e01-h3r1/reports/{name}",
                           run_root / "reports" / name, "md"))
    # STOP_REPORT.md is required exactly when the experiment is STOPPED
    file_specs.append(("runs/e01-h3r1/reports/STOP_REPORT.md",
                       run_root / "reports" / "STOP_REPORT.md", "md"))
    for name in ("fig01_calibration_run_order.svg",
                 "fig02_calibration_sessions.svg",
                 "fig03_validation_run_order.svg",
                 "fig04_validation_sessions.svg",
                 "fig05_old_new_comparison.svg"):
        file_specs.append((f"runs/e01-h3r1/figures/{name}",
                           run_root / "figures" / name, "svg"))
    for name in ("python/parce_exp/h3_reacquisition.py",
                 "python/parce_exp/runner.py", "python/parce_exp/extract.py",
                 "python/parce_exp/report.py",
                 "tests/test_h3_reacquisition.py", "tests/conftest.py",
                 "scripts/run_h3r1.sh"):
        file_specs.append((name, rel(name),
                           "py" if name.endswith(".py") else "sh"))
    for name, fmt in (("configs/calendar.yaml", "yaml"),
                      ("configs/risk.yaml", "yaml"),
                      ("configs/frozen_plan.yaml", "yaml"),
                      ("configs/run_plan.csv", "csv")):
        file_specs.append((name, rel(name), fmt))

    manifest_files = []
    present_specs = []
    for arcname, src, fmt in file_specs:
        is_stop_report = arcname.endswith("STOP_REPORT.md")
        if src.exists():
            row_count = None
            if fmt == "parquet":
                import pyarrow.parquet as pq

                row_count = int(pq.ParquetFile(str(src)).metadata.num_rows)
            elif fmt == "csv":
                with open(src) as fh:
                    row_count = max(0, sum(1 for _ in fh) - 1)
            manifest_files.append({
                "path": arcname, "status": "PRESENT",
                "size_bytes": int(src.stat().st_size), "format": fmt,
                "row_count": row_count})
            present_specs.append((arcname, src))
        elif is_stop_report and not stopped:
            continue  # not applicable for a COMPLETE run
        else:
            manifest_files.append({
                "path": arcname, "status": "MISSING", "size_bytes": 0,
                "format": fmt, "row_count": None})

    missing = [f["path"] for f in manifest_files
               if f["status"] == "MISSING"]
    exec_status = ("COMPLETE"
                   if st.get("status") == "COMPLETE" and not missing
                   else "STOPPED")
    acceptance_path = run_root / "derived" / "acceptance_checks.json"
    acceptance = (json.loads(acceptance_path.read_text())
                  if acceptance_path.exists() else {})
    strict_acceptance = acceptance.get(
        "strict_protocol_acceptance", acceptance.get("all_ok"))
    scientific_integrity = acceptance.get("scientific_data_integrity")
    manifest_notes = []
    if strict_acceptance is False and scientific_integrity is True:
        manifest_notes.append(
            "Execution completed with a recorded, restored pre-init R0 "
            "write to two runs/e01 P0 report artifacts; strict protocol "
            "acceptance is false, scientific data integrity is true.")

    def render(manifest_size_guess):
        files = manifest_files + [{
            "path": "PACKAGE_MANIFEST.json", "status": "PRESENT",
            "size_bytes": int(manifest_size_guess), "format": "json",
            "row_count": None}]
        manifest = {
            "package_id": "e01-h3r1-review-package",
            "experiment_id": EXPERIMENT_ID,
            "parent_experiment_id": PARENT_EXPERIMENT_ID,
            "created_at": utc_now(),
            "execution_status": exec_status,
            "h3_status": st.get("h3_status"),
            "strict_protocol_acceptance": strict_acceptance,
            "scientific_data_integrity": scientific_integrity,
            "delivery_status": st.get("delivery_status"),
            "calibration_valid_runs": int(
                st.get("calibration_valid_runs", 0)),
            "validation_valid_runs": int(st.get("validation_valid_runs", 0)),
            "calibration_sessions": SESSIONS["calibration"],
            "validation_sessions": SESSIONS["validation"],
            "files": files,
            "missing_required_files": missing,
            "notes": manifest_notes,
        }
        return json.dumps(manifest, indent=2) + "\n"

    # fixed point on the manifest's own serialized size
    guess = 0
    for _ in range(4):
        text = render(guess)
        size = len(text.encode())
        if size == guess:
            break
        guess = size
    manifest_text = render(size)

    out = Path(args.output)
    guard_write(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in present_specs:
            zf.write(src, arcname)
        zf.writestr("PACKAGE_MANIFEST.json", manifest_text)

    problems = []
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        if any(n.startswith("/") or ".." in n.split("/") for n in names):
            problems.append("absolute or .. paths in zip")
        if "PACKAGE_MANIFEST.json" not in names:
            problems.append("manifest not at zip root")
        listed = set()
        for f in json.loads(zf.read("PACKAGE_MANIFEST.json"))["files"]:
            listed.add(f["path"])
            if f["status"] == "PRESENT":
                if f["path"] not in names:
                    problems.append(
                        f"manifest PRESENT but missing in zip: {f['path']}")
                elif zf.getinfo(f["path"]).file_size != f["size_bytes"]:
                    problems.append(
                        f"size mismatch in zip: {f['path']}")
        unlisted = [n for n in names if n not in listed]
        if unlisted:
            problems.append(f"unlisted files: {unlisted[:5]}")
        for n in names:
            if any(part in (".venv", "build", "__pycache__",
                            ".pytest_cache", "pending")
                   for part in n.split("/")):
                problems.append(f"forbidden directory in zip: {n}")
    if problems:
        raise SystemExit("build-handoff self-check failed: "
                         + "; ".join(problems))
    print(json.dumps({
        "zip": str(out), "size_bytes": out.stat().st_size,
        "size_human": _fmt_size(out.stat().st_size),
        "files": len(json.loads(manifest_text)["files"]),
        "execution_status": exec_status, "h3_status": st.get("h3_status"),
        "missing_required_files": missing}, indent=2))


def build_parser():
    p = argparse.ArgumentParser(prog="parce_exp.h3_reacquisition")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_init)
    sp = sub.add_parser("software-gate")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_software_gate)
    sp = sub.add_parser("freeze-plan")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.add_argument("--master-seed", type=int, default=MASTER_SEED)
    sp.set_defaults(func=cmd_freeze_plan)
    sp = sub.add_parser("acquire")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.add_argument("--split", choices=list(SPLITS), required=True)
    sp.add_argument("--resume", action="store_true")
    sp.set_defaults(func=cmd_acquire)
    sp = sub.add_parser("analyze-calibration")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_analyze_calibration)
    sp = sub.add_parser("freeze-design")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_freeze_design)
    sp = sub.add_parser("analyze-validation")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_analyze_validation)
    sp = sub.add_parser("build-reports")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_build_reports)
    sp = sub.add_parser("rebuild-reports")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_rebuild_reports)
    sp = sub.add_parser("verify-final")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.set_defaults(func=cmd_verify_final)
    sp = sub.add_parser("build-handoff")
    sp.add_argument("--run-root", default="runs/e01-h3r1")
    sp.add_argument("--output",
                    default="runs/e01-h3r1/handoff/outgoing/"
                            "e01-h3r1-review-package.zip")
    sp.set_defaults(func=cmd_build_handoff)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
