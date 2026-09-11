import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE_BIN = ROOT / "build" / "software_gate"
RESULTS = {"groups": []}


def record_group(name, passed, **metrics):
    RESULTS["groups"].append({"name": name, "passed": bool(passed), **metrics})


def run_calcheck(cases):
    """Run the C++ reference operator on a batch of cases.

    cases: iterable of (eligible, demand, period, windows) with windows as
    (open, close) pairs. Returns a list of (start, cycle, window, reason).
    """
    if not GATE_BIN.exists():
        pytest.fail(
            f"missing {GATE_BIN}; build first: "
            "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build"
        )
    lines = []
    for eligible, demand, period, windows in cases:
        parts = [eligible, demand, period, len(windows)]
        for open_ns, close_ns in windows:
            parts.extend([open_ns, close_ns])
        lines.append(" ".join(str(x) for x in parts))
    proc = subprocess.run(
        [str(GATE_BIN), "--calcheck"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [tuple(int(t) for t in line.split()) for line in proc.stdout.splitlines() if line.strip()]
    assert len(rows) == len(lines), "calcheck line count mismatch"
    return rows


@pytest.fixture(scope="session")
def calcheck():
    return run_calcheck


@pytest.fixture(scope="module")
def rec():
    return record_group
