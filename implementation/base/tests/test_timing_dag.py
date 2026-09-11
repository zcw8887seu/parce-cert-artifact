"""1,000 random small timing-DAG tests (spec 7-P0).

Properties: per-coordinate monotonicity of sink completion, analytical
propagation equals per-event replay, shared coordinate counts one risk event.
"""

import numpy as np

from parce_exp.allocator import DagNode, TimingDAG, distinct_exceedances

N_DAGS = 1000
SEED = np.random.SeedSequence(entropy=20260816, spawn_key=(1,))
P = 1_000_000
W1 = [(400_000, 700_000)]
DEMAND = 50_000
N_COORDS = 4


def _gen_dag(rng):
    n_nodes = int(rng.integers(3, 8))
    nodes = [DagNode("delay", coordinate=0, parents=())]
    has_join = False
    has_gate = False
    for i in range(1, n_nodes):
        n_parents = 1
        if i >= 2 and rng.random() < 0.4:
            n_parents = 2
            has_join = True
        parents = tuple(sorted(rng.choice(i, size=n_parents, replace=False).tolist()))
        want_gate = (not has_gate) or rng.random() < 0.25
        if want_gate:
            nodes.append(DagNode("gate", parents=parents))
            has_gate = True
        else:
            nodes.append(DagNode("delay", coordinate=int(rng.integers(0, N_COORDS)), parents=parents))
    if not has_join:
        parents = (0, 1)
        nodes[-1] = DagNode("gate" if not has_gate else "delay",
                            coordinate=nodes[-1].coordinate, parents=parents)
        has_gate = True
    if not nodes[-1].parents:
        nodes[-1] = DagNode(nodes[-1].kind, coordinate=nodes[-1].coordinate,
                            parents=(len(nodes) - 2,))
    # force a shared coordinate: make the second delay node reuse coordinate 0
    delay_idx = [i for i, n in enumerate(nodes) if n.kind == "delay" and i != len(nodes) - 1]
    if len(delay_idx) >= 2 and nodes[delay_idx[1]].coordinate != 0:
        nodes[delay_idx[1]] = DagNode("delay", coordinate=0, parents=nodes[delay_idx[1]].parents)
    return TimingDAG(nodes)


def test_timing_dag(rec):
    rng = np.random.default_rng(SEED)
    replay_mismatches = 0
    monotonic_violations = 0
    event_inclusion_violations = 0
    joins = 0
    gates = 0
    shared_coord_dags = 0

    for _ in range(N_DAGS):
        dag = _gen_dag(rng)
        joins += sum(1 for n in dag.nodes if len(n.parents) >= 2)
        gates += sum(1 for n in dag.nodes if n.kind == "gate")
        coord_use = {}
        for node in dag.nodes:
            if node.kind == "delay":
                coord_use[node.coordinate] = coord_use.get(node.coordinate, 0) + 1
        if any(c >= 2 for c in coord_use.values()):
            shared_coord_dags += 1

        values = [int(rng.integers(0, 300_000)) for _ in range(N_COORDS)]
        analytical = dag.analytical_completion(values, DEMAND, P, W1)
        replay = dag.replay_completion(values, DEMAND, P, W1)
        if analytical != replay:
            replay_mismatches += 1

        for coord in range(N_COORDS):
            delta = int(rng.choice([1, 1000, 70_000]))
            bumped = list(values)
            bumped[coord] += delta
            a2 = dag.analytical_completion(bumped, DEMAND, P, W1)
            r2 = dag.replay_completion(bumped, DEMAND, P, W1)
            if a2 < analytical or r2 < replay or a2 != r2:
                monotonic_violations += 1

        envelopes = [v + 10_000 for v in values]
        k_violated = int(rng.integers(1, N_COORDS + 1))
        violated = set(rng.choice(N_COORDS, size=k_violated, replace=False).tolist())
        sample = [
            envelopes[j] + int(rng.integers(1, 1000)) if j in violated
            else envelopes[j] - int(rng.integers(1, 1000))
            for j in range(N_COORDS)
        ]
        if distinct_exceedances(sample, envelopes) != sorted(violated):
            event_inclusion_violations += 1

    rec(
        "timing_dag",
        replay_mismatches == 0 and monotonic_violations == 0 and event_inclusion_violations == 0,
        cases=N_DAGS,
        analytical_replay_mismatch_count=replay_mismatches,
        monotonicity_violation_count=monotonic_violations,
        event_inclusion_violation_count=event_inclusion_violations,
        join_nodes=joins,
        gate_nodes=gates,
        shared_coordinate_dags=shared_coord_dags,
    )
    assert replay_mismatches == 0
    assert monotonic_violations == 0
    assert event_inclusion_violations == 0
    assert joins > 0 and gates > 0 and shared_coord_dags > 0
