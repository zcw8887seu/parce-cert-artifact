"""Read-only scientific checks; does not collect data or run new experiments."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"implementation/base/python"))
from parce_exp.calendar_ref import next_feasible_start
result={}
for campaign,count in (("e01",278600),("h3r1",260000)):
    frame=pd.read_parquet(ROOT/f"data/{campaign}/raw/events.parquet")
    assert len(frame)==count
    mismatches=0
    for r in frame.itertuples(index=False):
        start,cycle,window,reason=next_feasible_start(
            int(r.t_gate_ingress_ns),int(r.service_demand_ns),1_000_000,[(400_000,700_000)])
        mismatches += int(start!=r.t_gate_target1_ns or cycle!=r.target_cycle_id or window!=r.target_window_id)
    result[campaign]={"events":len(frame),"first_target_reference_mismatches":mismatches}
    assert mismatches==0
    if campaign=="h3r1":
        selected=frame.loc[frame.selected_single & ~frame.is_warmup & frame.valid_event].copy()
        selected["X1"]=selected.t_compute_end_ns-selected.t_cause_ns
        selected["X2"]=selected.t_gate_ingress_ns-selected.t_compute_end_ns
        selected["X3"]=selected.t_gate_service_start_ns-selected.t_gate_target1_ns
        selected["X4"]=(selected.t_rx_ns-selected.t_gate_service_start_ns-selected.service_demand_ns).astype(float)
        selected["E2E"]=(selected.t_rx_ns-selected.t_cause_ns).astype(float)
        selected.loc[selected.packet_lost,["X4","E2E"]]=np.inf
        assert selected.groupby("split").size().to_dict()=={"calibration":3000,"validation":3500}
        public=pd.read_csv(ROOT/"core_artifact/data/h3r1_reduced_runs.csv")
        for split in ("calibration","validation"):
            for coordinate in ("X1","X2","X3","X4","E2E"):
                a=selected.loc[selected.split.eq(split)].sort_values(["planned_session_index","session_order"])[coordinate].to_numpy(dtype=float)
                b=public.loc[public.split.eq(split)&public.coordinate_id.eq(coordinate)].sort_values(["session_label","run_order_in_session"])
                values=b.value_ns.to_numpy(dtype=float)
                values[b.is_infinite.astype(bool).to_numpy()]=np.inf
                assert np.array_equal(a,values), (split,coordinate)
        result[campaign]["selected_vectors_equal_archived_core_release"]=True
phase=pd.read_parquet(ROOT/"data/e01/derived/phase_sweep.parquet")
assert len(phase)==1720 and (phase.phase_bound_ns<=phase.maxwait_bound_ns).all()
result["phase"]={"instances":len(phase),"phase_dominance_violations":int((phase.phase_bound_ns>phase.maxwait_bound_ns).sum())}
coupling=pd.read_parquet(ROOT/"data/tds1/derived/coupling_checks.parquet")
validation=pd.read_parquet(ROOT/"data/tds1/derived/validation_results.parquet")
summary=pd.read_parquet(ROOT/"data/tds1/derived/repetition_summary.parquet")
assert len(coupling)==5200 and coupling.marginals_preserved.all()
assert coupling.event_inclusion_violations.sum()==0
assert len(summary)==12000 and summary.unsafe_pipeline_upgrade.sum()==0
main_count=int(validation.loc[validation.item_id.eq("E2E"),"n_units"].sum())
recoupled_count=int(coupling.n_units.sum())
assert main_count+recoupled_count==60_200_000
result["tds1"]={"repetitions":len(summary),"unsafe_upgrades":0,
                "TD4_replay_instances":main_count+recoupled_count,
                "TD4_coupling_rows":len(coupling),"TD4_multiset_checks":4*len(coupling)}
print(json.dumps(result,indent=2))
