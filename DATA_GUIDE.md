# Scientific data guide

All timing values with the suffix _ns are nanoseconds. Integer event timestamps have session-relative origins; they are not wall-clock dates. Durations and modulo-period phases retain their original values. Infinite upper tails are represented by explicit flags or infinity; they must not be dropped or replaced with zero.

## Physical events and inference units

The raw event tables contain one row per recorded instance, keyed by run_id and instance_id. The split, is_warmup, selected_single, valid_event and packet_lost fields distinguish the experimental role and outcome. In H3R1, actual_session_id is a neutral label; planned_session_index and session_order retain the original within-split ordering. Form lag pairs only within a session.

The physical timing coordinates are:

    X1 = t_compute_end_ns - t_cause_ns
    X2 = t_gate_ingress_ns - t_compute_end_ns
    X3 = t_gate_service_start_ns - t_gate_target1_ns
    X4 = t_rx_ns - t_gate_service_start_ns - service_demand_ns
    E2E = t_rx_ns - t_cause_ns

Known packet loss makes X4 and E2E infinite. Check the event flags before forming statistical samples. The Gate uses a 1,000,000-ns period, a [400,000, 700,000)-ns service window and 50,000-ns non-preemptive demand. Additional calendar definitions are in implementation/base/configs/calendar.yaml. The selected H3R1 vectors are reconstructed in verify_bundle.py.

## Derived tables

- stage_samples and calibration/validation_stage_samples: long-form statistical units, coordinate_id, value_ns and infinity/validity indicators.
- menu: one protected stage-risk candidate per row; n_units, order_k, q_ns, alpha_row and status describe its finite-sample acquisition.
- selected_design.json: fixed chosen menu, phase and candidate bound. This is not by itself a certificate.
- validation_results: fixed item family, counts, risk targets, one-sided CP limits and nominal three-way outcomes. The temporal guard is a separate issuance condition.
- run_order_diagnostics and session summaries: split/coordinate lag correlations and within-session descriptive diagnostics.
- phase_sweep: instance-level phase-aware and maximum-wait bounds, observed delay, reference match and rescheduling information.
- selection_validity: repeated known-truth menu-selection experiments. The main comparison and additional-budget cells are distinguished in the schema; do not pool them.
- allocator_results: method/instance outcomes, exactness, frontier peak and environment-specific runtime. Exact-oracle and boundary tables provide the corresponding numerical checks.
- dependence_results and coupling_checks: finite executed alignments and event-inclusion/multiset checks, not population-risk confidence intervals.

## TDS1

repetition_summary contains 12,000 complete repetitions (four processes times 3,000). Its flags preserve nominal acquisition/validation and the frozen guard. calibration_menu_results, selected_designs, validation_results and run_order_diagnostics retain the intermediate objects. dgp_marginal_checks and the five-million-sample oracle retain the reference implementation checks. simulation_parameters.json holds the model parameters and seed schedule.

The corrected TD4 denominator is 60,200,000 executed E2E replay instances: ordinary validation replay plus the recoupled instances. The 5,200 coupling rows each check four preserved stage multisets, for 20,800 multiset checks. These are exact structural audit counts; they are not independent binomial trials.

## Source versions and figures

Use implementation/base for the base algorithms, including the corrected selection-validity code. The H3R1 and TDS1 additions are separate modules; identical copies of base files are not repeated. See implementation/README.md. Figure inputs include full tables and explicit plotting subsets. The reduced core representation is described separately in core_artifact/DATA_DICTIONARY.md.
