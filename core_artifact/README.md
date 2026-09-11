# PARCE-Cert core

Core algorithms and reduced data from [parce-cert-artifact v0.2.0](https://github.com/zcw8887seu/parce-cert-artifact). The numerical sources and data are unchanged; this copy has condensed documentation. The parent archive contains the additional physical records and simulation outputs.

The implementation provides integer periodic-service timing, order-statistic bounds, one-sided Clopper–Pearson limits and exact finite-menu allocation for memoryless serial chains. Risks use exact finite-decimal semantics. The allocator minimizes (bound, risk sum, lexicographic menu IDs) and rejects repeated uncertainty coordinates, which require an augmented-state implementation.

## Use

From this directory, after creating a Python environment:

    python -m pip install -e '.[test]'
    python examples/run_demo.py
    pytest -q
    python scripts/reconstruct_h3r1.py

The demo illustrates a Gate-boundary transition, order-statistic indices, nominal validation and allocation against an exhaustive oracle. H3R1 reconstruction computes the selected envelopes, candidate bound, five validation rows and session-aware run-order diagnostics. To save its numerical outputs:

    python scripts/reconstruct_h3r1.py --write-dir reconstructed_h3r1

## Data and interpretation

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md). Reduced H3R1 records contain durations and within-session order, not raw timestamps. The physical result remains INCONCLUSIVE_CORRELATION_SENSITIVITY; nominal validation does not override the acquisition guard. Temporal-dependence results concern the four tested processes, not arbitrary dependent sequences. Runtime tables are environment-specific measurements.

## Attribution

Chengwei Zhang and Yun Wang, School of Computer Science and Engineering, Southeast University. Citation metadata are in CITATION.cff.

Code and documentation: [MIT](LICENSE). Data: [CC BY 4.0](DATA_LICENSE.md).
