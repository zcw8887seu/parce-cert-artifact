# PARCE-Cert Core Reproducibility Artifact

This repository contains a deliberately small, host-independent artifact for the paper **“PARCE-Cert: Conditional End-to-End Timing Certification with Selection-Valid Phase-Aware Composition and Exact Chain Risk Allocation.”**

The release is designed to expose the central executable ideas without publishing the complete physical-event archive or the internal experiment workflow.

## Repository and release

- Repository: <https://github.com/zcw8887seu/parce-cert-artifact>
- Submission release: <https://github.com/zcw8887seu/parce-cert-artifact/releases/tag/v0.1.0>

## Included

- integer reference operator for fit-within-one-window periodic service;
- finite-sample order-statistic menu construction;
- one-sided Clopper--Pearson validation and three-way nominal verdicts;
- exact Pareto-frontier risk allocation for a finite serial chain;
- a brute-force oracle for small allocator instances;
- selected aggregate tables used to report phase, selection, H3, allocator, and temporal-sensitivity results.

## Intentionally excluded

- raw physical-event traces and packet-level logs;
- machine, user, process, filesystem, and hardware identifiers;
- launch scripts and complete experiment-execution documents;
- internal review packages and campaign-management records;
- hashes, manifests, authorization files, and security/integrity checks;
- the full experiment dataset.

The aggregate tables are sufficient to inspect the reported decisions, but they are not a replacement for the full private research archive.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test]"
python examples/run_demo.py
pytest
```

The demo prints:

1. the one-nanosecond gate-boundary transition;
2. the order-statistic indices for the three frozen risk rows;
3. a validation interval and nominal verdict;
4. an exact chain allocation checked against exhaustive enumeration.

## Data

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md). All published CSV files are aggregate or reduced tables. They contain no local paths, usernames, hostnames, MAC/IP addresses, process identifiers, or raw absolute timestamps.

## Scientific boundary

The physical H3 result remains **INCONCLUSIVE** with reason **CORRELATION_SENSITIVITY**. Favorable nominal validation counts do not override the acquisition guard. The temporal-dependence simulation tables describe four frozen data-generating processes and do not establish a theorem for arbitrary dependent sequences.

## Authors

- Chengwei Zhang, School of Computer Science and Engineering, Southeast University. ORCID: [0009-0008-4623-4570](https://orcid.org/0009-0008-4623-4570)
- Yun Wang, School of Computer Science and Engineering, Southeast University. Corresponding author.

## License

The source code and repository documentation are licensed under the
[MIT License](LICENSE). The aggregate CSV files in `data/` are licensed under
[CC BY 4.0](DATA_LICENSE.md).
