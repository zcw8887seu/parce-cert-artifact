# PARCE-Cert: data and code

Materials supporting *Phase-aware timing certification with selection-valid bounds and explicit dependence assumptions*.

Release **v0.3.0** expands the previously published core artifact with the event-level and simulation data, analysis sources and figure inputs used in the manuscript. It does not introduce new experiments or change the reported scientific outcomes.

## Contents

| Directory | Contents |
|---|---|
| core_artifact/ | Executable timing, statistical and allocation methods; reduced H3R1 data and reconstruction tests. |
| data/e01/ | 278,600 physical event records and the initial campaign's derived results. |
| data/s1/ | Corrected selection-validity repetitions, summaries and run-order diagnostics. |
| data/h3r1/ | 260,000 physical event records, calibration menus, selected design and validation results. |
| data/tds1/ | Temporal-dependence simulation parameters, repetition-level results and oracle values. |
| implementation/ | Collection and analysis sources, with separate H3R1 and TDS1 modules. |
| figure_data/, figure_code/ | Inputs and code for the eight paper figures. |

See [DATA_GUIDE.md](DATA_GUIDE.md) for units, statistical samples and table meanings. The independently usable core is in `core_artifact/`; its package version remains 0.2.0. Earlier GitHub tags and Zenodo records retain the original core-only releases.

## Download and cite

- [GitHub release v0.3.0](https://github.com/zcw8887seu/parce-cert-artifact/releases/tag/v0.3.0): download `parce-cert-data-code-v0.3.0.zip` for this complete data/code package.
- [Zenodo version series](https://doi.org/10.5281/zenodo.22255539): select version v0.3.0 and use its version-specific DOI when citing these results. The series DOI resolves to the latest published version.

Authors: Chengwei Zhang and Yun Wang, School of Computer Science and Engineering, Southeast University, Nanjing, China. Chengwei Zhang: [ORCID 0009-0008-4623-4570](https://orcid.org/0009-0008-4623-4570). Yun Wang is the corresponding author.

## Reconstruct the retained results

Python 3.11 or newer. From this directory on Linux:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python verify_bundle.py
    python core_artifact/scripts/reconstruct_h3r1.py
    PYTHONPATH=core_artifact/src python -m pytest core_artifact/tests -q

The verification script replays the first feasible Gate target for all 538,600 physical records, compares selected H3R1 vectors with the reduced data, checks phase dominance and recomputes TD4 replay counts. The core reconstruction recomputes stage menus, the selected candidate bound, nominal validation and the H3 disposition. Neither command collects new data.

To generate the figures:

    python figure_code/build_paper_figures.py

PDFs, PNGs and the plotting summary are written to figures/; input files are not modified. Figure code was developed with OpenAI ChatGPT/Codex assistance.

## Interpretation

Physical timestamps have session-relative origins: subtracting one whole-period offset per session preserves duration differences, periodic phases and event order. Session labels are neutral. Do not compare absolute time origins between sessions.

The 538,600 event rows are not independent statistical units. H3R1 uses one selected instance per valid run: 3,000 calibration and 3,500 validation runs. Losses, flags and adverse outcomes are retained. The H3 result remains INCONCLUSIVE_CORRELATION_SENSITIVITY; 15,544,694 ns is a candidate bound, not an issued certificate.

TDS1 synthetic input sequences are regenerated from the stored parameters and seeds rather than individually archived. Full simulation regeneration is distinct from the retained-result checks above. Physical collection is environment-sensitive. See [implementation/README.md](implementation/README.md) for source organization and the limits of the historical command-line interfaces.

## License

Original code and documentation: [MIT](LICENSE). Data: [CC BY 4.0](DATA_LICENSE.md). Authorship and citation metadata for the separately released core are retained in core_artifact/.
