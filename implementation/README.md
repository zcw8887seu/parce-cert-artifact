# Implementation sources

`base/` contains the corrected allocator, periodic-service reference, selection simulation, diagnostics, C++ collection programs and tests. The S1 selection and diagnostic modules are identical to this version and are not duplicated. `h3r1/` and `tds1/` contain their distinct later modules; shared imports resolve against `base/python/parce_exp/`.

These are version-separated collection and analysis components, not a single installed package. Use `../core_artifact/` and the commands in the root README for the supported retained-result reconstruction.

The former `report.py` campaign controller is reduced to numerical summary helpers. Historical freeze-plan, platform-capture, handoff and report-building commands are not public entry points. Tests retain numerical assertions but no longer automatically write campaign reports. Some historical collection/simulation controllers also perform statistical analysis and therefore remain; their full CLI workflows expect the original campaign layout and are not validated here as a turnkey pipeline.

Scientific seed derivation, model definitions, state isolation and fail-closed checks remain unchanged. Do not interchange the retained version-specific modules when regenerating synthetic results. Low-level C++ collection requires Linux and CMake; its operating-system timing is not bit-for-bit reproducible on another host.

To run the numerical report-summary tests from the archive root:

```bash
PYTHONPATH=implementation/base/python python -m pytest implementation/base/tests/test_report_consistency.py -q
```
