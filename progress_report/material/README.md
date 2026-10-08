# Progress-report material

Raw material for the APS360 progress report (due 2026-11-20) and the final report: every decision and
every operation on the project, kept as we go (user, 2026-10-08). Entries before 2026-10-08 were
reconstructed from `docs/HANDOFF.md` §12, the git history and the status reports in `reports/`.

| file | what goes in it |
|---|---|
| `01_decisions.md` | every decision: date, what, who decided, why (full wording stays in HANDOFF §12) |
| `02_operations_log.md` | what was run or built, when, on which machine, and the outcome |
| `03_data_and_pipeline.md` | sources, pipeline stages and the numbers each stage produced; current corpus and mixture |
| `04_problems_and_fixes.md` | problems met, their cause, the fix and the measured effect (material for "challenges") |

Rules for keeping it: append, do not rewrite history (if something is superseded, add a new entry that
says so); dates are ISO; numbers are copied from a stage MANIFEST or a report and name it; the
`progress_report/` folder will later hold the report itself, this `material/` subfolder only the notes.
