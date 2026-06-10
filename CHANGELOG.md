# Changelog

## 2026-06-10 — Repository overhaul

Documentation-only reorganization. No Python file moved, renamed, or edited; no
result changed.

- **README.md** rewritten: what SVA is, headline results with exact configs, how the
  three-stage mechanism works, the late-layer deployment finding, verified quickstart
  commands, repo map, and an explicit limitations section. The old README had grown
  into a 488-line chronological log.
- **docs/research_log.md**: the old README's chronological narrative, preserved
  verbatim.
- **docs/snapshots/**: all 80 dated result snapshots moved here from `results/`
  (unchanged), with a new index grouping them into nine experimental phases.
  `results/` now holds only artifact bundles (`hf_artifacts/`) and gitignored Modal
  run logs (`modal_runs/`).
- **docs/h100_runbook.md**: the Modal H100 launch commands from the old README.
- **experiments/README.md**: new index table — script, what it tests, key result.
- Added `results/modal_runs/.gitkeep` (the path `.gitignore` already expected).

Prior history (2026-05-13 to 2026-05-14) is the research itself; see
`docs/research_log.md` and `git log` before this date.
