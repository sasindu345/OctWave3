# CLAUDE.md

The operating rules for this repo live in **[AGENTS.md](AGENTS.md)** — read it first
and follow it exactly. It is the single source of truth for how work is done here.

Short version, so it is never skipped:

1. **Verify, don't guess.** No claim about the dataset without running code and
   reading the output. `UNKNOWN` is an acceptable answer; a confident guess is not.
2. **No invented numbers.** Never report a metric not computed this session.
3. **`decide()` or it didn't happen.** Model comparisons go through the paired
   bootstrap + McNemar test in [src/analysis.py](src/analysis.py). Two headline
   scores are not a comparison.
4. **Every decision ships with a chart** and a stated uncertainty (CI or fold std).
5. **Ties go to the simpler model.**

Project layout and setup: [README.md](README.md) · [docs/SETUP.md](docs/SETUP.md) ·
[docs/WORKFLOW.md](docs/WORKFLOW.md)
