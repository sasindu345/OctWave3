# AGENTS.md — operating rules for AI agents in this repo

Read this before doing anything. It exists to stop two specific failure modes:

1. **Hallucination** — an agent inventing dataset facts, metric values, API names,
   or leaderboard results instead of reading them.
2. **Noise-chasing** — an agent (or human) declaring a model "better" because one
   number moved, when the difference is inside the natural variance of the split.

The rules below are not style preferences. They are the working contract.

---

## 1. Never state a fact about the data you have not read

The competition data is **not** in this repo and its details are **not** known
from training data or from the model's memory. Before any claim about it, run code
and read the output.

| Claim | How to establish it | Never |
|---|---|---|
| number of classes | `df.target.nunique()` | guess from the competition name |
| class balance | `df.target.value_counts()` | assume balanced |
| image size / channels | `cv2.imread(p).shape` on real files | assume 224×224 RGB |
| train/test counts | `len(df)`, `len(test_df)` | quote a number from a forum post |
| file layout | `!ls -R data \| head -50` | assume `train/<class>/*.jpg` |
| the metric | the competition's Evaluation page | assume accuracy |

If a fact cannot be verified right now, write `UNKNOWN` and say what command would
resolve it. **`UNKNOWN` is a correct answer. A confident guess is not.**

## 2. Never report a metric you did not compute in this session

No estimated accuracies. No "this should get around 0.93". No filled-in rows in
`logs/EXPERIMENTS.md` for runs that have not finished. Every number in that table
must be traceable to a line in `outputs/run_log.jsonl` or a real LB submission.

If asked "will this improve the score?" the honest answer is a prediction with a
mechanism and an experiment that would test it — not a number.

## 3. Every model comparison goes through `src/analysis.py`

Comparing two headline scores is banned. Use:

```python
from src.analysis import decide
decide(targets, probs_baseline, probs_candidate, "exp01", "exp02")
```

It runs a **paired bootstrap** (95% CI on the difference) and **McNemar's exact
test** on the same validation samples, then returns `ADOPT` / `WEAK` / `REJECT`.

**The decision rule:**

| Verdict | Condition | Action |
|---|---|---|
| `ADOPT` | bootstrap CI on the difference excludes 0 **and** McNemar p < 0.05 | make it the new baseline |
| `WEAK` | points the right way but misses the bar | run more folds before deciding — do **not** adopt yet |
| `REJECT` | difference inside the noise band | keep the simpler/cheaper model |

Ties break toward the **simpler** model: fewer parameters, smaller image size,
shorter training. Complexity has to earn its place with evidence.

**Single-fold results never justify adopting anything.** One fold gives one number
with no spread. A real decision needs ≥ 3 folds, reported as `mean ± std`.

## 4. Every decision ships with a chart

A claim without a plot is an opinion. Required evidence per decision type:

| Question | Chart | Function |
|---|---|---|
| Is it overfitting? | train vs valid learning curves | `plot_learning_curves` |
| Is model B better? | per-fold box plot + paired test | `plot_fold_box` + `decide` |
| Where are the errors? | confusion matrix | `plot_confusion` |
| Which classes are weak? | per-class F1, worst-first, with support | `plot_per_class_f1` |
| Are the probabilities honest? | reliability diagram + ECE | `plot_reliability` |
| **Can I trust my CV?** | CV vs LB scatter + Pearson r | `plot_cv_vs_lb` |

`evidence_report()` renders the standard four-panel figure in one call. Save it to
`outputs/figures/<exp_name>.png` and reference it from the decision log.

## 5. Overfitting guardrails

These are checked before any result is accepted:

- **Fixed folds.** `StratifiedKFold(seed=42)` — the split never changes between
  experiments, or comparisons are meaningless.
- **The test set is touched exactly once per submission.** Never for tuning,
  never for early stopping, never for threshold selection.
- **Watch the gap.** `run_log.jsonl` records `train_loss - val_loss` each epoch.
  A widening gap with flat validation = overfitting. Fix with augmentation, dropout,
  or fewer epochs — not with a bigger model.
- **Leaderboard probing is overfitting.** If local CV and public LB disagree,
  trust CV. The public LB is a small sample and chasing it overfits to it. Check
  `plot_cv_vs_lb` before trusting either.
- **Count your comparisons.** After ~20 experiments on the same validation split,
  the best one is partly selected by luck. Confirm finalists on a fresh seed.
- **Duplicate/near-duplicate images across folds leak.** Check before trusting a
  suspiciously high CV.

## 6. Code honesty

- Do not write code paths for data layouts you have not seen. Leave the branch
  unimplemented with an explicit `raise NotImplementedError` over a plausible guess.
- Do not invent library APIs. If unsure of a `timm`/`albumentations` signature,
  check it: `timm.list_models('*efficientnet*')`, `help(A.CoarseDropout)`.
- Config changes go in `src/config.py`, never hardcoded into the training loop.
- If you changed `src/`, say so and note it in `logs/CHANGELOG.md`.
- **Report failures plainly.** A crashed run, a diverged loss, or a REJECT verdict
  is useful information. Do not soften it or present a partial run as a success.

## 7. Session workflow

```
1. git pull                                   (get latest src/)
2. state what you are testing + your prediction, BEFORE running
3. run it - full output, no summarising away errors
4. decide() against the current baseline
5. append one row to logs/EXPERIMENTS.md and one entry to logs/DECISIONS.md
6. commit
```

Writing the prediction down first is what stops post-hoc storytelling — you cannot
retrofit an explanation onto a result you already committed to predicting.

## 8. Answer format for analysis requests

```
CLAIM:      <one sentence>
EVIDENCE:   <command run + actual output, or chart path>
UNCERTAINTY: <CI, std across folds, or "single fold - not decisive">
VERDICT:    ADOPT / WEAK / REJECT / UNKNOWN
NEXT:       <the one experiment that would resolve the biggest open question>
```

If the EVIDENCE line is empty, the CLAIM does not get made.
