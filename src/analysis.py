"""Statistical evidence for model decisions.

Every function here answers one question: **is this difference real, or is it noise?**

The rule this repo runs on: a change is adopted only when a paired test on the
same validation samples says the improvement is unlikely to be chance. Comparing
two headline accuracies (0.912 vs 0.908) and declaring a winner is the single most
common way Kaggle time gets wasted - that gap is usually well inside the noise band.

Plot conventions follow one fixed system so every chart in the repo reads the same:
  - Okabe-Ito categorical palette, assigned in FIXED order, never cycled.
  - Sequential magnitude (confusion matrix) = one hue, light -> dark. Never a rainbow.
  - One y-axis per chart, always. Two scales -> two charts.
  - Recessive grid, thin marks, legend whenever there are 2+ series.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

# Okabe-Ito: the standard colorblind-safe categorical set. Fixed order, never cycled.
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
INK = "#222222"
MUTED = "#888888"
GRID = "#DDDDDD"


def use_house_style():
    """Call once per notebook session."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "font.size": 9,
        }
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

METRICS = {
    "accuracy": lambda y, p: accuracy_score(y, p.argmax(1)),
    "acc": lambda y, p: accuracy_score(y, p.argmax(1)),
    "macro_f1": lambda y, p: f1_score(y, p.argmax(1), average="macro"),
    "f1": lambda y, p: f1_score(y, p.argmax(1), average="macro"),
    "log_loss": lambda y, p: -log_loss(
        y, p, labels=list(range(p.shape[1]))
    ),  # negated: higher=better
}


# --------------------------------------------------------------------------
# 0. Decision rule: is plain argmax the best way to turn probs into labels?
# --------------------------------------------------------------------------


def _macro_f1_mult(probs, targets, mult):
    return f1_score(targets, (probs * mult).argmax(1), average="macro")


def fit_class_multipliers(probs, targets, num_classes, grid=None, passes=3):
    """Coordinate ascent on per-class probability multipliers, maximising macro F1.

    Under class imbalance, plain argmax is not the macro-F1-optimal decision rule:
    macro F1 weights a rare class as heavily as a common one, so trading a little
    majority-class precision for minority-class recall can raise the score. This
    searches one multiplier per class.

    FIT ONLY. Always evaluate on data not used here - see tune_class_multipliers.
    """
    if grid is None:
        grid = np.linspace(0.4, 2.6, 45)
    mult = np.ones(num_classes, dtype=np.float64)
    best = _macro_f1_mult(probs, targets, mult)
    for _ in range(passes):
        improved = False
        for c in range(num_classes):
            for g in grid:
                trial = mult.copy()
                trial[c] = g
                s = _macro_f1_mult(probs, targets, trial)
                if s > best + 1e-9:
                    best, mult, improved = s, trial, True
        if not improved:
            break
    return mult, best


def tune_class_multipliers(out_dir, exp_name, folds, num_classes=4, verbose=True):
    """Nested (leave-one-fold-out) validation of the multiplier trick.

    For each fold: fit multipliers on the OTHER folds' OOF, score on this fold.
    The held-out numbers are the only honest estimate - the in-fit number is
    always optimistic and must never be reported as the result.

    Returns per-fold argmax vs tuned macro F1 plus the multipliers fitted on all folds.
    """
    from src.utils import load_oof

    data = {f: load_oof(out_dir, exp_name, f) for f in folds}
    rows = []
    for held in folds:
        tr_p = np.concatenate([data[f][0] for f in folds if f != held])
        tr_t = np.concatenate([data[f][1] for f in folds if f != held])
        va_p, va_t = data[held]

        mult, fit_score = fit_class_multipliers(tr_p, tr_t, num_classes)
        base = _macro_f1_mult(va_p, va_t, np.ones(num_classes))
        tuned = _macro_f1_mult(va_p, va_t, mult)
        rows.append({
            "held_fold": held, "argmax": base, "tuned": tuned,
            "delta": tuned - base, "fit_score": fit_score,
            "mult": np.round(mult, 3),
        })
        if verbose:
            print(f"  fold {held}: argmax {base:.4f} -> tuned {tuned:.4f} "
                  f"({tuned-base:+.4f})  mult={np.round(mult, 2)}")

    df = pd.DataFrame(rows)
    deltas = df["delta"].values
    n_pos = int((deltas > 0).sum())

    all_p = np.concatenate([data[f][0] for f in folds])
    all_t = np.concatenate([data[f][1] for f in folds])
    final_mult, _ = fit_class_multipliers(all_p, all_t, num_classes)

    if deltas.mean() > 0.01 and n_pos >= max(4, len(folds) - 1):
        verdict = "ADOPT"
        why = f"mean {deltas.mean():+.4f} held-out, improves {n_pos}/{len(folds)} folds"
    elif deltas.mean() > 0 and n_pos > len(folds) / 2:
        verdict = "WEAK"
        why = f"mean {deltas.mean():+.4f} - positive but under the 0.01 bar"
    else:
        verdict = "REJECT"
        why = f"mean {deltas.mean():+.4f}, only {n_pos}/{len(folds)} folds improve"

    if verbose:
        print(f"\n  held-out delta: {deltas.mean():+.4f} +/- {deltas.std(ddof=1):.4f}")
        print(f"  folds improved: {n_pos}/{len(folds)}")
        print(f"  multipliers fitted on all folds: {np.round(final_mult, 3)}")
        print(f"  VERDICT: {verdict} - {why}")

    return {"per_fold": df, "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "n_improved": n_pos, "verdict": verdict, "why": why,
            "multipliers": final_mult}


def per_class_report(targets, probs, classes, n_boot=1000, seed=42):
    """Per-class precision / recall / F1 / support with bootstrap CIs on F1.

    The CI is what stops you acting on a class whose score is built on 40 samples.
    """
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        targets, probs.argmax(1), labels=range(len(classes)), zero_division=0)

    rng = np.random.default_rng(seed)
    n = len(targets)
    boot = np.empty((n_boot, len(classes)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = f1_score(targets[idx], probs[idx].argmax(1),
                           labels=range(len(classes)), average=None, zero_division=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    df = pd.DataFrame({
        "class": classes, "precision": p.round(4), "recall": r.round(4),
        "f1": f.round(4), "f1_lo": lo.round(4), "f1_hi": hi.round(4), "support": s,
    })
    print(df.to_string(index=False))
    print(f"\nmacro F1 = {f.mean():.4f}   (weakest class: {classes[int(f.argmin())]} "
          f"at {f.min():.4f})")
    return df


# --------------------------------------------------------------------------
# 1. How uncertain is a single score?
# --------------------------------------------------------------------------


def bootstrap_ci(targets, probs, metric="accuracy", n_boot=2000, alpha=0.05, seed=42):
    """95% confidence interval for one model's score, by resampling the val set.

    Reported as "0.912 [0.898, 0.925]". If a rival model's point estimate sits
    inside this interval, you do NOT have evidence that they differ.
    """
    rng = np.random.default_rng(seed)
    fn = METRICS[metric]
    n = len(targets)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = fn(targets[idx], probs[idx])
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "point": fn(targets, probs),
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
    }


# --------------------------------------------------------------------------
# 2. Is model B actually better than model A?
# --------------------------------------------------------------------------


def paired_bootstrap(
    targets, probs_a, probs_b, metric="accuracy", n_boot=2000, seed=42
):
    """Paired test: resamples the SAME indices for both models.

    Pairing removes "these images happened to be easy" from the comparison, which
    is most of the variance. Far more sensitive than comparing two independent CIs.

    Returns p_b_better = fraction of resamples where B beat A. Treat >= 0.95 as
    real evidence, 0.80-0.95 as weak, < 0.80 as no evidence.
    """
    rng = np.random.default_rng(seed)
    fn = METRICS[metric]
    n = len(targets)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = fn(targets[idx], probs_b[idx]) - fn(targets[idx], probs_a[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "observed_diff": fn(targets, probs_b) - fn(targets, probs_a),
        "lo": lo,
        "hi": hi,
        "p_b_better": float((diffs > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),  # CI excludes zero
    }


def mcnemar(targets, probs_a, probs_b):
    """Exact McNemar test on the disagreement counts - the textbook test for
    comparing two classifiers on the same samples.

    Only the cases where the models disagree carry information:
      b = A right, B wrong    c = A wrong, B right
    Under "no real difference", b and c are a coin flip.
    """
    a_ok = probs_a.argmax(1) == targets
    b_ok = probs_b.argmax(1) == targets
    b_only = int((a_ok & ~b_ok).sum())
    c_only = int((~a_ok & b_ok).sum())
    if b_only + c_only == 0:
        return {
            "a_right_b_wrong": 0,
            "a_wrong_b_right": 0,
            "p_value": 1.0,
            "significant": False,
        }
    p = stats.binomtest(c_only, b_only + c_only, 0.5).pvalue
    return {
        "a_right_b_wrong": b_only,
        "a_wrong_b_right": c_only,
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def cv_summary(out_dir, exp_name, folds, metric="accuracy"):
    """Per-fold scores + mean/std across folds.

    Report as mean +/- std. A change that improves the mean by less than one std
    of the fold spread is not a demonstrated improvement.
    """
    from src.utils import load_oof

    fn = METRICS[metric]
    scores = []
    for f in folds:
        probs, targets = load_oof(out_dir, exp_name, f)
        scores.append(fn(targets, probs))
    scores = np.array(scores)
    return {
        "exp": exp_name,
        "metric": metric,
        "folds": list(folds),
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(ddof=1) if len(scores) > 1 else 0.0,
    }


# --------------------------------------------------------------------------
# 3. The decision rule
# --------------------------------------------------------------------------


def decide(
    targets, probs_a, probs_b, name_a="baseline", name_b="candidate", metric="accuracy"
):
    """Runs both tests and returns an explicit verdict. Print this into DECISIONS.md.

    ADOPT   - both tests agree B is better
    WEAK    - tests disagree or effect is marginal; needs more folds before adopting
    REJECT  - no evidence of improvement; keep the simpler/cheaper model
    """
    boot = paired_bootstrap(targets, probs_a, probs_b, metric)
    mc = mcnemar(targets, probs_a, probs_b)

    if boot["significant"] and mc["significant"] and boot["observed_diff"] > 0:
        verdict = "ADOPT"
        why = "paired bootstrap CI excludes 0 and McNemar p < 0.05"
    elif boot["observed_diff"] > 0 and (
        boot["p_b_better"] >= 0.80 or mc["p_value"] < 0.20
    ):
        verdict = "WEAK"
        why = "points the right way but below the evidence bar - run more folds"
    else:
        verdict = "REJECT"
        why = "difference is within noise; prefer the simpler model"

    print(f"{name_a} vs {name_b} [{metric}]")
    print(f"  observed diff : {boot['observed_diff']:+.4f}")
    print(f"  95% CI        : [{boot['lo']:+.4f}, {boot['hi']:+.4f}]")
    print(f"  P(B better)   : {boot['p_b_better']:.3f}")
    print(
        f"  McNemar       : {mc['a_wrong_b_right']} flips to B, "
        f"{mc['a_right_b_wrong']} to A, p={mc['p_value']:.4f}"
    )
    print(f"  VERDICT       : {verdict} - {why}")
    return {"verdict": verdict, "why": why, "bootstrap": boot, "mcnemar": mc}


# --------------------------------------------------------------------------
# 4. Charts
# --------------------------------------------------------------------------


def plot_learning_curves(run_log: pd.DataFrame, exp_name=None, ax=None):
    """Train vs valid loss. THE overfitting diagnostic.

    Read it: curves converging = still learning. Valid flattening while train keeps
    dropping = starting to overfit. Valid turning upward = overfitting; the useful
    epoch was the minimum, and early stopping should have caught it.
    """
    df = run_log if exp_name is None else run_log[run_log.exp == exp_name]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.6))
    for fold, g in df.groupby("fold"):
        ax.plot(
            g.epoch,
            g.train_loss,
            color=PALETTE[0],
            alpha=0.9,
            label="train" if fold == df.fold.min() else None,
        )
        ax.plot(
            g.epoch,
            g.val_loss,
            color=PALETTE[1],
            alpha=0.9,
            label="valid" if fold == df.fold.min() else None,
        )
    best = df.loc[df.val_loss.idxmin()]
    ax.axvline(best.epoch, color=MUTED, ls=":", lw=1)
    ax.annotate(
        f"best valid\nepoch {int(best.epoch)}",
        (best.epoch, best.val_loss),
        textcoords="offset points",
        xytext=(8, 10),
        fontsize=8,
        color=MUTED,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Learning curves - {exp_name or 'all runs'}")
    ax.legend()
    return ax


def plot_fold_box(summaries, ax=None):
    """Box plot of per-fold CV scores, one box per experiment.

    This is the chart that stops you chasing noise: if the boxes overlap heavily,
    the difference in means is not real. Individual fold scores are overlaid as
    dots, because with 5 folds a box alone hides how few points there are.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * len(summaries) + 2.5, 3.8))
    data = [s["scores"] for s in summaries]
    names = [s["exp"] for s in summaries]

    bp = ax.boxplot(
        data,
        labels=names,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color=INK, lw=1.6),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(markeredgecolor=MUTED),
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(0)
    for i, (scores, color) in enumerate(zip(data, PALETTE), start=1):
        ax.scatter(
            rng.normal(i, 0.045, len(scores)),
            scores,
            s=26,
            color=color,
            zorder=3,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_ylabel(summaries[0]["metric"])
    ax.set_title("CV score by fold")
    ax.margins(y=0.22)  # headroom so the mean±std labels clear the title
    for i, s in enumerate(summaries, start=1):
        ax.annotate(
            f"{s['mean']:.4f}\n±{s['std']:.4f}",
            (i, max(s["scores"])),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    return ax


def plot_confusion(targets, probs, classes, normalize=True, ax=None):
    """Where the errors actually are. Sequential single hue, light -> dark.

    Read the off-diagonal: a hot cell means two classes are being confused, which
    is a targeted fix (more data, better augmentation, higher resolution) rather
    than "train longer".
    """
    cm = confusion_matrix(targets, probs.argmax(1), labels=range(len(classes)))
    if normalize:
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.5 * len(classes) + 3, 0.5 * len(classes) + 2.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    ax.grid(False)
    if len(classes) <= 15:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] > 0.005:
                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK,
                    )
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax


def plot_per_class_f1(targets, probs, classes, ax=None):
    """Per-class F1 sorted worst-first, with the support count on each bar.

    Tells you where to spend effort. A low-F1 class with high support is a real
    modelling problem; low F1 with 20 samples is mostly sampling noise.
    """
    f1s = f1_score(targets, probs.argmax(1), average=None, labels=range(len(classes)))
    support = np.bincount(targets, minlength=len(classes))
    order = np.argsort(f1s)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.3 * len(classes) + 2))
    y = np.arange(len(classes))
    ax.barh(y, f1s[order], color=PALETTE[0], height=0.7)
    ax.axvline(
        f1s.mean(),
        color=PALETTE[4],
        ls="--",
        lw=1.5,
        label=f"macro avg {f1s.mean():.3f}",
    )
    ax.set_yticks(y, [classes[i] for i in order])
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1)
    ax.set_title("Per-class F1 (worst first)")
    for i, idx in enumerate(order):
        ax.text(
            f1s[idx] + 0.015,
            i,
            f"n={support[idx]}",
            va="center",
            fontsize=7,
            color=MUTED,
        )
    ax.legend(loc="lower right")
    return ax


def plot_reliability(targets, probs, n_bins=10, ax=None):
    """Calibration: does "90% confident" actually mean right 90% of the time?

    Bars below the diagonal = overconfident, the classic sign of an overfit net.
    Matters directly if the competition metric is log loss / AUC rather than accuracy.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == targets).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(conf, bins[1:-1])
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    ece = sum(n * abs(x - y) for x, y, n in zip(xs, ys, ns)) / sum(ns)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="perfect calibration")
    ax.plot(
        xs,
        ys,
        "o-",
        color=PALETTE[0],
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1,
        label="model",
    )
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"Reliability (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return ax


def plot_cv_vs_lb(experiments: pd.DataFrame, ax=None):
    """CV score vs public leaderboard score, one dot per submission.

    The most important chart in any competition. If the dots trend upward, your CV
    is trustworthy and you can iterate offline without burning submissions. If they
    scatter randomly, your validation split is wrong - fix that before anything else.
    """
    df = experiments.dropna(subset=["cv", "lb"])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(
        df.cv, df.lb, s=70, color=PALETTE[0], edgecolor="white", linewidth=1.2, zorder=3
    )
    for _, r in df.iterrows():
        ax.annotate(
            r["exp"],
            (r.cv, r.lb),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7.5,
            color=MUTED,
        )
    if len(df) >= 3:
        r, p = stats.pearsonr(df.cv, df.lb)
        m, b = np.polyfit(df.cv, df.lb, 1)
        xs = np.linspace(df.cv.min(), df.cv.max(), 10)
        ax.plot(xs, m * xs + b, color=PALETTE[1], lw=1.5, ls="--")
        verdict = "trust CV" if r > 0.7 else "CV unreliable - fix the split"
        ax.set_title(f"CV vs LB - r={r:.2f}, p={p:.3f} ({verdict})")
    else:
        ax.set_title("CV vs LB (need 3+ submissions)")
    ax.set_xlabel("local CV")
    ax.set_ylabel("public LB")
    return ax


def overfit_check(run_log: pd.DataFrame, exp_name=None, fold=None, metric="f1"):
    """Turn 'is it overfitting?' into four measurable tests with a verdict.

    Opinion is not evidence. These are the numbers that decide it:

      1. VAL LOSS TURN   - epochs since validation loss bottomed out.
                           Rising val loss + falling train loss = overfitting.
      2. LOSS GAP        - (val_loss - train_loss) at the best epoch vs at the end.
                           A widening gap means the model is memorising.
      3. METRIC PLATEAU  - did the competition metric still improve after the turn?
                           If yes, the overfitting is not costing us anything yet.
      4. BEST-EPOCH LAG  - how far the best epoch sits from the last one. Best epoch
                           at the very end means it was still learning: train longer.
    """
    df = run_log.copy()
    if exp_name is not None:
        df = df[df.exp == exp_name]
    if fold is not None:
        df = df[df.fold == fold]
    df = df.sort_values("epoch")
    if len(df) < 3:
        return {"verdict": "UNKNOWN", "why": "need at least 3 epochs"}

    metric_col = metric
    if metric_col not in df.columns:
        if metric in ("macro_f1", "f1") and "f1" in df.columns:
            metric_col = "f1"
        elif metric in ("accuracy", "acc") and "acc" in df.columns:
            metric_col = "acc"
        else:
            raise KeyError(
                f"Metric column '{metric}' not found in run_log columns: {list(df.columns)}"
            )

    last = int(df.epoch.iloc[-1])
    val_min_ep = int(df.loc[df.val_loss.idxmin(), "epoch"])
    best_ep = int(df.loc[df[metric_col].idxmax(), "epoch"])

    val_rise = float(df.val_loss.iloc[-1] - df.val_loss.min())
    gap_at_best = float(
        df.loc[df.epoch == val_min_ep, "val_loss"].iloc[0]
        - df.loc[df.epoch == val_min_ep, "train_loss"].iloc[0]
    )
    gap_at_end = float(df.val_loss.iloc[-1] - df.train_loss.iloc[-1])
    metric_after_turn = float(
        df[df.epoch >= val_min_ep][metric_col].max()
        - df.loc[df.epoch == val_min_ep, metric_col].iloc[0]
    )

    turned = val_min_ep <= last - 2 and val_rise > 0.01
    widening = gap_at_end > gap_at_best + 0.05
    still_gaining = metric_after_turn > 0.005

    if turned and widening and not still_gaining:
        verdict, why = (
            "OVERFITTING",
            "val loss rising, gap widening, metric no longer improving",
        )
    elif turned and widening:
        verdict, why = (
            "MILD",
            "val loss turned up but the metric is still improving - little cost",
        )
    elif best_ep >= last:
        verdict, why = (
            "UNDERTRAINED",
            "best epoch is the last epoch - it was still learning",
        )
    else:
        verdict, why = "HEALTHY", "no sustained rise in validation loss"

    r = {
        "verdict": verdict,
        "why": why,
        "last_epoch": last,
        "val_loss_min_epoch": val_min_ep,
        "best_metric_epoch": best_ep,
        "val_loss_rise_since_min": round(val_rise, 4),
        "gap_at_best": round(gap_at_best, 4),
        "gap_at_end": round(gap_at_end, 4),
        "metric_gain_after_turn": round(metric_after_turn, 4),
    }

    print(
        f"OVERFIT CHECK - {exp_name or 'all'}"
        + (f" fold {fold}" if fold is not None else "")
    )
    print(
        f"  val loss bottomed at epoch {val_min_ep} of {last}, then rose {val_rise:+.4f}"
    )
    print(f"  loss gap  {gap_at_best:.3f} (best) -> {gap_at_end:.3f} (end)")
    print(f"  {metric_col} gain after the turn: {metric_after_turn:+.4f}")
    print(f"  best {metric_col} at epoch {best_ep}")
    print(f"  VERDICT: {verdict} - {why}")
    return r


def evidence_report(cfg, exp_name, folds, classes, run_log=None, out_png=None):
    """One call -> the four-panel figure that should accompany every decision."""
    from src.utils import load_oof

    use_house_style()

    probs, targets = load_oof(cfg.out_dir, exp_name, folds[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    metric = getattr(cfg, "metric", "macro_f1")
    if run_log is not None:
        plot_learning_curves(run_log, exp_name, ax=axes[0, 0])
    plot_fold_box(
        [cv_summary(cfg.out_dir, exp_name, folds, metric=metric)], ax=axes[0, 1]
    )
    plot_confusion(targets, probs, classes, ax=axes[1, 0])
    plot_reliability(targets, probs, ax=axes[1, 1])

    ci = bootstrap_ci(targets, probs, metric=metric)
    fig.suptitle(
        f"{exp_name} - {metric} {ci['point']:.4f} "
        f"[{ci['lo']:.4f}, {ci['hi']:.4f}] 95% CI",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=130)
        print(f"saved {out_png}")
    return fig
