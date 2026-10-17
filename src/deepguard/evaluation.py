"""Metrics and plots computed exclusively from supplied held-out predictions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn import metrics
from .utils import write_json


def score(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not set(y) <= {0, 1}:
        raise ValueError("Nonempty, equally shaped binary labels and probabilities required")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite in [0,1]")
    decision = p >= .5
    tn, fp, fn, tp = metrics.confusion_matrix(y, decision, labels=[0, 1]).ravel()
    ece = 0.
    bins = np.minimum((p*10).astype(int),9)
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    both = len(np.unique(y)) == 2
    return {"n": len(y), "accuracy": float(metrics.accuracy_score(y, decision)),
            "precision": float(metrics.precision_score(y, decision, zero_division=0)),
            "recall": float(metrics.recall_score(y, decision, zero_division=0)),
            "f1": float(metrics.f1_score(y, decision, zero_division=0)),
            "balanced_accuracy": float(metrics.balanced_accuracy_score(y, decision)),
            "roc_auc": float(metrics.roc_auc_score(y, p)) if both else None,
            "pr_auc": float(metrics.auc(*reversed(metrics.precision_recall_curve(y, p)[:2]))) if both else None,
            "average_precision": float(metrics.average_precision_score(y, p)) if both else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "false_positive_rate": float(fp/(fp+tn)) if fp+tn else None,
            "false_negative_rate": float(fn/(fn+tp)) if fn+tp else None,
            "brier": float(metrics.brier_score_loss(y, p)), "ece_10_equal_width": float(ece)}


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    if len(np.unique(labels)) != 2 or len(labels) < 4:
        raise ValueError("Calibration needs a separate validation set with both classes")
    def objective(t: float) -> float:
        return float(np.mean(np.logaddexp(0, logits/t) - labels*logits/t))
    return float(minimize_scalar(objective, bounds=(.05, 20), method="bounded").x)


def evaluate(y: np.ndarray, p: np.ndarray, output: str, groups: np.ndarray | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    result = score(y, p)
    if groups is not None:
        rng = np.random.default_rng(42)
        unique = np.unique(groups)
        estimates = []
        if len(unique) >= 3:
            for _ in range(500):
                ids = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique))])
                estimates.append(metrics.accuracy_score(y[ids], p[ids] >= .5))
            result["accuracy_group_bootstrap_95_ci"] = np.quantile(estimates, [.025, .975]).tolist()
    write_json(folder / "metrics.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    metrics.ConfusionMatrixDisplay.from_predictions(y, p >= .5, labels=[0, 1], ax=axes[0], colorbar=False)
    if len(np.unique(y)) == 2:
        metrics.RocCurveDisplay.from_predictions(y, p, ax=axes[1])
        metrics.PrecisionRecallDisplay.from_predictions(y, p, ax=axes[2])
    fig.tight_layout()
    fig.savefig(folder / "curves.png", dpi=140)
    plt.close(fig)
    return result


def selective_risk(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Coverage and error at predeclared symmetric abstention bands."""
    output = []
    for margin in [0., .05, .1, .15, .2, .3, .4]:
        keep = np.abs(p-.5) >= margin
        output.append({"margin":margin, "coverage":float(keep.mean()),
                       "risk":float(np.mean((p[keep]>=.5)!=y[keep])) if keep.any() else None})
    return output
