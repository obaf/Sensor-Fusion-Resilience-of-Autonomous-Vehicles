"""Imbalance-aware, deployment-oriented metrics for frame-level CAN IDS."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def episodes(label: np.ndarray, ts: np.ndarray, gap: float = 0.5):
    """Contiguous attack episodes (merging gaps shorter than `gap` seconds).
    Returns list of (start_idx, end_idx_inclusive)."""
    pos = np.flatnonzero(label == 1)
    if len(pos) == 0:
        return []
    cuts = np.flatnonzero(np.diff(ts[pos]) > gap) + 1
    starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(pos)]])
    return [(int(pos[s]), int(pos[e - 1])) for s, e in zip(starts, ends)]


def evaluate(y: np.ndarray, score: np.ndarray, ts: np.ndarray, flag: np.ndarray | None = None,
             fixed_fpr: float = 1e-4) -> dict:
    """y: 0/1 labels; score: higher = more anomalous; flag: binary decision at the
    detector's own operating point (if None, threshold at max-F1)."""
    y = np.asarray(y).astype(int); score = np.asarray(score, dtype=np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=0.0)
    out = {"n": int(len(y)), "n_attack": int(y.sum())}
    dur_h = max((ts.max() - ts.min()) / 3600.0, 1e-9)
    benign = y == 0
    dur_benign_h = dur_h * benign.mean()
    if y.sum() > 0 and benign.sum() > 0:
        out["pr_auc"] = float(average_precision_score(y, score))
        try:
            out["roc_auc"] = float(roc_auc_score(y, score))
        except Exception:
            out["roc_auc"] = np.nan
        p, r, thr = precision_recall_curve(y, score)
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        k = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
        out["f1_max"] = float(f1[k]); out["thr_f1max"] = float(thr[k]) if len(thr) else np.nan
        # recall at fixed benign false-positive rate
        bscore = np.sort(score[benign])
        t_fixed = bscore[min(len(bscore) - 1, int(np.ceil((1 - fixed_fpr) * len(bscore))))]
        out[f"recall@fpr{fixed_fpr:g}"] = float((score[y == 1] > t_fixed).mean())
    else:
        out.update(pr_auc=np.nan, roc_auc=np.nan, f1_max=np.nan, thr_f1max=np.nan)
    if flag is None:
        flag = score >= out.get("thr_f1max", np.inf)
    flag = np.asarray(flag).astype(bool)
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    out.update(tp=tp, fp=fp, fn=fn, tn=tn)
    out["precision"] = tp / max(tp + fp, 1) if (tp + fp) else np.nan
    out["recall"] = tp / max(tp + fn, 1) if (tp + fn) else np.nan
    out["f1"] = 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fn) else np.nan
    out["fpr"] = fp / max(fp + tn, 1)
    out["fp_per_hour"] = fp / max(dur_benign_h, 1e-9)
    # episode-level detection and latency
    eps = episodes(y, ts)
    lat = []; det = 0
    for s, e in eps:
        hit = np.flatnonzero(flag[s:e + 1])
        if len(hit):
            det += 1; lat.append(ts[s + hit[0]] - ts[s])
    out["n_episodes"] = len(eps)
    out["episode_recall"] = det / len(eps) if eps else np.nan
    out["latency_median_s"] = float(np.median(lat)) if lat else np.nan
    out["latency_p95_s"] = float(np.percentile(lat, 95)) if lat else np.nan
    return out
