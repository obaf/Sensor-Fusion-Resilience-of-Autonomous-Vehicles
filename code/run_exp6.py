"""Experiment driver.

Part A  — receiver-level GNSS spoofing detection on TEXBAT observables
          (non-AI monitors vs AI, leave-one-scenario-out).
Part B  — sensor-fusion resilience on nuScenes trajectories under GNSS spoofing,
          perception attacks and coordinated attacks; defences F0..F4.
Part C  — learned uncertainty / attack models (GBM + GRU) trained on train scenes,
          evaluated on held-out scenes; ensemble.

Outputs in experiment6/results/: exp6A_*.csv, exp6B_*.csv, exp6C_*.csv, run.log
Usage: python run_exp6.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from gnss_texbat import load_set, epoch_features, derive_onset, identical_prefix_end, GNSSRuleMonitor, FEATS, SETS  # noqa
GUARD = 15.0  # s excluded around the estimated spoofing onset
from fusion_sim import (load_scenes, simulate_sensors, gnss_offset_profile, apply_lodo_bias, run_episode,  # noqa
                        perception_outcomes, texbat_profiles, DT, LANE_ERR, CATASTROPHIC)
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "experiment4" / "code"))
from evaluate import evaluate  # noqa

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
LOG = open(RES / "run.log", "a", encoding="utf8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ================================================================== Part A
def part_a():
    log("== Part A: TEXBAT receiver-level detection")
    raw = {s: load_set(s) for s in SETS}
    F = {s: epoch_features(raw[s]) for s in SETS}
    onsets = {s: derive_onset(F[s], F["cleanStatic"]) for s in ["ds2", "ds3", "ds7"]}
    # ds7 is digitally built on cleanStatic: the exact onset is where the observables first differ
    ds7_exact = identical_prefix_end(raw["ds7"], raw["cleanStatic"])
    log("z-score onsets (receiver time, s):", onsets, "| ds7 identical-prefix end:", ds7_exact, "(documented: 110 s recording time)")
    if not np.isnan(ds7_exact):
        onsets["ds7"] = float(ds7_exact)
    json.dump({"onsets_receiver_time_s": onsets, "ds7_identical_prefix_end_s": None if np.isnan(ds7_exact) else float(ds7_exact),
               "guard_band_s": GUARD}, open(RES / "texbat_onsets.json", "w"), indent=1)
    for s in SETS:
        on = onsets.get(s, 1e9)
        F[s]["label"] = (F[s].t_rel >= on).astype(int)
        # guard band around the (estimated) onset is excluded from scoring
        F[s]["scored"] = ~((F[s].t_rel > on - GUARD) & (F[s].t_rel < on + GUARD))
        F[s]["set"] = s
    # profiles for part B
    prof = {}
    for s in ["ds2", "ds3", "ds7"]:
        c = F["cleanStatic"].set_index(np.round(F["cleanStatic"].t_rel, 1)).dclk; x = F[s].set_index(np.round(F[s].t_rel, 1)).dclk
        idx = c.index.intersection(x.index); diff = (x.loc[idx] - c.loc[idx]); diff = diff - diff.iloc[:50].median()
        sec = np.arange(0, np.array(idx).max(), 1.0); prof[s] = np.interp(sec, np.array(idx), diff.values).tolist()
    json.dump(prof, open(RES / "texbat_profiles.json", "w"))
    feats = FEATS + (["symmdiff_rms"] if "symmdiff_rms" in F["ds3"] else [])
    rows = []
    clean = F["cleanStatic"]; ctr = clean.iloc[: int(0.6 * len(clean))]; cva = clean.iloc[int(0.6 * len(clean)):]
    # ---- non-AI monitors, calibrated on the first 60 % of the clean set
    mon = GNSSRuleMonitor().fit(ctr)
    # calibrate a persistence threshold so that clean-validation false alarm rate <= 1 per 10 min (i.e. 0.5 % epochs at 5 Hz -> use 1 alarm / 3000 epochs)
    sv = mon.score(cva).rule_persist.values
    thr_rule = float(max(1.0, np.quantile(sv, 1 - 1 / 3000)))
    for s in ["ds2", "ds3", "ds7", "cleanStatic"]:
        f = F[s] if s != "cleanStatic" else cva
        sc = mon.score(f)
        m = f.scored.values if "scored" in f else np.ones(len(f), bool)
        y = f.label.values[m]; ts = f.t.values[m]
        for name, col in [("Rule:max", "rule_max"), ("Rule:persist", "rule_persist")] + [(f"Rule:{c[4:]}", c) for c in sc.columns if c.startswith("mag_")]:
            sscore = sc[col].values[m]
            ev = evaluate(y, sscore, ts, sscore >= (thr_rule if col == "rule_persist" else 1.0))
            ev.update(part="A", set=s, detector=name, onset_s=onsets.get(s, np.nan)); rows.append(ev)
        # ---- AI: IsolationForest on clean train; GBM leave-one-scenario-out (train on the other two spoofed sets + clean)
        Xtr = ctr[feats].values; sc_ = StandardScaler().fit(Xtr)
        iso = IsolationForest(n_estimators=300, random_state=0).fit(sc_.transform(Xtr))
        s_if = -iso.score_samples(sc_.transform(f[feats].values))[m]
        thr_if = float(np.quantile(-iso.score_samples(sc_.transform(cva[feats].values)), 1 - 1 / 3000))
        ev = evaluate(y, s_if, ts, s_if >= thr_if); ev.update(part="A", set=s, detector="IsolationForest", onset_s=onsets.get(s, np.nan)); rows.append(ev)
        if s != "cleanStatic":
            others = [o for o in ["ds2", "ds3", "ds7"] if o != s]
            Xo = pd.concat([F[o][F[o].scored] for o in others] + [ctr.assign(label=0)]); yo = Xo.label.values
            gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1, random_state=0).fit(Xo[feats].values, yo)
            p = gbm.predict_proba(f[feats].values)[:, 1][m]
            ev = evaluate(y, p, ts, p >= 0.5); ev.update(part="A", set=s, detector="LightGBM(LOSO)", onset_s=onsets[s]); rows.append(ev)
            from scipy.stats import rankdata
            rp = sc.rule_persist.values[m]
            ens = (rankdata(rp) + rankdata(s_if) + rankdata(p)) / (3 * len(p))
            ev = evaluate(y, ens, ts, (rp >= thr_rule) | (p >= 0.5)); ev.update(part="A", set=s, detector="Ens(Rule|GBM)", onset_s=onsets[s]); rows.append(ev)
            ev = evaluate(y, ens, ts, None); ev.update(part="A", set=s, detector="Ens(rank-mean)", onset_s=onsets[s]); rows.append(ev)
        log(f"  {s}: done")
    A = pd.DataFrame(rows); A.to_csv(RES / "exp6A_results.csv", index=False)
    pd.concat(F.values()).to_parquet(RES / "texbat_epoch_features.parquet")
    return onsets


# ================================================================== Part B/C
GNSS_KINDS = ["ramp", "jump", "texbat_ds2", "texbat_ds3", "texbat_ds7"]
PERC_KINDS = ["ghost", "hide", "shift"]
CONFIGS = ["F0", "F1", "F2", "F3", "F4"]


def episode_grid(quick: bool):
    """(attack_family, gnss_kind, perception_kind, coordinated_lodo, rate)"""
    grid = [("none", "none", "none", False, None)]
    rates = [0.25, 0.5, 1.0, 2.0] if not quick else [0.5, 2.0]
    for r in rates:
        grid.append(("gnss", "ramp", "none", False, r))
    grid += [("gnss", "jump", "none", False, 10.0), ("gnss", "texbat_ds2", "none", False, None), ("gnss", "texbat_ds3", "none", False, None), ("gnss", "texbat_ds7", "none", False, None)]
    grid += [("perception", "none", p, False, None) for p in PERC_KINDS]
    grid += [("perception", "none", "none", True, 1.0)]  # LiDAR-odometry bias only
    for r in rates:
        grid.append(("coordinated", "ramp", "ghost", True, r))
    grid += [("coordinated", "ramp", "hide", True, 1.0), ("coordinated", "texbat_ds7", "ghost", True, None), ("coordinated", "jump", "ghost", True, 10.0)]
    return grid


def monitor_features(L: pd.DataFrame) -> np.ndarray:
    X = L[["d_dr", "d_lo", "nis_gnss", "nis_lodo", "nis_wheel", "cus_dr", "cus_lo", "P_pos", "v_est"]].copy()
    X = X.ffill().fillna(0.0)
    return X.values.astype(np.float32)


def part_b(quick: bool, rx_stats: dict):
    log("== Part B/C: fusion resilience on nuScenes")
    scenes = load_scenes(); profiles = texbat_profiles()
    names = [s["name"] for s in scenes]
    train_scenes = scenes[:6]; test_scenes = scenes[6:]
    reps = 2 if quick else 6
    grid = episode_grid(quick)
    rng = np.random.default_rng(0)

    def make_episode(sc, spec, rep_seed):
        fam, gk, pk, coord, rate = spec
        r = np.random.default_rng(rep_seed)
        sens = simulate_sensors(sc, r)
        n = len(sens.t)
        t_on = int(r.uniform(0.25, 0.45) * n) if fam != "none" else n
        off = gnss_offset_profile(gk, n, t_on, r, sc["yaw"], rate=rate, profiles=profiles)
        # LiDAR-odometry bias: coordinated (follows GNSS pull-off) or standalone (own ramp) when gk == none
        if coord and gk == "none":
            off_l = gnss_offset_profile("ramp", n, t_on, r, sc["yaw"], rate=rate or 1.0)
            lodo = apply_lodo_bias(sens.lodo, off_l, sc["yaw"])
        elif coord:
            lodo = apply_lodo_bias(sens.lodo, off, sc["yaw"])
        else:
            lodo = sens.lodo
        # receiver-level flag (F3): fires with latency drawn from TEXBAT part A, false alarms at its FP rate
        rx = np.zeros(n, bool)
        if gk != "none":
            lat = max(0.0, r.normal(rx_stats.get("latency_median_s", 30.0), 5.0))
            rx[min(n - 1, t_on + int(lat / DT)):] = True
        fp_rate = rx_stats.get("fp_per_hour", 2.0) / 3600.0 * DT
        fp_idx = np.flatnonzero(r.random(n) < fp_rate)
        for i in fp_idx:
            rx[i: i + 20] = True
        return sens, off, lodo, t_on, rx

    # ---- Part C training data: run F1 on train scenes to get monitor features and labels.
    # Detection label = localisation-affecting attack active (GNSS spoofing and/or LiDAR-odometry
    # bias).  Perception-only attacks (ghost/hide/shift) leave the localisation monitors untouched by
    # construction and are therefore *not* positives for these detectors.
    def loc_label(spec, n, t_on):
        fam, gk, pk, coord, rate = spec
        att = np.zeros(n, np.int8)
        if gk != "none" or coord:
            att[t_on:] = 1
        return att
    Xc, yc, yr = [], [], []
    benign_scores_train = {"GBM": [], "GRU": [], "CUSUM": [], "NIS": []}
    for sc in train_scenes:
        for spec in grid:
            for rep in range(reps):
                sens, off, lodo, t_on, rx = make_episode(sc, spec, rep * 7919 + hash(spec[1] + spec[2]) % 1000)
                L = run_episode(sens, off, lodo, "F1")
                X = monitor_features(L); n = len(L)
                att = loc_label(spec, n, t_on)
                Xc.append(X); yc.append(att); yr.append(L.err.values.astype(np.float32))
    Xc = np.concatenate(Xc); yc = np.concatenate(yc); yr = np.concatenate(yr)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0).fit(Xc, yc)
    reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0).fit(Xc, yr)
    learned = {"clf": clf, "reg": reg}
    log(f"  learned models fit on {len(Xc)} steps ({yc.mean():.2f} attack frac)")

    # ---- GRU sequence detector on monitor features (trained on the same train episodes)
    class GRUDet(nn.Module):
        def __init__(self, d, h=32):
            super().__init__(); self.g = nn.GRU(d, h, batch_first=True); self.o = nn.Linear(h, 1)

        def forward(self, x):
            h, _ = self.g(x); return self.o(h).squeeze(-1)
    scaler = StandardScaler().fit(Xc)
    # rebuild sequences (episodes) for the GRU
    seqs = []; ys = []
    pos = 0
    for sc in train_scenes:
        for spec in grid:
            for rep in range(reps):
                n = len(sc["t"]); seqs.append(torch.tensor(scaler.transform(Xc[pos:pos + n]).astype(np.float32))); ys.append(torch.tensor(yc[pos:pos + n].astype(np.float32))); pos += n
    gru = GRUDet(Xc.shape[1]); opt = torch.optim.Adam(gru.parameters(), lr=3e-3)
    T = min(len(s) for s in seqs); Xb = torch.stack([s[:T] for s in seqs]); Yb = torch.stack([y[:T] for y in ys])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1.0, float((1 - Yb.mean()) / max(Yb.mean(), 1e-3)))))
    for ep in range(10 if quick else 25):
        perm = torch.randperm(len(Xb))
        for i in range(0, len(Xb), 16):
            b = perm[i:i + 16]; loss = lossf(gru(Xb[b]), Yb[b]); opt.zero_grad(); loss.backward(); opt.step()
    log("  GRU fit")

    BINARY = {"Monitor:CUSUM(DR|LO)", "RxFlag(TEXBAT-derived)"}

    def det_scores(L, X, rx):
        sc_ = {"Monitor:CUSUM(DR|LO)": L.alarm_active.values.astype(float),
               "Monitor:NIS-GNSS": L.nis_gnss.ffill().fillna(0).values / 13.82,
               "GBM": clf.predict_proba(X)[:, 1],
               "GRU": torch.sigmoid(gru(torch.tensor(scaler.transform(X).astype(np.float32))[None])).detach().numpy()[0],
               "RxFlag(TEXBAT-derived)": rx.astype(float)}
        sc_["Ens(GBM+GRU)"] = 0.5 * (sc_["GBM"] + sc_["GRU"])
        return sc_

    def det_flags(sc_, thr):
        fl = {k: (v >= (0.5 if k in BINARY else thr[k])) for k, v in sc_.items()}
        # OR-ensemble: learned ensemble at its calibrated threshold OR a physics monitor alarm
        fl["Ens(GBM+GRU|CUSUM)"] = fl["Ens(GBM+GRU)"] | fl["Monitor:CUSUM(DR|LO)"]
        sc_["Ens(GBM+GRU|CUSUM)"] = np.maximum(sc_["Ens(GBM+GRU)"], sc_["Monitor:CUSUM(DR|LO)"])
        return fl

    def alarm_events_per_hour(flag, benign_mask):
        f = flag & benign_mask
        rises = int(((f[1:].astype(int) - f[:-1].astype(int)) > 0).sum() + int(f[0]))
        hours = benign_mask.sum() * DT / 3600.0
        return rises / max(hours, 1e-9)

    # ---- thresholds at a 10 false-alarm-per-hour budget (steps), calibrated on benign TRAIN episodes
    ben_scores: dict[str, list] = {}
    for sc in train_scenes:
        for rep in range(reps * 3):
            sens, off, lodo, t_on, rx = make_episode(sc, ("none", "none", "none", False, None), 55_000 + rep)
            L = run_episode(sens, off, lodo, "F1")
            for k_, v_ in det_scores(L, monitor_features(L), rx).items():
                ben_scores.setdefault(k_, []).append(v_)
    thr = {}
    for k_, parts in ben_scores.items():
        v = np.sort(np.concatenate(parts)); hours = len(v) * DT / 3600.0
        kk = int(np.floor(10.0 * hours))
        thr[k_] = float(np.nextafter(v[max(len(v) - kk - 1, 0)], np.inf)) if k_ not in BINARY else 0.5
    thr["Ens(GBM+GRU|CUSUM)"] = thr["Ens(GBM+GRU)"]
    log("  detection thresholds @10 FP/h:", {k: round(v, 3) for k, v in thr.items()})
    json.dump(thr, open(RES / "exp6C_thresholds.json", "w"), indent=1)

    # ---- evaluation on test scenes
    rows = []; det_rows = []
    for sc in test_scenes:
        for spec in grid:
            fam, gk, pk, coord, rate = spec
            for rep in range(reps):
                seed = 100_000 + rep * 7919 + hash(gk + pk) % 1000
                sens, off, lodo, t_on, rx = make_episode(sc, spec, seed)
                n = len(sens.t); att = loc_label(spec, n, t_on)
                for cfg in CONFIGS:
                    L = run_episode(sens, off, lodo, cfg, learned=learned if cfg == "F4" else None, rx_flag=rx if cfg == "F3" else None)
                    prng = np.random.default_rng(seed + 1)
                    po = perception_outcomes(sc, L, sens, pk if pk != "none" else "none", sens.t[t_on] if t_on < n else 1e9, prng)
                    e = L.err.values
                    ea = e[t_on:] if fam != "none" else e
                    rows.append(dict(scene=sc["name"], family=fam, gnss=gk, perception=pk, coordinated=coord, rate=rate, rep=rep, config=cfg,
                                     rmse=float(np.sqrt((e ** 2).mean())), rmse_attack=float(np.sqrt((ea ** 2).mean())) if len(ea) else np.nan,
                                     max_err=float(e.max()), frac_lane=float((e > LANE_ERR).mean()), frac_catastrophic=float((e > CATASTROPHIC).mean()),
                                     gnss_used_frac=float(L.gnss_used[sens.gnss_valid].mean()), alarms=int(L.alarm_dr.iloc[0] + L.alarm_lo.iloc[0]),
                                     **po))
                    if cfg == "F1":  # detection metrics from monitor features (config-independent inputs)
                        X = monitor_features(L)
                        scores = det_scores(L, X, rx); flags = det_flags(scores, thr)
                        for name, s_ in scores.items():
                            ev = evaluate(att, s_, sens.t, flags[name])
                            ev["fp_events_per_hour"] = alarm_events_per_hour(flags[name], att == 0)
                            ev.update(scene=sc["name"], family=fam, gnss=gk, perception=pk, coordinated=coord, rate=rate, rep=rep, detector=name, threshold=thr[name])
                            det_rows.append(ev)
        log(f"  scene {sc['name']} done")
    B = pd.DataFrame(rows); B.to_csv(RES / "exp6B_results.csv", index=False)
    D = pd.DataFrame(det_rows); D.to_csv(RES / "exp6C_detection.csv", index=False)
    # ---- summaries
    S = B.groupby(["family", "coordinated", "config"]).agg(rmse=("rmse", "mean"), rmse_attack=("rmse_attack", "mean"), max_err=("max_err", "mean"),
                                                            frac_lane=("frac_lane", "mean"), frac_catastrophic=("frac_catastrophic", "mean"),
                                                            ghost_brake=("ghost_brake", "mean"), ghost_brake_persist=("ghost_brake_persist", "mean"), brake_delayed=("brake_delayed", "mean"),
                                                            missed_lead=("missed_lead", "mean"), lead_pos_err=("lead_pos_err", "mean"), n=("rmse", "size")).reset_index()
    S.to_csv(RES / "exp6B_summary.csv", index=False)
    S2 = B[B.gnss == "ramp"].groupby(["coordinated", "rate", "config"]).agg(rmse_attack=("rmse_attack", "mean"), max_err=("max_err", "mean"), frac_catastrophic=("frac_catastrophic", "mean")).reset_index()
    S2.to_csv(RES / "exp6B_degradation_curve.csv", index=False)
    Dp = D.groupby(["family", "coordinated", "detector"]).agg(pr_auc=("pr_auc", "mean"), recall=("recall", "mean"), precision=("precision", "mean"),
                                                            fp_per_hour=("fp_per_hour", "mean"), fp_events_per_hour=("fp_events_per_hour", "mean"),
                                                            episode_recall=("episode_recall", "mean"), latency_median_s=("latency_median_s", "median")).reset_index()
    Dp.to_csv(RES / "exp6C_detection_summary.csv", index=False)
    log("Part B/C done")


def main(quick=False):
    log(f"START exp6 quick={quick}")
    t0 = time.time()
    onsets = part_a()
    A = pd.read_csv(RES / "exp6A_results.csv")
    rp = A[(A.detector == "Rule:persist") & (A.set != "cleanStatic")]
    rx_stats = {"latency_median_s": float(np.nanmedian(rp.latency_median_s)) if len(rp) else 30.0,
                "fp_per_hour": float(A[(A.detector == "Rule:persist") & (A.set == "cleanStatic")].fp_per_hour.mean()) if len(A) else 2.0}
    log("receiver flag stats used in F3:", rx_stats)
    json.dump(rx_stats, open(RES / "rx_flag_stats.json", "w"))
    part_b(quick, rx_stats)
    log(f"END exp6 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    try:
        main(a.quick)
    except Exception:
        log("!! failed\n" + traceback.format_exc()); raise
