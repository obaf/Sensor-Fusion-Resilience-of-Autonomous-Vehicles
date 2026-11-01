"""Receiver-level GNSS spoofing detection on TEXBAT processed observables (part A).

Data: UT Austin TEXBAT processed outputs of the GRID software receiver for
  cleanStatic (authentic), ds2 (static overpowered time push), ds3 (static
  matched-power time push), ds7 (matched-power, carrier-phase-aligned time push).
Files per set: channel.mat (per-channel pseudorange, Doppler, carrier phase, C/N0
  at 5 Hz), navsol.mat (position, clock bias/rate, 5 Hz), symmdiff.mat (symmetric
  difference of the correlation function, i.e. correlation-distortion monitor).

Epoch features (5 Hz) computed from the observables:
  cn0_mean, cn0_std, cn0_jump (mean |ΔC/N0| across channels), n_ch (channels tracked),
  clk_rate (receiver clock drift in m/s, from navsol deltR differences),
  clk_accel (second difference of clock bias),
  pos_jump (|Δposition| between epochs, m), vel_norm (m/s),
  ccd_mean (code-carrier divergence: d/dt(pseudorange) + λ d/dt(carrier phase)),
  dopp_resid (Doppler vs carrier-phase rate consistency), symmdiff_rms.

Labels: spoof-free lead-in per TEXBAT documentation (every scenario begins clean); the
spoofer switches on at ~ +100 s (ds2/ds3) and ~ +110 s (ds7) after start — taken from
the receiver clock-bias divergence relative to the clean run (see run log, derived
onsets are stored in results). Windows before onset are labelled benign.

Detectors: non-AI (per-feature bounds calibrated on cleanStatic + C/N0 & clock-drift
monitors + correlation distortion) and AI (IsolationForest, Autoencoder-free GBM
trained on clean vs spoofed epochs of *other* scenarios, leave-one-scenario-out).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "texbat"
sys.path.insert(0, str(ROOT.parent / "experiment4" / "code"))
from evaluate import evaluate  # noqa: E402

L1_WAVELENGTH = 0.190293672798365  # m
SETS = ["cleanStatic", "ds2", "ds3", "ds7"]


def load_set(name: str) -> dict:
    ch = sio.loadmat(DATA / name / "channel.mat")["channel"].T
    nav = sio.loadmat(DATA / name / "navsol.mat")["navsol"].T
    out = {"channel": pd.DataFrame(ch, columns=["rrt_wk", "rrt", "ort_wk", "ort_s", "ort_f", "dopp", "phase", "pr", "cn0", "valid", "err", "status", "sigtype", "txid"]),
           "navsol": pd.DataFrame(nav, columns=["wk", "s", "f", "x", "y", "z", "dclk", "vx", "vy", "vz", "dclk_rate", "flag"])}
    out["channel"]["t"] = out["channel"].ort_s + out["channel"].ort_f
    out["navsol"]["t"] = out["navsol"].s + out["navsol"].f
    p = DATA / name / "symmdiff.mat"
    if p.exists():
        try:
            sd = sio.loadmat(p)
            key = [k for k in sd if not k.startswith("__")][0]
            out["symmdiff"] = sd[key].T
        except Exception:
            pass
    return out


def epoch_features(d: dict) -> pd.DataFrame:
    ch = d["channel"]; nav = d["navsol"]
    ch = ch[(ch.valid == 1) & (ch.status >= 5) & (ch.ort_wk < 9999)].copy()
    ch["tr"] = np.round(ch.t * 5) / 5  # 0.2 s grid
    rows = []
    # per-channel derivatives
    ch = ch.sort_values(["txid", "t"])
    ch["d_pr"] = ch.groupby("txid").pr.diff() / ch.groupby("txid").t.diff()
    ch["d_phase"] = ch.groupby("txid").phase.diff() / ch.groupby("txid").t.diff()
    ch["ccd"] = ch.d_pr - L1_WAVELENGTH * ch.d_phase          # code-carrier divergence rate (m/s); phase changes in the same sense as pseudorange in channel.mat
    ch["dopp_resid"] = ch.dopp + ch.d_phase                     # Doppler vs phase-rate (Hz), phase changes opposite sense
    ch["d_cn0"] = ch.groupby("txid").cn0.diff().abs()
    g = ch.groupby("tr")
    f = pd.DataFrame({
        "t": g.t.mean(), "n_ch": g.txid.nunique(), "cn0_mean": g.cn0.mean(), "cn0_std": g.cn0.std().fillna(0),
        "cn0_jump": g.d_cn0.mean().fillna(0), "ccd_mean": g.ccd.apply(lambda s: np.nanmean(np.abs(s))).fillna(0),
        "ccd_max": g.ccd.apply(lambda s: np.nanmax(np.abs(s)) if s.notna().any() else 0).fillna(0),
        "dopp_resid": g.dopp_resid.apply(lambda s: np.nanmean(np.abs(s))).fillna(0),
        "spoof_flag_rx": g.err.apply(lambda s: float(((s.astype(int) & 2) > 0).mean())),
    }).reset_index(drop=True)
    # navigation-solution features on the same grid
    nav = nav.sort_values("t").copy()
    nav["tr"] = np.round(nav.t * 5) / 5
    nav["clk_rate"] = nav.dclk.diff() / nav.t.diff()
    nav["clk_accel"] = nav.clk_rate.diff() / nav.t.diff()
    nav["pos_jump"] = np.sqrt(nav.x.diff() ** 2 + nav.y.diff() ** 2 + nav.z.diff() ** 2)
    nav["vel_norm"] = np.sqrt(nav.vx ** 2 + nav.vy ** 2 + nav.vz ** 2)
    nv = nav.groupby("tr")[["clk_rate", "clk_accel", "pos_jump", "vel_norm", "dclk_rate", "dclk"]].mean()
    f = f.merge(nv, left_on=np.round(f.t * 5) / 5, right_index=True, how="left").drop(columns=["key_0"], errors="ignore")
    if "symmdiff" in d:
        # symmdiff.mat: 24 columns = [rrt_week, rrt_s, channel_idx, txid, 20 complex taps of the
        # symmetric-difference correlation function] at 100 Hz.  Correlation-distortion monitor:
        # per-channel RMS magnitude of the symmetric difference (normalised later by the clean set).
        sd = d["symmdiff"]
        try:
            rrt = np.real(sd[:, 1]); txid = np.real(sd[:, 3]).astype(int)
            mag = np.sqrt(np.mean(np.abs(sd[:, 4:]) ** 2, axis=1))
            # RRT -> ORT offset from the channel table (ORT = RRT + tOffset)
            chv = d["channel"]; chv = chv[chv.ort_wk < 9999]
            toff = float(np.median((chv.ort_s + chv.ort_f - chv.rrt).values))
            t_ort = rrt + toff
            s = pd.DataFrame({"tr": np.round(t_ort * 5) / 5, "m": mag}).groupby("tr").m.mean()
            f["symmdiff_rms"] = np.round(f.t * 5) / 5
            f["symmdiff_rms"] = f["symmdiff_rms"].map(s)
        except Exception:
            f["symmdiff_rms"] = np.nan
    f = f.fillna(0.0)
    f["t_rel"] = f.t - f.t.min()
    return f


FEATS = ["n_ch", "cn0_mean", "cn0_std", "cn0_jump", "ccd_mean", "ccd_max", "dopp_resid", "clk_rate", "clk_accel", "pos_jump", "vel_norm", "dclk_rate"]


def clock_divergence(f_spoof: pd.DataFrame, f_clean: pd.DataFrame, grid_dt: float = 1.0):
    """Receiver clock bias of the spoofed run minus the clean run (same receiver, same site),
    both interpolated to a common grid and zeroed on the first 30 s. Returns (t, diff_m)."""
    tmax = min(f_spoof.t_rel.max(), f_clean.t_rel.max())
    t = np.arange(0, tmax, grid_dt)
    s = np.interp(t, f_spoof.t_rel.values, f_spoof.dclk.values)
    c = np.interp(t, f_clean.t_rel.values, f_clean.dclk.values)
    d = s - c
    d = d - np.median(d[: max(int(30 / grid_dt), 1)])
    return t, d


def identical_prefix_end(d_spoof: dict, d_clean: dict, tol_m: float = 0.5) -> float:
    """For scenarios digitally built on cleanStatic (ds7), the receiver observables are identical
    until the spoofer is switched on.  Returns the receiver-time (t_rel, s) at which the per-channel
    pseudoranges of the two runs first differ by more than tol_m for any satellite."""
    a = d_spoof["channel"]; b = d_clean["channel"]
    a = a[(a.valid == 1) & (a.ort_wk < 9999)]; b = b[(b.valid == 1) & (b.ort_wk < 9999)]
    t0 = min(a.t.min(), b.t.min())
    m = a.merge(b, on=["txid", "ort_s"], suffixes=("_s", "_c"))
    m = m[np.abs(m.ort_f_s - m.ort_f_c) < 1e-3]
    diff = np.abs(m.pr_s - m.pr_c)
    bad = m[diff > tol_m]
    if len(bad) == 0:
        return np.nan
    return float(bad.t_s.min() - t0)


def derive_onset(f_spoof: pd.DataFrame, f_clean: pd.DataFrame, z_thresh: float = 4.0, min_onset: float = 60.0) -> float:
    """Data-driven spoofing onset (s after start).

    TEXBAT documentation states that every scenario begins spoof-free and that the spoofer is
    switched on roughly 1.5-2 minutes into the recording.  We locate the onset as the first time
    (>= min_onset) at which a 5 s rolling median of the maximum z-score of three spoofing-sensitive
    statistics — received power (C/N0), correlation distortion (symmetric-difference RMS) and the
    clock-bias divergence from the clean run — exceeds z_thresh relative to the clean baseline.
    """
    t, d = clock_divergence(f_spoof, f_clean)
    zs = []
    for col in ["cn0_mean", "symmdiff_rms"]:
        if col in f_spoof and col in f_clean:
            mu = f_clean[col].mean(); sd = f_clean[col].std() + 1e-9
            v = np.interp(t, f_spoof.t_rel.values, f_spoof[col].values)
            zs.append(np.abs((v - mu) / sd))
    sd_c = np.std(np.diff(np.interp(t, f_clean.t_rel.values, f_clean.dclk.values))) + 1e-9
    zs.append(np.abs(d) / (10 * sd_c))
    z = np.max(np.stack(zs), axis=0)
    zr = pd.Series(z).rolling(5, min_periods=1).median().values
    cand = np.flatnonzero((zr > z_thresh) & (t >= min_onset))
    if len(cand) == 0:
        return 100.0
    return float(t[cand[0]])


class GNSSRuleMonitor:
    """Non-AI receiver monitors calibrated on clean data (99.9th percentile bounds)."""

    def __init__(self, q=0.999):
        self.q = q; self.bounds = {}

    def fit(self, f_clean: pd.DataFrame):
        for c in ["cn0_jump", "ccd_mean", "ccd_max", "dopp_resid", "pos_jump", "clk_accel", "symmdiff_rms"]:
            if c in f_clean:
                v = np.abs(f_clean[c].values); self.bounds[c] = float(np.quantile(v, self.q)) + 1e-9
        # clock drift monitor: authentic receiver clock drift rate is ~constant; bound its deviation
        cr = f_clean.clk_rate.values; self.clk_mu = float(np.median(cr)); self.clk_sig = float(np.std(cr) + 1e-9)
        self.cn0_hi = float(np.quantile(f_clean.cn0_mean, self.q)); self.cn0_lo = float(np.quantile(f_clean.cn0_mean, 1 - self.q))
        return self

    def score(self, f: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=f.index)
        for c, b in self.bounds.items():
            if c in f:
                out[f"mag_{c}"] = np.abs(f[c].values) / b
        out["mag_clk_drift"] = np.abs(f.clk_rate.values - self.clk_mu) / (6 * self.clk_sig)
        span = max(self.cn0_hi - self.cn0_lo, 0.5)
        out["mag_cn0_level"] = np.maximum(f.cn0_mean.values - self.cn0_hi, self.cn0_lo - f.cn0_mean.values).clip(min=0) / (0.5 * span) + (np.maximum(f.cn0_mean.values - self.cn0_hi, self.cn0_lo - f.cn0_mean.values) > 0)
        out["rule_max"] = out.max(axis=1)
        # persistence: require 3 consecutive epochs (0.6 s) above bound to reduce single-epoch glitches
        out["rule_persist"] = pd.Series(out.rule_max.values).rolling(3, min_periods=1).min().values
        return out
