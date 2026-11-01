"""Sensor-fusion resilience simulation on nuScenes trajectories (part B).

Ego-motion ground truth comes from nuScenes v1.0-mini ego poses (LiDAR-localised,
~20 Hz).  From it we synthesise the sensors of a typical AV localisation stack
(assumption: nuScenes does not release raw GNSS/IMU with the mini split; the CAN-bus
expansion requires a login):

    IMU (20 Hz)        : forward acceleration and yaw rate + white noise + slow bias
    wheel odometry     : speed (20 Hz) + noise + 1 % scale error
    GNSS (5 Hz)        : position + N(0, 1 m) noise, optionally spoofed
    LiDAR odometry     : relative pose increments (scan matching proxy) + small noise,
                         optionally biased by a perception attack

Obstacle perception comes from nuScenes 3D annotations at key frames (2 Hz): the
simulated detector returns ego-frame detections with distance-dependent misses and
noise; attacks inject ghost obstacles, hide the lead obstacle or shift it.

Attack models
-------------
GNSS : ramp pull-off (rate r m/s, cross-track), step jump, or a TEXBAT-derived
       clock-bias profile scaled to position (ds2/ds3/ds7 shapes)
LiDAR odometry bias : translation bias per step consistent with the GNSS pull-off
       (a scan-matching perturbation) -> "coordinated" attack where GNSS and LiDAR
       odometry agree while both are wrong
Perception : ghost object ahead, hide lead object, shift lead object

Fusion / defence configurations
-------------------------------
F0 naive EKF          : all sensors fused, no gating
F1 gated EKF          : chi-square innovation gating per sensor (physics consistency)
F2 gated + DR monitor : F1 + dead-reckoning (IMU+wheel) vs GNSS and GNSS vs LiDAR-odom
                        CUSUM monitors; GNSS dropped while alarmed
F3 F2 + receiver flag : F2 + a receiver-level spoofing flag with latency/FPR drawn from
                        the TEXBAT results (part A)
F4 learned uncertainty: F1 + GBM/GRU predicting attack probability and expected
                        position error from monitor features; GNSS weight scaled by it
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUSC = ROOT / "data" / "nuscenes" / "v1.0-mini"
DT = 0.05          # 20 Hz simulation step
GNSS_EVERY = 4     # 5 Hz
LANE_ERR = 1.5     # m, lane-level failure
CATASTROPHIC = 5.0  # m


# ------------------------------------------------------------------ nuScenes
def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_scenes() -> list[dict]:
    T = {n: json.load(open(NUSC / f"{n}.json")) for n in ["scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "sample_annotation", "category", "instance"]}
    ego = {e["token"]: e for e in T["ego_pose"]}
    sensor = {s["token"]: s["channel"] for s in T["sensor"]}
    cs = {c["token"]: sensor[c["sensor_token"]] for c in T["calibrated_sensor"]}
    inst_cat = {}
    cat = {c["token"]: c["name"] for c in T["category"]}
    for i in T["instance"]:
        inst_cat[i["token"]] = cat[i["category_token"]]
    sd_by_scene: dict[str, list] = {}
    sample_scene = {s["token"]: s["scene_token"] for s in T["sample"]}
    for sd in T["sample_data"]:
        if cs[sd["calibrated_sensor_token"]] != "LIDAR_TOP":
            continue
        sd_by_scene.setdefault(sample_scene[sd["sample_token"]], []).append(sd)
    ann_by_sample: dict[str, list] = {}
    for a in T["sample_annotation"]:
        ann_by_sample.setdefault(a["sample_token"], []).append(a)
    scenes = []
    for sc in T["scene"]:
        sds = sorted(sd_by_scene[sc["token"]], key=lambda s: s["timestamp"])
        rows = []
        for sd in sds:
            e = ego[sd["ego_pose_token"]]
            rows.append((e["timestamp"] / 1e6, e["translation"][0], e["translation"][1], quat_to_yaw(e["rotation"]), sd["is_key_frame"], sd["sample_token"]))
        tr = pd.DataFrame(rows, columns=["t", "x", "y", "yaw", "key", "sample"])
        tr["t"] -= tr.t.iloc[0]
        # resample to a uniform 20 Hz grid
        tg = np.arange(0, tr.t.iloc[-1], DT)
        x = np.interp(tg, tr.t, tr.x); y = np.interp(tg, tr.t, tr.y)
        yaw = np.unwrap(tr.yaw.values); yawg = np.interp(tg, tr.t, yaw)
        # key-frame objects (global frame)
        objs = []
        for _, r in tr[tr.key].iterrows():
            for a in ann_by_sample.get(r["sample"], []):
                name = inst_cat[a["instance_token"]]
                if not (name.startswith("vehicle") or name.startswith("human")):
                    continue
                objs.append((r["t"], a["instance_token"], a["translation"][0], a["translation"][1], name))
        objs = pd.DataFrame(objs, columns=["t", "inst", "x", "y", "cls"])
        scenes.append(dict(name=sc["name"], t=tg, x=x, y=y, yaw=yawg, objs=objs, desc=sc["description"]))
    return scenes


# -------------------------------------------------------------- sensor model
@dataclass
class Sensors:
    t: np.ndarray; acc: np.ndarray; gyro: np.ndarray; wheel: np.ndarray
    gnss: np.ndarray; gnss_valid: np.ndarray; lodo: np.ndarray  # lodo: (N,3) dx,dy,dyaw in body frame
    truth: np.ndarray  # (N,3) x,y,yaw
    v_true: np.ndarray


def simulate_sensors(sc: dict, rng: np.random.Generator, gnss_sigma=1.0) -> Sensors:
    t, x, y, yaw = sc["t"], sc["x"], sc["y"], sc["yaw"]
    n = len(t)
    vx = np.gradient(x, DT); vy = np.gradient(y, DT)
    v = np.hypot(vx, vy)
    v_s = pd.Series(v).rolling(5, center=True, min_periods=1).mean().values
    acc = np.gradient(v_s, DT) + rng.normal(0, 0.15, n) + rng.normal(0, 0.02) + np.cumsum(rng.normal(0, 0.002, n))
    gyro = np.gradient(yaw, DT) + rng.normal(0, 0.01, n) + rng.normal(0, 0.002) + np.cumsum(rng.normal(0, 0.0002, n))
    wheel = v * (1 + rng.normal(0, 0.01)) + rng.normal(0, 0.05, n)
    gnss = np.stack([x, y], 1) + rng.normal(0, gnss_sigma, (n, 2))
    gnss_valid = (np.arange(n) % GNSS_EVERY) == 0
    # LiDAR odometry: body-frame increments
    dx = np.diff(x, prepend=x[0]); dy = np.diff(y, prepend=y[0]); dyaw = np.diff(yaw, prepend=yaw[0])
    c, s = np.cos(yaw), np.sin(yaw)
    bx = c * dx + s * dy; by = -s * dx + c * dy
    lodo = np.stack([bx + rng.normal(0, 0.02, n), by + rng.normal(0, 0.02, n), dyaw + rng.normal(0, 0.002, n)], 1)
    return Sensors(t, acc, gyro, wheel, gnss, gnss_valid, lodo, np.stack([x, y, yaw], 1), v)


# -------------------------------------------------------------------- attacks
def texbat_profiles() -> dict[str, np.ndarray]:
    """Clock-bias divergence (m) vs time for ds2/ds3/ds7 relative to clean, resampled to 1 s.
    Loaded from results/texbat_profiles.json if part A produced it, else synthetic ramps."""
    p = ROOT / "results" / "texbat_profiles.json"
    if p.exists():
        d = json.load(open(p))
        return {k: np.array(v) for k, v in d.items()}
    return {"ds2": np.linspace(0, 600, 300), "ds3": np.linspace(0, 10, 300), "ds7": np.linspace(0, 300, 300)}


def gnss_offset_profile(kind: str, n: int, t_on: int, rng: np.random.Generator, yaw: np.ndarray, rate: float | None = None, profiles=None) -> np.ndarray:
    """(n,2) additive GNSS position offset in the global frame."""
    off = np.zeros((n, 2))
    if kind == "none" or t_on >= n:
        return off
    steps = np.arange(n - t_on) * DT
    # cross-track direction (left of heading) with random sign
    sgn = rng.choice([-1, 1])
    if kind == "ramp":
        r = rate if rate is not None else rng.choice([0.25, 0.5, 1.0, 2.0])
        mag = np.minimum(r * steps, 40.0)
    elif kind == "jump":
        mag = np.full(len(steps), rate if rate is not None else rng.choice([5.0, 10.0, 20.0]))
    elif kind.startswith("texbat_"):
        prof = profiles[kind.split("_", 1)[1]]
        # scale: TEXBAT clock-bias divergence (m) -> position pull-off of at most 30 m over the attack
        tp = np.arange(len(prof)) * 1.0
        m = np.interp(steps, tp, prof, right=prof[-1]) if len(prof) > 1 else np.zeros(len(steps))
        scale = 30.0 / max(np.abs(prof).max(), 1e-6)
        mag = np.abs(m) * scale
    else:
        raise ValueError(kind)
    for i, k in enumerate(range(t_on, n)):
        d = np.array([-np.sin(yaw[k]), np.cos(yaw[k])]) * sgn
        off[k] = d * mag[i]
    return off


def apply_lodo_bias(lodo: np.ndarray, off: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Coordinated perception attack: bias LiDAR odometry increments so that integrated LiDAR
    odometry follows the same pull-off as the spoofed GNSS (global-frame offset increments rotated
    into the body frame)."""
    doff = np.diff(off, axis=0, prepend=off[:1])
    c, s = np.cos(yaw), np.sin(yaw)
    bx = c * doff[:, 0] + s * doff[:, 1]; by = -s * doff[:, 0] + c * doff[:, 1]
    out = lodo.copy(); out[:, 0] += bx; out[:, 1] += by
    return out


# --------------------------------------------------------------------- EKF
class EKF:
    """State [x, y, yaw, v]; predict with IMU (acc, gyro); measurements: wheel speed, GNSS xy,
    LiDAR odometry increments (as body-frame velocity)."""

    def __init__(self, x0, gate: bool, v0: float = 0.0, chi2_gnss=13.82, chi2_odo=10.83, warmup: int = 40):
        self.x = np.array([x0[0], x0[1], x0[2], v0]); self.P = np.diag([1.0, 1.0, 0.05, 1.0])
        # process noise per 50 ms step: model error of the constant-heading/velocity kinematics
        self.Q = np.diag([0.20, 0.20, 0.01, 0.5]) ** 2
        self.gate = gate; self.chi2_gnss = chi2_gnss; self.chi2_odo = chi2_odo
        self.last_nis = {"gnss": 0.0, "wheel": 0.0, "lodo": 0.0}
        self.gnss_rejected = 0; self.k = 0; self.warmup = warmup

    def predict(self, acc, gyro):
        x, y, th, v = self.x
        self.x = np.array([x + v * np.cos(th) * DT, y + v * np.sin(th) * DT, th + gyro * DT, v + acc * DT])
        F = np.array([[1, 0, -v * np.sin(th) * DT, np.cos(th) * DT], [0, 1, v * np.cos(th) * DT, np.sin(th) * DT], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Q
        self.k += 1

    def _update(self, z, h, H, R, key, chi2):
        y = z - h; S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y)); self.last_nis[key] = nis
        if self.gate and self.k > self.warmup and nis > chi2:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(4) - K @ H) @ self.P
        return True

    def update_wheel(self, v, sigma=0.1):
        return self._update(np.array([v]), np.array([self.x[3]]), np.array([[0, 0, 0, 1.0]]), np.array([[sigma ** 2]]), "wheel", 6.63)

    def update_gnss(self, xy, sigma=1.0, weight=1.0):
        R = np.eye(2) * (sigma ** 2) / max(weight, 1e-3)
        ok = self._update(xy, self.x[:2], np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]]), R, "gnss", self.chi2_gnss)
        if not ok:
            self.gnss_rejected += 1
        return ok

    def update_lodo(self, inc, sigma=0.05):
        # body-frame increment -> speed and yaw-rate observation: v = bx/DT ; yaw += dyaw (direct)
        v_obs = inc[0] / DT
        ok = self._update(np.array([v_obs]), np.array([self.x[3]]), np.array([[0, 0, 0, 1.0]]), np.array([[(sigma / DT) ** 2]]), "lodo", self.chi2_odo)
        # heading increment as a soft measurement of yaw
        th_obs = self.x[2] + inc[2] - 0.0
        return ok


class CUSUM:
    def __init__(self, k=0.5, h=8.0):
        self.k, self.h, self.s = k, h, 0.0

    def step(self, z):
        self.s = max(0.0, self.s + z - self.k)
        fired = self.s > self.h
        if fired:
            self.s = 0.0
        return fired, self.s


# ------------------------------------------------------------------ episode
def run_episode(sens: Sensors, gnss_off: np.ndarray, lodo_used: np.ndarray, config: str, learned=None, rx_flag: np.ndarray | None = None,
                rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Run the fusion stack under one configuration; returns per-step log."""
    n = len(sens.t)
    gate = config != "F0"
    ekf = EKF(sens.truth[0], gate=gate, v0=float(sens.wheel[0]))
    # dead-reckoning integrator (IMU + wheel) reset to fused estimate every 10 s
    dr = sens.truth[0, :2].copy(); dr_th = sens.truth[0, 2]
    # LiDAR-odometry integrator
    lo = sens.truth[0, :2].copy(); lo_th = sens.truth[0, 2]
    cus_dr = CUSUM(k=0.5, h=10.0); cus_lo = CUSUM(k=0.5, h=10.0)
    alarm_dr = 0; alarm_lo = 0; drop_until = -1; alarm_until = -1
    p_ema = 0.0
    logs = []
    lastreset = 0; dist_since_reset = 0.0
    for k in range(n):
        ekf.predict(sens.acc[k], sens.gyro[k])
        ekf.update_wheel(sens.wheel[k])
        ekf.update_lodo(lodo_used[k])
        # integrators
        dr_th += sens.gyro[k] * DT; dr += sens.wheel[k] * DT * np.array([np.cos(dr_th), np.sin(dr_th)])
        lo_th += lodo_used[k, 2]; lo += np.array([np.cos(lo_th) * lodo_used[k, 0] - np.sin(lo_th) * lodo_used[k, 1], np.sin(lo_th) * lodo_used[k, 0] + np.cos(lo_th) * lodo_used[k, 1]])
        dist_since_reset += abs(sens.wheel[k]) * DT
        gnss_used = False; nis_g = np.nan; d_dr = np.nan; d_lo = np.nan; unc = np.nan; p_att = 0.0
        if sens.gnss_valid[k]:
            z = sens.gnss[k] + gnss_off[k]
            d_dr = float(np.linalg.norm(z - dr)); d_lo = float(np.linalg.norm(z - lo))
            # monitors (F2+): disagreement in excess of the expected dead-reckoning drift
            # (GNSS noise ~1 m + 3 % of distance travelled since re-anchoring for IMU/wheel, 1 % for LiDAR odometry)
            exp_dr = 2.0 + 0.03 * dist_since_reset; exp_lo = 2.0 + 0.01 * dist_since_reset
            f_dr, s_dr = cus_dr.step((d_dr - exp_dr) / 1.0); f_lo, s_lo = cus_lo.step((d_lo - exp_lo) / 1.0)
            if f_dr or f_lo:
                alarm_until = k + 20 * 5  # monitor alarm considered active for 5 s (used as detector score)
                alarm_dr += int(f_dr); alarm_lo += int(f_lo)
            if config in ("F2", "F3") and (f_dr or f_lo):
                drop_until = k + 20 * 5  # drop GNSS for 5 s after an alarm
            if config == "F3" and rx_flag is not None and rx_flag[k]:
                drop_until = max(drop_until, k + 20 * 5)
            weight = 1.0
            if config == "F4" and learned is not None:
                feat = np.array([[d_dr, d_lo, ekf.last_nis["gnss"], ekf.last_nis["lodo"], ekf.last_nis["wheel"], s_dr, s_lo, np.sqrt(ekf.P[0, 0] + ekf.P[1, 1]), ekf.x[3]]])
                p_now = float(learned["clf"].predict_proba(feat)[0, 1]); unc = float(learned["reg"].predict(feat)[0])
                p_ema = 0.7 * p_ema + 0.3 * p_now; p_att = p_ema
                # GNSS down-weighting: quadratic in attack probability, further scaled by the predicted error
                weight = float(np.clip((1.0 - p_att) ** 2, 0.01, 1.0)) / (1.0 + max(unc, 0.0))
                if p_att > 0.8:
                    drop_until = k + 20 * 2
            if k >= drop_until:
                gnss_used = ekf.update_gnss(z, weight=weight)
            nis_g = ekf.last_nis["gnss"]
        # periodic re-anchoring of integrators to the fused estimate (every 10 s) to bound their drift
        if k - lastreset >= 200:
            dr = ekf.x[:2].copy(); dr_th = ekf.x[2]; lo = ekf.x[:2].copy(); lo_th = ekf.x[2]; lastreset = k; dist_since_reset = 0.0
        err = float(np.linalg.norm(ekf.x[:2] - sens.truth[k, :2]))
        logs.append((sens.t[k], err, gnss_used, nis_g, d_dr, d_lo, cus_dr.s, cus_lo.s, ekf.last_nis["lodo"], ekf.last_nis["wheel"], np.sqrt(ekf.P[0, 0] + ekf.P[1, 1]), ekf.x[3], p_att, unc, ekf.x[0], ekf.x[1], float(k < alarm_until)))
    L = pd.DataFrame(logs, columns=["t", "err", "gnss_used", "nis_gnss", "d_dr", "d_lo", "cus_dr", "cus_lo", "nis_lodo", "nis_wheel", "P_pos", "v_est", "p_att", "unc", "x_est", "y_est", "alarm_active"])
    L["alarm_dr"] = alarm_dr; L["alarm_lo"] = alarm_lo
    return L


# ------------------------------------------------------------ perception
def perception_outcomes(sc: dict, L: pd.DataFrame, sens: Sensors, attack: str, t_on: float, rng: np.random.Generator, gnss_off=None) -> dict:
    """Simulated object perception at key frames (2 Hz) transformed into the global frame with the
    *estimated* ego pose; returns safety-relevant outcomes under the perception attack."""
    objs = sc["objs"]
    if len(objs) == 0:
        return dict(ghost_brake=0, missed_lead=0, lead_pos_err=np.nan, n_frames=0)
    est = L[["t", "x_est", "y_est"]].values
    truth = sens.truth
    ghost_brake = 0; missed = 0; lead_errs = []; n_frames = 0
    ghost_brake_persist = 0; brake_delayed = 0
    prev_det = None  # ego-frame detections of the previous key frame (for the persistence check)
    for tk, g in objs.groupby("t"):
        k = int(min(np.searchsorted(sens.t, tk), len(sens.t) - 1))
        ex, ey, eth = truth[k]
        # ego-frame detections with distance-dependent misses
        rel = np.stack([g.x.values - ex, g.y.values - ey], 1)
        c, s = np.cos(eth), np.sin(eth)
        body = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], 1)
        d = np.hypot(body[:, 0], body[:, 1])
        keep = rng.random(len(d)) > (0.05 + 0.3 * (d / 50.0) ** 2)
        det = body[keep] + rng.normal(0, 0.3, (keep.sum(), 2))
        attacked = attack != "none" and tk >= t_on
        ahead = (det[:, 0] > 0) & (np.abs(det[:, 1]) < 1.75)
        lead_idx = np.argmin(np.where(ahead, det[:, 0], np.inf)) if ahead.any() else None
        if attacked and attack in ("ghost", "coord_ghost"):
            det = np.vstack([det, [[rng.uniform(8, 15), rng.uniform(-0.5, 0.5)]]])
        if attacked and attack in ("hide", "coord_hide") and lead_idx is not None:
            det = np.delete(det, lead_idx, axis=0)
        if attacked and attack in ("shift",) and lead_idx is not None:
            det[lead_idx] += [rng.uniform(-4, 4), rng.uniform(-2, 2)]
        # safety logic on ego-frame detections (independent of localisation)
        ahead2 = (det[:, 0] > 0) & (np.abs(det[:, 1]) < 1.75) & (det[:, 0] < 15)
        v = sens.v_true[k]
        real_ahead = (body[:, 0] > 0) & (np.abs(body[:, 1]) < 1.75) & (body[:, 0] < 15)
        real_trigger = real_ahead.any() and (body[real_ahead, 0].min() / max(v, 0.1)) < 2.0
        if ahead2.any() and v > 2.0 and (det[ahead2, 0].min() / max(v, 0.1)) < 2.0:
            # emergency brake trigger; count as ghost-induced if no real object satisfies it
            if not real_trigger:
                ghost_brake += 1
            # persistence check (perception-side defence): the triggering object must have been
            # observed in the previous key frame within 6 m of where it is now (in the ego frame,
            # after removing ego motion ~ v*0.5 s along x)
            trig = det[ahead2][np.argmin(det[ahead2, 0])]
            persistent = False
            if prev_det is not None and len(prev_det):
                pred_prev = trig + np.array([v * 0.5, 0.0])  # where it should have been half a second ago
                persistent = bool(np.min(np.hypot(prev_det[:, 0] - pred_prev[0], prev_det[:, 1] - pred_prev[1])) < 6.0)
            if persistent:
                if not real_trigger:
                    ghost_brake_persist += 1
            elif real_trigger:
                brake_delayed += 1  # a genuine trigger delayed by one frame by the persistence check
        prev_det = det.copy()
        real_lead = (body[:, 0] > 0) & (np.abs(body[:, 1]) < 1.75) & (body[:, 0] < 20)
        if real_lead.any():
            det_lead = (det[:, 0] > 0) & (np.abs(det[:, 1]) < 1.75) & (det[:, 0] < 20)
            if not det_lead.any():
                missed += 1
        # global-frame lead position error caused by localisation error (map-relative planning)
        if real_lead.any():
            i = np.argmin(np.where(real_lead, body[:, 0], np.inf))
            ex_e, ey_e = est[k, 1], est[k, 2]
            glob_est = np.array([ex_e + c * body[i, 0] - s * body[i, 1], ey_e + s * body[i, 0] + c * body[i, 1]])
            glob_true = np.array([g.x.values[i], g.y.values[i]])
            lead_errs.append(float(np.linalg.norm(glob_est - glob_true)))
        n_frames += 1
    return dict(ghost_brake=ghost_brake, ghost_brake_persist=ghost_brake_persist, brake_delayed=brake_delayed, missed_lead=missed,
                lead_pos_err=float(np.mean(lead_errs)) if lead_errs else np.nan, n_frames=n_frames)
