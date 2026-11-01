# Results

These are the result files of the run reported in the article, produced by `code/run_exp6.py` on
13 September 2026 (full mode, about 16 minutes on a laptop CPU). `tables/` holds the article's
Tables 1–3 as CSV files and `figures/` holds its Figures 1–4 as PNG images.

Licence: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Please cite the release (see
`CITATION.cff` in the repository root).

No TEXBAT or nuScenes data are contained in these files. They hold detector scores, error statistics
and settings computed from those datasets.

## The article's tables and figures

Each table and figure is a view of one of the result files listed under *Files* below:

| Article | File | Underlying result file |
|---|---|---|
| Table 1 | `tables/table1_texbat_detection.csv` | `exp6A_results.csv` |
| Table 2 | `tables/table2_localisation_error.csv` | `exp6B_summary.csv` |
| Table 3 | `tables/table3_attack_detection.csv` | `exp6C_detection_summary.csv` |
| Figure 1 | `figures/fig3_texbat_prauc.png` | `exp6A_results.csv` |
| Figure 2 | `figures/fig1_degradation.png` | `exp6B_degradation_curve.csv` |
| Figure 3 | `figures/fig2_catastrophic.png` | `exp6B_summary.csv` |
| Figure 4 | `figures/fig4_detection.png` | `exp6C_detection_summary.csv` |

The figure file names do not follow the article's numbering; use the mapping above.

## Files

### Part A: receiver-level spoofing detection on TEXBAT

**`exp6A_results.csv`**: one row per scenario (`set`: cleanStatic, ds2, ds3, ds7) and detector
(57 rows). Detectors are the individual monitors `Rule:<feature>`, their maximum `Rule:max`, the
three-epoch persistent combination `Rule:persist`, `IsolationForest`, `LightGBM(LOSO)` (trained
leave-one-scenario-out) and the ensembles `Ens(Rule|GBM)` and `Ens(rank-mean)`. The cleanStatic rows
are scored on the 40 % of the clean recording held out for calibration. Columns:

| Column | Meaning |
|---|---|
| `n`, `n_attack` | scored epochs (5 Hz), of which under attack; the 15 s guard band around the onset is excluded |
| `pr_auc`, `roc_auc` | area under the precision–recall and ROC curves of the detector score |
| `f1_max`, `thr_f1max` | best F1 over all thresholds and the threshold that achieves it |
| `recall@fpr0.0001` | recall at a benign false-positive rate of 10⁻⁴ |
| `tp`, `fp`, `fn`, `tn` | confusion counts at the detector's calibrated operating point |
| `precision`, `recall`, `f1`, `fpr` | metrics at the calibrated operating point |
| `fp_per_hour` | false-positive epochs per hour of benign recording |
| `n_episodes`, `episode_recall` | number of attack episodes and fraction detected |
| `latency_median_s`, `latency_p95_s` | detection delay, median and 95th percentile, measured from the first scored attack epoch; because the guard band is excluded, that epoch lies 15 s after `onset_s` |
| `part`, `set`, `detector` | identifiers |
| `onset_s` | attack onset in receiver time used for labelling |

**`texbat_onsets.json`**: attack onsets in receiver time (s) for ds2, ds3 and ds7, the end of the
part of ds7 that is identical to cleanStatic, and the guard band (15 s).

**`rx_flag_stats.json`**: median latency (s) and false alarms per hour of the persistent monitor on
TEXBAT. Part B uses them to simulate the receiver-level spoofing flag of configuration F3.

### Parts B and C: fusion resilience and attack detection on nuScenes trajectories

Four held-out scenes (`scene-0916`, `scene-1077`, `scene-1094`, `scene-1100`) × 20 attack
specifications × 6 random repetitions = 480 test episodes. The other six mini scenes trained the
learned components.

Shared identifier columns:

| Column | Meaning |
|---|---|
| `scene` | nuScenes scene name |
| `family` | `none`, `gnss` (spoofing only), `perception` (perception attack or LiDAR-odometry bias only), `coordinated` |
| `gnss` | GNSS attack: `none`, `ramp` (cross-track pull-off), `jump` (step), `texbat_ds2`/`ds3`/`ds7` (TEXBAT-shaped profile) |
| `perception` | perception attack: `none`, `ghost`, `hide`, `shift` |
| `coordinated` | `True` when the LiDAR-odometry increments are biased; with `gnss = none` this is the LiDAR-odometry-bias-only attack |
| `rate` | ramp rate (m/s) of a GNSS pull-off or of the LiDAR-odometry bias, or step size (m) of a jump; empty for TEXBAT-shaped profiles, perception-only attacks and no-attack episodes |
| `rep` | repetition index (0–5) |

**`exp6B_results.csv`**: one row per episode and fusion configuration (`config`: F0 naive EKF, F1
gated EKF, F2 gated plus dead-reckoning and LiDAR-odometry CUSUM monitors, F3 F2 plus receiver flag,
F4 learned uncertainty); 2,400 rows.

| Column | Meaning |
|---|---|
| `rmse`, `rmse_attack` | position RMSE over the whole episode and from attack onset (m) |
| `max_err` | maximum position error over the whole episode (m) |
| `frac_lane`, `frac_catastrophic` | fraction of the whole episode, not only of the attack period, with position error above 1.5 m and above 5 m |
| `gnss_used_frac` | fraction of GNSS fixes accepted by the filter |
| `alarms` | number of CUSUM monitor alarms in the episode |
| `ghost_brake` | emergency-brake triggers with no real object satisfying the same condition |
| `ghost_brake_persist` | ghost-induced triggers that survive the one-frame persistence check |
| `brake_delayed` | genuine triggers delayed by the persistence check |
| `missed_lead` | key frames in which a real lead vehicle within 20 m is not detected |
| `lead_pos_err` | mean global-frame position error of the lead vehicle caused by the ego-pose error (m) |
| `n_frames` | number of key frames (2 Hz) containing at least one annotated vehicle or person |

**`exp6B_summary.csv`**: means of the above per `family`, `coordinated` and `config`; `n` is the number
of episodes averaged. **`exp6B_degradation_curve.csv`**: mean `rmse_attack`, `max_err` and
`frac_catastrophic` per ramp `rate`, `coordinated` and `config`.

**`exp6C_detection.csv`**: one row per episode and detector (3,360 rows). Detectors run on the monitor
features of configuration F1: `Monitor:NIS-GNSS`, `Monitor:CUSUM(DR|LO)`, `RxFlag(TEXBAT-derived)`,
`GBM` (LightGBM), `GRU`, `Ens(GBM+GRU)` and `Ens(GBM+GRU|CUSUM)`. Metric columns are as in
`exp6A_results.csv`, plus `fp_events_per_hour` (rising edges of the alarm in benign time) and
`threshold` (the operating threshold). The positive class is "GNSS spoofing or LiDAR-odometry bias
active"; perception-only episodes contain no positives, so they contribute false alarms only.

**`exp6C_detection_summary.csv`**: means per `family`, `coordinated` and `detector` (median for
`latency_median_s`). **`exp6C_thresholds.json`**: detector thresholds calibrated on benign training
episodes at 10 false-alarm steps per hour.

### Log

**`run.log`**: timestamped log of the runs of 13 September 2026: two quick development runs and the
full run (15:21–15:37) that produced these files.
