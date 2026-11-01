# Graceful or Catastrophic? Sensor-Fusion Resilience of Autonomous-Vehicle Localisation under Coordinated Global Navigation Satellite System Spoofing and Perception Attacks

Code and result files for the article of the same title by Akinwole Obafemi (Edinburgh Napier
University, ORCID [0009-0003-6875-1900](https://orcid.org/0009-0003-6875-1900)), submitted to
*IET Radar, Sonar & Navigation* (Wiley). Everything needed to re-run the experiments is here except the
two third-party input datasets, which must be downloaded from their providers as explained below.

> Article DOI: to follow on publication.

## 1. Introduction

Autonomous vehicles localise themselves by fusing several sensors, among them a Global Navigation
Satellite System (GNSS) receiver, an inertial measurement unit (IMU), wheel-speed sensors and LiDAR
odometry. Civil GNSS signals are unauthenticated and can be spoofed, and the vehicle's perception
system (the object detector) can be attacked as well. Almost all published defences assume that only
one of these sensing modalities is compromised at a time: GNSS defences treat the inertial, wheel and
LiDAR-derived motion as a trustworthy reference, and perception defences assume that the vehicle's own
pose is correct. This study asks what happens when an adversary violates both assumptions at once,
and whether the fused localisation then degrades gracefully or fails catastrophically.

The experiment has three parts, all driven by `code/run_exp6.py`:

- **Part A – receiver-level spoofing detection on real recordings.** Physics-based monitors
  (received power, correlation distortion, code-carrier divergence, clock drift), an Isolation Forest
  and a LightGBM classifier are benchmarked on identical inputs from the Texas Spoofing Test Battery
  (TEXBAT): the authentic recording `cleanStatic` and the spoofing scenarios `ds2` (overpowered time
  push), `ds3` (matched-power time push) and `ds7` (matched-power, carrier-phase-aligned time push).
- **Part B – fusion resilience on real driving trajectories.** The ten nuScenes v1.0-mini trajectories
  are replayed through a synthetic IMU, wheel-speed, GNSS and LiDAR-odometry sensor suite and an
  extended Kalman filter (EKF). GNSS pull-off profiles, a coordinated LiDAR-odometry bias and
  object-level perception attacks (ghost obstacle, hidden lead vehicle, shifted lead vehicle) are
  injected, and five fusion configurations are compared on 480 held-out attack episodes:
  F0 naive EKF, F1 chi-square gating, F2 gating plus dead-reckoning and LiDAR-odometry CUSUM monitors,
  F3 as F2 plus a receiver-level spoofing flag with TEXBAT-derived latency, and F4 a learned
  uncertainty model (LightGBM) that softly down-weights GNSS.
- **Part C – attack detection from the fusion monitors.** Seven detectors (NIS gate, CUSUM monitors,
  receiver flag, LightGBM, a GRU sequence model and two ensembles) are trained on six scenes and
  evaluated on the four held-out scenes at a common false-alarm budget.

The repository holds the code, the exact result files behind the article's three tables and four
figures, and a data dictionary for every result file. The datasets themselves are not redistributed (their terms do not allow it), and nothing in the
repository contains TEXBAT or nuScenes data; the result files hold only detector scores, error
statistics and settings computed from them.

## 2. Contents

| Path | Contents |
|---|---|
| `code/run_exp6.py` | Runs Parts A, B and C and writes all result files to `results/`; unchanged from the version that produced the deposited results |
| `code/run_repeatable.py` | Wrapper that fixes every random source and then runs `run_exp6.py`, so that repeated runs give identical files |
| `code/gnss_texbat.py` | TEXBAT loading, per-epoch features, onset labelling and the physics-based monitors (Part A) |
| `code/fusion_sim.py` | Trajectory loading, sensor synthesis, attack models, the EKF with configurations F0 to F4, and the perception outcomes (Part B) |
| `code/evaluate.py` | Detection metrics shared by all parts: PR-AUC, recall at a false-alarm budget, episode recall and latency |
| `code/download_texbat.py` | Downloads the twelve TEXBAT files used by Part A from the University of Texas at Austin |
| `data/README.md` | Where the two input datasets come from and the folder layout the code expects |
| `results/` | The result files behind the article, with a column-by-column data dictionary in `results/README.md` |
| `results/tables/`, `results/figures/` | The article's Tables 1 to 3 (CSV) and Figures 1 to 4 (PNG) |
| `requirements.txt` | Exact Python package versions used |
| `CITATION.cff` | Citation metadata for this repository |
| `LICENSE` | MIT licence for the code (results are CC BY 4.0, see *Licences*) |

## 3. Requirements and installation

Python 3.14 with the packages pinned in `requirements.txt` (NumPy, pandas, SciPy, scikit-learn,
LightGBM, PyTorch CPU build, PyArrow). No GPU is needed; the full run takes 7 to 16 minutes
on an 8-core laptop CPU and needs about 2 GB of RAM.

```
git clone <repository URL>
cd <repository folder>
python -m pip install -r requirements.txt
```

Newer package versions are expected to work but may change numbers in the last decimal places.

## 4. How to download the required data

The code reads two public datasets that are **not** included in this repository and must not be
committed to it (`.gitignore` excludes `data/texbat/` and `data/nuscenes/`):

| Dataset | Used by | Size the code needs | Provider and terms |
|---|---|---|---|
| TEXBAT processed receiver observables | Part A | about 351 MB (12 files) | University of Texas at Austin Radionavigation Laboratory; research use permitted, redistribution not permitted |
| nuScenes v1.0-mini metadata tables | Parts B and C | about 33 MB (9 JSON files, extracted from a 4.2 GB archive) | Motional; free registration, CC BY-NC-SA 4.0 plus the nuScenes Terms of Use |

### 4.1 TEXBAT (Part A)

TEXBAT is a battery of recorded GPS spoofing attacks. The study does not process the raw IQ
recordings (about 48 GB per scenario); it uses the laboratory's published outputs of its GRID software
receiver: channel observables at 5 Hz (`channel.mat`), the navigation solution (`navsol.mat`) and the
symmetric difference of the correlation function at 100 Hz (`symmdiff.mat`) for each of the four
scenarios `cleanStatic`, `ds2`, `ds3` and `ds7`.

From the repository root:

```
python code/download_texbat.py --dry-run   # checks that the 12 files are reachable (no download)
python code/download_texbat.py             # downloads them into data/texbat/<scenario>/
```

The script fetches each file from `https://rnl-data.ae.utexas.edu/datastore/texbat/processed/<scenario>/`
and skips files that are already present with the right size. To download by hand instead, take
`channel.mat`, `navsol.mat` and `symmdiff.mat` from each scenario folder at that address and place them
as shown in `data/README.md`. The TEXBAT home page, with the scenario descriptions and the data terms,
is https://radionavlab.ae.utexas.edu/texbat/ ; the column definitions of the `.mat` files are published
at `http://radionavlab.ae.utexas.edu/datastore/satNavCourse/` (for example `channeldef.txt`).

### 4.2 nuScenes v1.0-mini (Parts B and C)

1. Create a free account at https://www.nuscenes.org/nuscenes and accept the nuScenes Terms of Use.
2. Under *Downloads*, *Full dataset (v1.0)*, download **Mini** (`v1.0-mini.tgz`, about 4.2 GB).
3. Extract the archive and copy the nine JSON tables from its `v1.0-mini/` folder into
   `data/nuscenes/v1.0-mini/`: `scene.json`, `sample.json`, `sample_data.json`, `ego_pose.json`,
   `calibrated_sensor.json`, `sensor.json`, `sample_annotation.json`, `category.json`, `instance.json`.

The images, LiDAR sweeps and maps in the archive are not used and can be deleted. The code reads only
the ego poses of the LIDAR_TOP sample data (the vehicle trajectories) and the 3D object annotations at
the 2 Hz key frames; every other sensor signal is synthesised from those trajectories by
`code/fusion_sim.py` with the noise models stated in the article.

### 4.3 Expected layout

```
data/
├── texbat/
│   ├── cleanStatic/   channel.mat  navsol.mat  symmdiff.mat
│   ├── ds2/           channel.mat  navsol.mat  symmdiff.mat
│   ├── ds3/           channel.mat  navsol.mat  symmdiff.mat
│   └── ds7/           channel.mat  navsol.mat  symmdiff.mat
└── nuscenes/
    └── v1.0-mini/     scene.json  sample.json  sample_data.json  ego_pose.json  calibrated_sensor.json
                       sensor.json  sample_annotation.json  category.json  instance.json
```

Please cite both datasets if you use them (references in `data/README.md`).

## 5. Reproducing the experiment

Run from the repository root:

```
python code/run_repeatable.py          # Parts A, B and C; 7 to 16 minutes on an 8-core laptop CPU
```

- `run_repeatable.py` sets `PYTHONHASHSEED=0` and seeds PyTorch, then runs `run_exp6.py` unchanged, so
  two runs give identical files. `python code/run_exp6.py` also works, but its Part B and C numbers
  then vary slightly from run to run (see *Known limits of reproducibility*).
- Add `--quick` to run a reduced design (two ramp rates, two repetitions, fewer GRU epochs) in a few
  minutes, which is useful to check the installation.
- Each run overwrites the result files in `results/` and appends to `results/run.log`. To keep a re-run
  separate from the deposited results, copy the repository or move the deposited `results/` folder
  aside first. A run also writes `results/texbat_epoch_features.parquet` and
  `results/texbat_profiles.json`; these are derived from TEXBAT and are therefore excluded from version
  control.

The two source files `gnss_texbat.py` and `run_exp6.py` still contain a `sys.path` line pointing at the
author's original workspace (`../experiment4/code`), where the shared module `evaluate.py` lived. That
path does not need to exist: `evaluate.py` is included in `code/`, and Python falls back to it.

## 6. What the result files, tables and figures show

The raw outputs of the run are the `exp6*.csv` files in `results/` (one row per scenario and
detector for Part A, one row per episode and configuration or detector for Parts B and C) together
with the onset, threshold and receiver-flag settings in the JSON files. `results/README.md` documents
every column. The article's tables and figures are views of these files:

| Article | File | What it shows and how to read it |
|---|---|---|
| Table 1 | `results/tables/table1_texbat_detection.csv` | Receiver-level detection on TEXBAT: for each spoofing scenario (ds2, ds3, ds7) and detector, the PR-AUC, recall, precision, false positives per hour of clean recording and median detection latency. The overpowered ds2 is detected at once with no false alarms; the power-matched ds3 is caught through correlation distortion; the carrier-phase-aligned ds7 is detected only when the spoofer starts to pull the clock, about 46 s after onset. |
| Table 2 | `results/tables/table2_localisation_error.csv` | Localisation error per threat family (no attack, GNSS only, coordinated, perception only, LiDAR-odometry bias only) and fusion configuration F0 to F4: RMSE during the attack, maximum error, fraction of time beyond 1.5 m (lane-level failure) and beyond 5 m (catastrophic), and the resulting global-frame error of the lead vehicle. Error falls from F0 to F4 under GNSS-only attacks, rises again for F2 under the coordinated attack, and is unaffected by perception-only attacks. |
| Table 3 | `results/tables/table3_attack_detection.csv` | Attack detection per threat family and detector: PR-AUC, fraction of attack episodes detected, fraction of attack time steps flagged, false-alarm events per hour and median latency. Perception-only rows contain no true positives, so they report false alarms only. |
| Figure 1 | `results/figures/fig3_texbat_prauc.png` | PR-AUC of each receiver-level detector on the three TEXBAT scenarios; the bars show that physics-based monitors match or beat the learned detectors on every scenario. |
| Figure 2 | `results/figures/fig1_degradation.png` | Position RMSE during the attack against the pull-off rate, for GNSS-only and coordinated attacks and each configuration. F0 and F1 grow linearly with the pull-off rate; F2 to F4 flatten the curve, with F4 lowest. |
| Figure 3 | `results/figures/fig2_catastrophic.png` | Fraction of time with localisation error above 5 m per threat family and configuration; the catastrophic fraction is largest for F0, smallest for F4, and zero for perception-only and no-attack episodes. |
| Figure 4 | `results/figures/fig4_detection.png` | Episode detection rate of each detector for GNSS-only, coordinated and LiDAR-odometry-bias-only attacks; the CUSUM monitors and the OR-ensemble lead, the plain NIS gate trails. |

The figure file names do not follow the article's numbering; use the mapping above.

## 7. What the experiments found, and why it matters

**Receiver-level detection is precise but slow against a competent spoofer.** On TEXBAT the
overpowered attack (ds2) is trivially detected by the received-power monitor alone (PR-AUC 1.000, no
false alarms). The power-matched attack (ds3) is caught through correlation distortion (PR-AUC 0.885).
The carrier-phase-aligned attack (ds7) is the hard case: during its 40 s take-over the observables are
indistinguishable from clean data, and every detector, physics-based or learned, fires only once the
spoofer starts to pull the receiver clock, about 46 s after onset. Calibrated deterministic monitors
reach a PR-AUC of at least 0.985, and the learned detectors are no earlier than the physics monitors.
A receiver-level flag therefore buys a confirmation with a latency of tens of seconds, not an
instantaneous alarm.

**Fusion without consistency checks is catastrophic; fusion with physics monitors degrades
gracefully.** Under GNSS-only attacks the naive EKF (F0) follows the spoofer (RMSE 9.7 m, 29 % of the
attack time more than 5 m off). Chi-square gating (F1) rejects jumps but is walked away by slow ramps
(6.0 m). Cross-checking GNSS against dead reckoning and LiDAR odometry (F2) breaks the walk (3.6 m,
12 % catastrophic), the receiver flag (F3) adds little on top (3.5 m), and the learned uncertainty
model (F4) is the most resilient (3.4 m, 11 %).

**A coordinated attack on the reference modality erodes, but does not destroy, that resilience.** When
the LiDAR-odometry bias is shaped to follow the GNSS pull-off, the odometry cross-check is blinded and
the F2 error rises to 4.7 m (18 % catastrophic). Degradation stays graceful only because IMU and
wheel dead reckoning remain an independent witness that the attacker does not control; F4 again gives
the lowest error (4.1 m). The design lesson is diversity of independent references rather than the
sophistication of any single check.

**Perception attacks are invisible to every localisation defence.** Ghost obstacles and hidden lead
vehicles produce 4.8 ghost-induced emergency-brake events and 3.1 missed-lead-vehicle frames per
episode, identically for all five fusion configurations. A one-frame persistence check reduces ghost
braking to 3.9 events per episode but is defeated by a spoofer that re-injects the ghost consistently,
so perception attacks need perception-side (physics-based) defences. Conversely, localisation attacks
do leak into perception: the global-frame position error of the lead vehicle is 3.4 m for F0 against
0.8 m for F4 under GNSS attack, so a planner that reasons in map coordinates inherits the full
localisation failure even when the detector itself is untouched.

**Attack detection.** The physics CUSUM monitors detect 70 % of GNSS-only episodes with a median
latency of 6.4 s and 77 % of coordinated episodes; the GRU has the highest PR-AUC (0.972) but a lower
episode recall at a matched false-alarm budget; the receiver flag catches every GNSS episode but only
after about 11 s.

**Why it matters.** These results map directly onto the threat analysis and risk assessment that
ISO/SAE 21434 requires of vehicle manufacturers: a coordinated GNSS-plus-LiDAR-odometry attack is a
distinct threat scenario with a higher impact than either component alone, and the effective risk
treatment is architectural, namely an independent inertial witness and monitor-gated fusion, rather
than any single detector. They also show that localisation security and perception security cannot
be engineered in isolation, because each attack surface propagates into the other.

## 8. Known limits of reproducibility

- **Part A is deterministic.** Its learners use fixed random states; a re-run reproduced
  `exp6A_results.csv`, `texbat_onsets.json` and `rx_flag_stats.json` byte for byte.
- **Parts B and C are deterministic only through `run_repeatable.py`.** `run_exp6.py` derives episode
  seeds partly from Python's salted `hash()` and does not seed PyTorch, so direct runs differ slightly
  from one another. The wrapper fixes both sources; two runs through it produced identical files.
- **The deposited results cannot be regenerated bit for bit.** The run that produced them on
  13 September 2026 did not record its hash seed, so a re-run draws different sensor noise and attack
  start times. In a full re-run through `run_repeatable.py` on 6 October 2026, the median difference
  across 27 headline numbers of the article was 3.3 % and the largest 15 % (on a small quantity,
  F4's lead-object error of 0.80 m against 0.68 m), and every qualitative conclusion held: the
  ordering F0 > F1 > F2 > F4 in error under GNSS-only attacks, F4 lowest under the coordinated attack,
  the coordinated attack raising F2's error above the GNSS-only case, identical ghost braking for all
  configurations, and the persistence check reducing it.
- A quick-mode run (`--quick`) uses a reduced design and is meant only to verify the installation; its
  numbers are not comparable with the article.
- **Independent check of this package (25 October 2026).** The files in this repository were copied to
  a clean folder together with only the 21 data files listed in Section 4, and
  `python code/run_repeatable.py --quick` ran end to end in 114 s with no errors. The Part A outputs (`exp6A_results.csv`, `texbat_onsets.json`,
  `rx_flag_stats.json`) were byte-identical to the deposited ones, and the reduced Part B design
  reproduced every qualitative conclusion (F0 > F1 > F2 > F3 > F4 in error under GNSS-only attacks,
  the coordinated attack raising F2's error, F4 lowest in both cases).

## 9. Data availability and citation

**Data availability.** The TEXBAT processed observables and the nuScenes v1.0-mini dataset analysed in
this study are openly available from their providers at https://radionavlab.ae.utexas.edu/texbat/ and
https://www.nuscenes.org/nuscenes (see Section 4). The code, scenario definitions and result files that
support the findings of the study are the contents of this repository; nothing else was used.

**Citation.** Please cite this repository (`CITATION.cff`) together with the article:

> Obafemi, A. (2026). Graceful or Catastrophic? Sensor-Fusion Resilience of Autonomous-Vehicle
> Localisation under Coordinated Global Navigation Satellite System Spoofing and Perception Attacks.
> *IET Radar, Sonar & Navigation* (submitted).

If you use the datasets, cite them as well (references in `data/README.md`).

## 10. Licences

- Code in `code/`: MIT licence (`LICENSE`).
- Result files in `results/`: Creative Commons Attribution 4.0 (CC BY 4.0).
- TEXBAT and nuScenes are third-party datasets under their providers' terms and are not part of this
  repository. TEXBAT may be used for research but not redistributed; nuScenes is available after free
  registration under CC BY-NC-SA 4.0 with additional terms of use. Do not commit either dataset, or
  files derived from them (`results/texbat_epoch_features.parquet`, `results/texbat_profiles.json`),
  to a public repository.

## 11. Contact

Akinwole Obafemi, Edinburgh Napier University, Edinburgh, United Kingdom.
Email: akinwole.obafemi@napier.ac.uk. ORCID: https://orcid.org/0009-0003-6875-1900
