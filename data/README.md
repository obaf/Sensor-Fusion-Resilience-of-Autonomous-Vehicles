# Input data

The code reads two public third-party datasets. Neither is included in this repository, because
their terms do not allow it:

- **TEXBAT** (University of Texas at Austin Radionavigation Laboratory) may be used, copied and
  modified for research, but its providers ask users not to redistribute it.
- **nuScenes** (Motional) is available after free registration under CC BY-NC-SA 4.0 with additional
  terms of use, which require each user to obtain it from nuScenes directly.

Download both from their providers as described below. About 400 MB of disk space is needed for the
files the code reads (the full nuScenes mini archive is 4.2 GB while it is being extracted).

## Expected layout

```
data/
├── texbat/
│   ├── cleanStatic/   channel.mat  navsol.mat  symmdiff.mat
│   ├── ds2/           channel.mat  navsol.mat  symmdiff.mat
│   ├── ds3/           channel.mat  navsol.mat  symmdiff.mat
│   └── ds7/           channel.mat  navsol.mat  symmdiff.mat
└── nuscenes/
    └── v1.0-mini/     scene.json  sample.json  sample_data.json  ego_pose.json
                       calibrated_sensor.json  sensor.json  sample_annotation.json
                       category.json  instance.json
```

## TEXBAT processed receiver observables (Part A), about 350 MB

These are the outputs of the UT GRID software receiver for the authentic recording `cleanStatic`
and the spoofing scenarios `ds2` (static overpowered time push), `ds3` (static matched-power time
push) and `ds7` (matched-power, carrier-phase-aligned time push).

From the repository root, run:

```
python code/download_texbat.py
```

The script fetches the 12 files from
`https://rnl-data.ae.utexas.edu/datastore/texbat/processed/<scenario>/` into the layout above and
skips files that are already present. `python code/download_texbat.py --dry-run` only checks that the
files are reachable. To download by hand instead, take `channel.mat`, `navsol.mat` and
`symmdiff.mat` from each of the four scenario folders. The column definitions of the `.mat` files are
published at `http://radionavlab.ae.utexas.edu/datastore/satNavCourse/` (for example
`channeldef.txt`). TEXBAT page: https://radionavlab.ae.utexas.edu/texbat/

## nuScenes v1.0-mini metadata (Parts B and C), about 33 MB

1. Create a free account at https://www.nuscenes.org/nuscenes and accept the nuScenes Terms of Use.
2. Under *Downloads*, *Full dataset (v1.0)*, download **Mini** (`v1.0-mini.tgz`, 4.2 GB).
3. Extract the archive and copy the nine JSON tables listed above from its `v1.0-mini/` folder into
   `data/nuscenes/v1.0-mini/`. The images, LiDAR sweeps and maps in the archive are not used.

The code uses only the ego poses of the LIDAR_TOP sample data (the vehicle trajectories) and the 3D
object annotations at key frames. All other sensor signals are synthesised from those trajectories
by `code/fusion_sim.py`.

## Citing the datasets

Humphreys, T. E., Bhatti, J. A., Shepard, D. P., & Wesson, K. D. (2012). The Texas spoofing test
battery: Toward a standard for evaluating GPS signal authentication techniques. In *Proceedings of
ION GNSS 2012* (pp. 3569–3583).

Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., … Beijbom, O. (2020). nuScenes:
A multimodal dataset for autonomous driving. In *Proceedings of the 2020 IEEE/CVF Conference on
Computer Vision and Pattern Recognition* (pp. 11618–11628). https://doi.org/10.1109/CVPR42600.2020.01164
