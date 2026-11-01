"""Download the TEXBAT processed receiver observables used by Part A into data/texbat/.

The files come directly from the University of Texas at Austin Radionavigation Laboratory. They are
not redistributed with this repository: the TEXBAT terms allow research use, copying and modification
but ask users not to redistribute the data. See https://radionavlab.ae.utexas.edu/texbat/

Usage:  python code/download_texbat.py            # download the 12 files (about 360 MB)
        python code/download_texbat.py --dry-run  # only check that the files are reachable
Files that already exist with the expected size are skipped.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://rnl-data.ae.utexas.edu/datastore/texbat/processed/"
SETS = ["cleanStatic", "ds2", "ds3", "ds7"]
FILES = ["channel.mat", "navsol.mat", "symmdiff.mat"]
DEST = Path(__file__).resolve().parents[1] / "data" / "texbat"
UA = {"User-Agent": "Mozilla/5.0 (texbat-download for exp6 reproduction)"}


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def download(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    total = 0; failed = 0
    for s in SETS:
        for fn in FILES:
            url = BASE + f"{s}/{fn}"; path = DEST / s / fn
            try:
                size = remote_size(url)
            except Exception as e:
                print(f"UNREACHABLE  {url}  ({e})"); failed += 1; continue
            total += size or 0
            if a.dry_run:
                print(f"ok  {size / 1e6:8.1f} MB  {url}")
                continue
            if path.exists() and size is not None and path.stat().st_size == size:
                print(f"skip (present)  {path}"); continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {size / 1e6:6.1f} MB  {url}", flush=True)
            download(url, path)
            if size is not None and path.stat().st_size != size:
                print(f"SIZE MISMATCH  {path}"); failed += 1
    print(f"total {total / 1e6:.0f} MB; {'dry run, nothing written' if a.dry_run else 'files in ' + str(DEST)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
