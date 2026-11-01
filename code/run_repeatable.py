"""Run code/run_exp6.py with every random source fixed, so that repeated runs give identical results.

run_exp6.py, kept unchanged from the version that produced the deposited results, leaves two random
sources unfixed:
  1. it derives episode seeds partly from Python's hash() of attack names, which Python salts
     differently in every process unless PYTHONHASHSEED is set;
  2. it does not seed PyTorch, which initialises and shuffles the GRU detector of Part C.
This wrapper sets PYTHONHASHSEED=0 (restarting itself if necessary) and torch.manual_seed(0), then
runs run_exp6.py exactly as it is.

Usage:  python code/run_repeatable.py [--quick]
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

SEED = "0"

if os.environ.get("PYTHONHASHSEED") != SEED:
    # the hash seed is read once at interpreter start-up, so restart with it set
    sys.exit(subprocess.call([sys.executable, __file__, *sys.argv[1:]], env=dict(os.environ, PYTHONHASHSEED=SEED)))

import torch  # noqa: E402

torch.manual_seed(int(SEED))
script = Path(__file__).resolve().with_name("run_exp6.py")
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
