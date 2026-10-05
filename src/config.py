"""Paths and global settings. Override locations with environment variables."""
import os
from pathlib import Path

_HOME = Path(os.environ.get("AMZML_HOME", Path.home() / "amzml"))
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", _HOME / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", _HOME / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT_DIR", _HOME / "output"))

N_JOBS = int(os.environ.get("BER_N_JOBS", os.cpu_count() or 4))          # threads (matmul, rapidfuzz)
N_PROCS = int(os.environ.get("BER_N_PROCS", min(6, os.cpu_count() or 4)))  # processes (RAM-heavy)
SEED = 42

for _d in (WORK_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
