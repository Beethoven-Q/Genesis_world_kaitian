#!/usr/bin/env python3
"""Entry point for the FULL-DR pick-place collector. See genesis_firefly/collectors/pickplace_collector.py.
  CUDA_VISIBLE_DEVICES=0 DATA_DIR=/data3/genesis_fulldr ./.venv/bin/python genesis_firefly/scripts/collect_pickplace.py <N> [seed]
"""
import os, sys
sys.path.insert(0, "genesis_firefly")
import genesis as gs
from collectors.pickplace_collector import collect
if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    gs.init(backend=gs.gpu)
    collect(N, SEED, os.environ.get("DATA_DIR", "/data3/genesis_fulldr"),
            os.environ.get("OUT_DIR", "genesis_firefly/output/temp/fulldr_collect"))
