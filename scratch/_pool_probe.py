"""What does an idle render worker cost? Run as a script."""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))


def commit_of_self(_):
    import fiducia.ortho  # noqa: F401  -- what a worker loads
    import numpy as np

    np.dot(np.ones((400, 400)), np.ones((400, 400)))   # wake the BLAS threads
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-Process -Id {os.getpid()}).PrivateMemorySize64"],
                         capture_output=True, text=True)
    return int(out.stdout.strip()), os.environ.get("OPENBLAS_NUM_THREADS")


if __name__ == "__main__":
    for single in (False, True):
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            if single:
                os.environ[name] = "1"
            else:
                os.environ.pop(name, None)
        with ProcessPoolExecutor(max_workers=1) as pool:
            size, setting = pool.submit(commit_of_self, 0).result()
        print(f"BLAS threads {'1 ' if single else 'default'}: idle worker commits {size / 2**20:.0f} MB")
