"""MERIT model selection."""

import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from .factory import build_merit

__all__ = ["build_merit"]
