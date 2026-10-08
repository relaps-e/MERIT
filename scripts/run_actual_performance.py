"""Run the 24 actual-performance targets, or a selected benchmark and dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merit.runner import main

if __name__ == "__main__":
    main(batch=True, actual=True)
