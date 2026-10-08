"""Run MERIT for a scenario, dataset and missing-performance ratio."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merit.runner import main

if __name__ == "__main__":
    main()
