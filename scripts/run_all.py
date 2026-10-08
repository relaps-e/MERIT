"""Run MERIT for multiple datasets or missing-performance ratios."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merit.runner import main


if __name__ == "__main__":
    main(batch=True)
