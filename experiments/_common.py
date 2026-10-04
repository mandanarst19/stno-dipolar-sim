"""Shared helpers for the experiment scripts (command-line options, output folder)."""
import argparse
import os
import sys

# make the package importable when running `python experiments/<script>.py` from the repo root
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:                      # pasted into a notebook cell: assume cwd is the repo root
    REPO_ROOT = os.getcwd()
sys.path.insert(0, REPO_ROOT)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--fast", action="store_true",
                        help="smoke test: much shorter runs (numbers are NOT meaningful)")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args, _ = parser.parse_known_args()   # ignore extra arguments added by Jupyter kernels
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return args


def time_scale(args):
    """Factor applied to all simulated durations (0.05 in --fast mode)."""
    return 0.05 if args.fast else 1.0


class Report:
    """Collects text output, prints it and writes it to results/<name>.txt."""

    def __init__(self, name):
        self.path = os.path.join(RESULTS_DIR, name + ".txt")
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self):
        with open(self.path, "w") as fh:
            fh.write("\n".join(self.lines) + "\n")
        print(f"\n[saved] {os.path.relpath(self.path, REPO_ROOT)}")
