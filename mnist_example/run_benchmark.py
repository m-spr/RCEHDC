#!/usr/bin/env python3
"""Run online_learn_eval across multiple dimension sizes and repetitions."""

import json
import subprocess
import sys
import re
import pathlib

DIMENSIONS = [1000, 2000, 4000, 8000, 10000]
RUNS = 20
OUTPUT_FILE = "benchmark_results.txt"

script_dir = pathlib.Path(__file__).parent.resolve()
config_path = script_dir / "config.json"
eval_script = script_dir / "online_learn_eval.py"


def update_dimensions(dim: int):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["DIMENSIONS"] = dim
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)


def run_once() -> str:
    """Run online_learn_eval.py and return its stdout."""
    result = subprocess.run(
        [sys.executable, str(eval_script)],
        capture_output=True,
        text=True,
        cwd=str(script_dir),
    )
    return result.stdout


def parse_accuracies(output: str) -> list[float]:
    """Extract the three 'Testing accuracy of XX.XXX%' values."""
    return [float(m) for m in re.findall(r"Testing accuracy of ([\d.]+)%", output)]


def main():
    with open(script_dir / OUTPUT_FILE, "w") as out:
        for dim in DIMENSIONS:
            out.write(f"=== DIMENSIONS {dim} ===\n")
            print(f"\n{'='*40}")
            print(f"  DIMENSIONS = {dim}")
            print(f"{'='*40}")
            update_dimensions(dim)

            for run in range(1, RUNS + 1):
                print(f"  dim={dim}  run {run}/{RUNS} …", flush=True)
                stdout = run_once()
                accs = parse_accuracies(stdout)

                if len(accs) == 3:
                    line = f"test: {accs[0]:.2f} epoch1: {accs[1]:.2f} epoch2: {accs[2]:.2f}"
                else:
                    line = f"ERROR: expected 3 accuracy values, got {len(accs)}: {accs}"

                out.write(line + "\n")
                out.flush()
                print(f"    -> {line}")

    print(f"\nResults written to {script_dir / OUTPUT_FILE}")


if __name__ == "__main__":
    main()
