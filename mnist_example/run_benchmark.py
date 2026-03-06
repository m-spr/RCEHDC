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
use_standard_onlinehd = len(sys.argv) > 1 and sys.argv[1].lower() == "onlinehd"


def update_dimensions(dim: int):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["DIMENSIONS"] = dim
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)


def run_once() -> subprocess.CompletedProcess[str]:
    """Run online_learn_eval.py and return the completed process."""
    cmd = [sys.executable, str(eval_script)]
    if use_standard_onlinehd:
        cmd.append("onlineHD")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(script_dir),
    )


def parse_accuracies(output: str) -> list[float]:
    """Extract the three 'Testing accuracy of XX.XXX%' values."""
    return [float(m) for m in re.findall(r"Testing accuracy of ([\d.]+)%", output)]


def summarize_stderr(stderr: str, max_lines: int = 6) -> str:
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-max_lines:])


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
                result = run_once()
                combined_output = result.stdout + "\n" + result.stderr
                accs = parse_accuracies(combined_output)

                if result.returncode != 0:
                    err_summary = summarize_stderr(result.stderr)
                    line = (
                        f"ERROR: eval failed (returncode={result.returncode})"
                        + (f": {err_summary}" if err_summary else "")
                    )
                elif len(accs) == 3:
                    line = f"test: {accs[0]:.2f} epoch1: {accs[1]:.2f} epoch2: {accs[2]:.2f}"
                else:
                    err_summary = summarize_stderr(result.stderr)
                    line = (
                        f"ERROR: expected 3 accuracy values, got {len(accs)}: {accs}"
                        + (f" | stderr: {err_summary}" if err_summary else "")
                    )

                out.write(line + "\n")
                out.flush()
                print(f"    -> {line}")

    print(f"\nResults written to {script_dir / OUTPUT_FILE}")


if __name__ == "__main__":
    main()
