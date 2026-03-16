#!/usr/bin/env python3
"""Benchmark eval_other_datasets.py across dimensions, runs, datasets and modes.

Produces 4 result files (written incrementally as results come in):
  results_ISOLET_qa.txt
  results_ISOLET_onlinehd.txt
  results_UCIHAR_qa.txt
  results_UCIHAR_onlinehd.txt

Each file has the same format as benchmark_results.txt:
  === DIMENSIONS 1000 ===
  test: XX.XXX epoch1: XX.XXX epoch2: XX.XXX
  ...
"""

import json
import subprocess
import sys
import re
import pathlib

DIMENSIONS = [1000, 2000, 4000, 8000, 10000]
RUNS = 20
DATASETS = ["UCIHAR", "ISOLET"]
MODES = ["qa", "onlinehd"]

script_dir = pathlib.Path(__file__).parent.resolve()
repo_root = script_dir.parent
project_dir = repo_root / "mnist_example"
config_path = project_dir / "config.json"
eval_script = script_dir / "eval_other_datasets.py"


def update_dimensions(dim: int):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["DIMENSIONS"] = dim
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)


def run_once(dataset: str, mode: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(eval_script), dataset]
    if mode == "onlinehd":
        cmd.append("onlineHD")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(project_dir),
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
    # Open all 4 output files upfront so results flush to disk as they arrive
    output_paths = {
        (ds, mode): project_dir / f"results_{ds}_{mode}.txt"
        for ds in DATASETS
        for mode in MODES
    }
    handles = {key: open(path, "w") for key, path in output_paths.items()}

    try:
        for dataset in DATASETS:
            for mode in MODES:
                key = (dataset, mode)
                out = handles[key]
                print(f"\n{'='*60}")
                print(f"  Dataset: {dataset}  |  Mode: {mode.upper()}")
                print(f"{'='*60}")

                for dim in DIMENSIONS:
                    out.write(f"=== DIMENSIONS {dim} ===\n")
                    out.flush()
                    print(f"\n  --- dim={dim} ---")
                    update_dimensions(dim)

                    for run in range(1, RUNS + 1):
                        print(
                            f"    [{dataset}][{mode}] dim={dim}  run {run}/{RUNS} …",
                            flush=True,
                        )
                        result = run_once(dataset, mode)
                        combined = result.stdout + "\n" + result.stderr
                        accs = parse_accuracies(combined)

                        if result.returncode != 0:
                            err = summarize_stderr(result.stderr)
                            line = (
                                f"ERROR: eval failed (returncode={result.returncode})"
                                + (f": {err}" if err else "")
                            )
                        elif len(accs) == 3:
                            line = (
                                f"test: {accs[0]:.3f} "
                                f"epoch1: {accs[1]:.3f} "
                                f"epoch2: {accs[2]:.3f}"
                            )
                        else:
                            err = summarize_stderr(result.stderr)
                            line = (
                                f"ERROR: expected 3 accuracy values, got {len(accs)}: {accs}"
                                + (f" | stderr: {err}" if err else "")
                            )

                        out.write(line + "\n")
                        out.flush()
                        print(f"      -> {line}")

    finally:
        for fh in handles.values():
            fh.close()

    print("\nDone. Results written to:")
    for path in output_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
