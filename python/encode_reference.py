#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import List


def read_coe_bits(path: str) -> List[List[int]]:
    vectors: List[List[int]] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("memory_initialization"):
                continue
            line = line.rstrip(",;")
            if not line:
                continue
            if all(c in "01" for c in line):
                vectors.append([1 if c == "1" else 0 for c in line])
    if not vectors:
        raise ValueError(f"No vectors found in {path}")
    width = len(vectors[0])
    for i, v in enumerate(vectors):
        if len(v) != width:
            raise ValueError(f"Inconsistent width at line {i} in {path}")
    return vectors


def read_pixels_txt(path: str, expected: int) -> List[int]:
    pixels: List[int] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            pixels.append(int(line))
    if len(pixels) != expected:
        raise ValueError(f"Pixel file has {len(pixels)} values, expected {expected}")
    return pixels


def write_pixels_txt(path: str, pixels: List[int]) -> None:
    with open(path, "w") as f:
        for p in pixels:
            f.write(f"{p}\n")


def encode_hv(bv_mem: List[List[int]], id_mem: List[List[int]], pixels: List[int]) -> List[int]:
    dim = len(bv_mem[0])
    counts = [0] * dim
    feature_size = len(pixels)

    for i in range(feature_size):
        id_vec = id_mem[pixels[i]]
        bv_vec = bv_mem[i]
        for j in range(dim):
            if id_vec[j] == bv_vec[j]:
                counts[j] += 1

    threshold = feature_size // 2
    return [1 if c > threshold else 0 for c in counts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference encoder for base-level HDC.")
    parser.add_argument("--project-dir", default="mnist_example/", help="Project directory")
    parser.add_argument("--pixels-txt", default="", help="Optional input pixel file (one int per line)")
    parser.add_argument("--pixels-out", default="", help="Output pixel file")
    parser.add_argument("--hv-out", default="", help="Output encoder HV file")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for random pixels")
    args = parser.parse_args()

    project_dir = args.project_dir
    if not project_dir.endswith("/"):
        project_dir += "/"

    config_path = os.path.join(project_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    features = int(cfg["FEATURES"])
    num_levels = int(cfg["NUM_LEVELS"])

    bv_path = os.path.join(project_dir, "mem", "BV_img.coe")
    id_path = os.path.join(project_dir, "mem", "ID_img.coe")

    bv_mem = read_coe_bits(bv_path)
    id_mem = read_coe_bits(id_path)

    if len(bv_mem) != features:
        raise ValueError(f"BV entries {len(bv_mem)} != FEATURES {features}")

    if args.pixels_txt:
        pixels = read_pixels_txt(args.pixels_txt, features)
    else:
        rng = random.Random(args.seed)
        pixels = [rng.randrange(num_levels) for _ in range(features)]

    pixels_out = args.pixels_out or os.path.join(project_dir, "encoder_pixels.txt")
    hv_out = args.hv_out or os.path.join(project_dir, "encoder_hv.txt")

    write_pixels_txt(pixels_out, pixels)
    hv_bits = encode_hv(bv_mem, id_mem, pixels)

    with open(hv_out, "w") as f:
        f.write("".join("1" if b else "0" for b in hv_bits))
        f.write("\n")

    print(f"Wrote pixels: {pixels_out}")
    print(f"Wrote encoder HV: {hv_out}")


if __name__ == "__main__":
    main()
