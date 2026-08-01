"""
check_episode_difficulty.py — Inspect shortest_path_length distribution
across your eval (or train) episode set.

Usage:
    python check_episode_difficulty.py --episodes_path imagenav_dataset/val/episodes --success_distance 1.0
"""

import argparse
import gzip
import json
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def _read_one(fp: str) -> List[Dict]:
    if fp.endswith(".gz"):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(fp, "r") as f:
            data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("episodes", [data])
    else:
        raise ValueError(f"Unexpected JSON format in {fp}")


def load_episodes(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Episodes path not found: {path}")

    episodes: List[Dict] = []

    if os.path.isdir(path):
        files: Dict[str, str] = {}
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json.gz"):
                stem = fname[: -len(".json.gz")]
                files[stem] = os.path.join(path, fname)
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json") and not fname.endswith(".json.gz"):
                stem = fname[: -len(".json")]
                files.setdefault(stem, os.path.join(path, fname))
        if not files:
            raise FileNotFoundError(f"No .json/.json.gz files found in {path}")
        for fp in files.values():
            episodes.extend(_read_one(fp))
    else:
        episodes.extend(_read_one(path))

    if not episodes:
        raise ValueError(f"No episodes found in {path}")
    return episodes


def get_spl_length(ep: Dict) -> float:
    """Prefer the pre-stored shortest_path_length field; fall back to
    computing it from waypoints if missing."""
    if "shortest_path_length" in ep and ep["shortest_path_length"] is not None:
        return float(ep["shortest_path_length"])

    wp = ep.get("shortest_path", [])
    if len(wp) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(wp[:-1], wp[1:]):
        total += float(np.linalg.norm([a["x"] - b["x"], a["z"] - b["z"]]))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_path", required=True)
    parser.add_argument("--success_distance", type=float, default=1.0)
    parser.add_argument("--out", default="episode_difficulty.png")
    args = parser.parse_args()

    episodes = load_episodes(args.episodes_path)
    lengths = np.array([get_spl_length(ep) for ep in episodes])

    n = len(lengths)
    free_success = int((lengths <= args.success_distance).sum())

    print(f"\n=== Episode difficulty stats: {args.episodes_path} ===")
    print(f"  num_episodes        : {n}")
    print(f"  mean                : {lengths.mean():.3f} m")
    print(f"  std                 : {lengths.std():.3f} m")
    print(f"  min                 : {lengths.min():.3f} m")
    print(f"  max                 : {lengths.max():.3f} m")
    print(f"  median              : {np.median(lengths):.3f} m")
    print(f"  p10 / p25 / p75 / p90: "
          f"{np.percentile(lengths,10):.2f} / {np.percentile(lengths,25):.2f} / "
          f"{np.percentile(lengths,75):.2f} / {np.percentile(lengths,90):.2f} m")
    print(f"  episodes <= success_distance ({args.success_distance}m): "
          f"{free_success} / {n}  ({free_success/n:.1%})")
    print(f"  episodes <= 0.5m    : {(lengths <= 0.5).sum()} ({(lengths<=0.5).mean():.1%})")
    print(f"  episodes <= 2.0m    : {(lengths <= 2.0).sum()} ({(lengths<=2.0).mean():.1%})")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Histogram
    axes[0].hist(lengths, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].axvline(args.success_distance, color="red", linestyle="--",
                     label=f"success_distance={args.success_distance}m")
    axes[0].set_xlabel("shortest_path_length (m)")
    axes[0].set_ylabel("num episodes")
    axes[0].set_title("Histogram of episode difficulty")
    axes[0].legend()

    # 2. Sorted line (continuous curve, low -> high difficulty)
    sorted_lengths = np.sort(lengths)
    axes[1].plot(sorted_lengths, color="darkorange")
    axes[1].axhline(args.success_distance, color="red", linestyle="--",
                     label=f"success_distance={args.success_distance}m")
    axes[1].set_xlabel("episode rank (sorted, easiest -> hardest)")
    axes[1].set_ylabel("shortest_path_length (m)")
    axes[1].set_title("Sorted difficulty curve")
    axes[1].legend()

    # 3. Cumulative fraction <= x (i.e. "how many episodes solvable within X m")
    cdf_x = np.sort(lengths)
    cdf_y = np.arange(1, n + 1) / n
    axes[2].plot(cdf_x, cdf_y, color="seagreen")
    axes[2].axvline(args.success_distance, color="red", linestyle="--",
                     label=f"success_distance={args.success_distance}m")
    axes[2].set_xlabel("shortest_path_length (m)")
    axes[2].set_ylabel("cumulative fraction of episodes")
    axes[2].set_title("CDF of episode difficulty")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()