#!/usr/bin/env python3
"""
Filter short-distance episodes from the See2Seek RoboTHOR training dataset.

Drops episodes where shortest_path_length <= threshold (default 1.5m).
These near-zero start-to-goal episodes are the root cause identified in
the entropy-collapse debugging: ~18% of episodes had start == goal-adjacent,
making "ignore the goal embedding and just call Stop nearby" a locally
optimal policy. Removing them forces the agent to actually use the goal
embedding to get reward.

Usage:
    python filter_short_episodes.py --dataset_dir /home/dipin/See2Seek/dataset/train
    python filter_short_episodes.py --dataset_dir ... --dry_run   # report only
    python filter_short_episodes.py --dataset_dir ... --threshold 1.0


"""
import argparse
import gzip
import json
import shutil
from pathlib import Path


def load_episodes(jf: Path):
    if jf.suffix == ".gz":
        with gzip.open(jf, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(jf, "r") as f:
        return json.load(f)


def save_episodes(jf: Path, episodes):
    if jf.suffix == ".gz":
        with gzip.open(jf, "wt", encoding="utf-8") as f:
            json.dump(episodes, f)
    else:
        with open(jf, "w") as f:
            json.dump(episodes, f, indent=4)


def process_file(jf: Path, threshold: float, backup: bool, dry_run: bool):
    episodes = load_episodes(jf)

    before = len(episodes)
    kept = [ep for ep in episodes if ep.get("shortest_path_length", 0.0) > threshold]
    dropped_ids = [ep["id"] for ep in episodes if ep.get("shortest_path_length", 0.0) <= threshold]
    after = len(kept)

    if dropped_ids and not dry_run:
        if backup:
            backup_path = jf.with_suffix(jf.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(jf, backup_path)
        save_episodes(jf, kept)

    return before, after, dropped_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, required=True,
                         help="Path to dataset/train (contains episodes/ and embeddings.pt)")
    parser.add_argument("--episodes_subdir", type=str, default="episodes",
                         help="Subfolder holding per-scene episode json files (default: episodes)")
    parser.add_argument("--threshold", type=float, default=1.5,
                         help="Drop episodes with shortest_path_length <= threshold (default: 1.5)")
    parser.add_argument("--no_backup", action="store_true",
                         help="Skip creating .bak copies before overwriting json files")
    parser.add_argument("--dry_run", action="store_true",
                         help="Only report counts, don't modify anything")
    parser.add_argument("--dump_dropped_ids", type=str, default=None,
                         help="Optional path to write a .txt of all dropped episode ids "
                              "(useful if you need to also prune embeddings.pt)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    episodes_dir = dataset_dir / args.episodes_subdir

    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes dir not found: {episodes_dir}")

    json_files = sorted(episodes_dir.glob("*.json")) + sorted(episodes_dir.glob("*.json.gz"))
    if not json_files:
        raise FileNotFoundError(f"No .json or .json.gz files found in {episodes_dir}")

    total_before = total_after = 0
    all_dropped_ids = []
    rows = []

    for jf in json_files:
        before, after, dropped_ids = process_file(jf, args.threshold, not args.no_backup, args.dry_run)
        total_before += before
        total_after += after
        all_dropped_ids.extend(dropped_ids)
        rows.append((jf.name, before, after, before - after))

    name_w = max(len(r[0]) for r in rows) + 2
    print(f"{'FILE':<{name_w}} {'BEFORE':>8} {'AFTER':>8} {'DROPPED':>8}")
    for name, b, a, d in rows:
        print(f"{name:<{name_w}} {b:>8} {a:>8} {d:>8}")

    total_dropped = total_before - total_after
    pct = (total_dropped / total_before * 100) if total_before else 0
    print("-" * (name_w + 28))
    print(f"TOTAL: {total_before} -> {total_after} episodes ({total_dropped} dropped, {pct:.1f}%)")

    if args.dump_dropped_ids:
        with open(args.dump_dropped_ids, "w") as f:
            f.write("\n".join(all_dropped_ids))
        print(f"\nWrote {len(all_dropped_ids)} dropped episode ids to {args.dump_dropped_ids}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
    else:
        print(f"\nDropped episodes with shortest_path_length <= {args.threshold}m.")
        if not args.no_backup:
            print("Originals backed up alongside each file with a .bak extension.")



if __name__ == "__main__":
    main()