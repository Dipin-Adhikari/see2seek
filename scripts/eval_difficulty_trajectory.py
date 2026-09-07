"""
Parse RoboTHOR-style eval logs and report SR / SPL broken down by
episode difficulty (easy / medium / hard), based on the shortest-path
geodesic distance of each episode.

Usage:
    python eval_difficulty_breakdown.py eval.log
    python eval_difficulty_breakdown.py eval.log --easy-max 3 --medium-max 6
    python eval_difficulty_breakdown.py eval.log --per-scene

Difficulty buckets (by `shortest` geodesic distance in meters) default to:
    easy   : shortest <= 3.0
    medium : 3.0 < shortest <= 6.0
    hard   : shortest > 6.0
These are close to the RoboTHOR ObjectNav convention but are just
thresholds on distance-to-goal, so tune them with --easy-max / --medium-max
if your episode set uses a different split.
"""

import argparse
import re
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"scene=(?P<scene>\S+)\s+"
    r"id=(?P<id>\S+)\s+"
    r"success=(?P<success>True|False)\s+"
    r"steps=(?P<steps>\d+)\s+"
    r"collisions=(?P<collisions>\d+)\s+"
    r"spl=(?P<spl>[\d.]+)\s+"
    r"path=(?P<path>[\d.]+)\s+"
    r"shortest=(?P<shortest>[\d.]+)"
)


def parse_log(path):
    episodes = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            episodes.append({
                "scene": d["scene"],
                "id": d["id"],
                "success": d["success"] == "True",
                "steps": int(d["steps"]),
                "collisions": int(d["collisions"]),
                "spl": float(d["spl"]),
                "path": float(d["path"]),
                "shortest": float(d["shortest"]),
            })
    return episodes


def difficulty_of(ep, easy_max, medium_max):
    s = ep["shortest"]
    if s <= easy_max:
        return "easy"
    elif s <= medium_max:
        return "medium"
    else:
        return "hard"


def summarize(episodes):
    n = len(episodes)
    if n == 0:
        return {"n": 0, "sr": 0.0, "spl": 0.0}
    sr = sum(e["success"] for e in episodes) / n
    spl = sum(e["spl"] for e in episodes) / n
    return {"n": n, "sr": sr, "spl": spl}


def print_id_list(bucket, level, only_failed):
    """Print episode ids from a difficulty bucket, optionally filtered to failures."""
    if only_failed:
        ids = [e["id"] for e in bucket if not e["success"]]
        label = f"Failed {level} episode ids"
    else:
        ids = [e["id"] for e in bucket if e["success"]]
        label = f"Successful {level} episode ids"
    print(f"\n{label} ({len(ids)} of {len(bucket)} {level} episodes):")
    for eid in ids:
        print(eid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", help="path to eval log file")
    ap.add_argument("--easy-max", type=float, default=3.0, help="max shortest-path distance (m) for 'easy' (default 3.0)")
    ap.add_argument("--medium-max", type=float, default=6.0, help="max shortest-path distance (m) for 'medium' (default 6.0); above this is 'hard'")
    ap.add_argument("--per-scene", action="store_true", help="also break down by scene within each difficulty bucket")
    ap.add_argument("--list-hard", action="store_true", help="print the ids of successful 'hard' episodes")
    ap.add_argument("--list-hard-failed", action="store_true", help="print only the ids of 'hard' episodes that failed")
    ap.add_argument("--list-medium", action="store_true", help="print the ids of successful 'medium' episodes")
    ap.add_argument("--list-medium-failed", action="store_true", help="print only the ids of 'medium' episodes that failed")
    args = ap.parse_args()

    episodes = parse_log(args.logfile)
    if not episodes:
        print(f"No episodes parsed from {args.logfile}. Check the log format.", file=sys.stderr)
        sys.exit(1)

    buckets = defaultdict(list)
    for ep in episodes:
        buckets[difficulty_of(ep, args.easy_max, args.medium_max)].append(ep)

    overall = summarize(episodes)
    print(f"Parsed {overall['n']} episodes from {args.logfile}")
    print(f"Difficulty thresholds: easy <= {args.easy_max}m, medium <= {args.medium_max}m, hard > {args.medium_max}m\n")

    header = f"{'Difficulty':<10} {'N':>6} {'SR':>8} {'SPL':>8}"
    print(header)
    print("-" * len(header))
    for level in ("easy", "medium", "hard"):
        s = summarize(buckets[level])
        print(f"{level:<10} {s['n']:>6} {s['sr']*100:>7.1f}% {s['spl']:>8.3f}")

    s = summarize(episodes)
    print("-" * len(header))
    print(f"{'overall':<10} {s['n']:>6} {s['sr']*100:>7.1f}% {s['spl']:>8.3f}")

    if args.per_scene:
        print("\nPer-scene breakdown within each difficulty bucket:")
        for level in ("easy", "medium", "hard"):
            print(f"\n[{level}]")
            by_scene = defaultdict(list)
            for ep in buckets[level]:
                by_scene[ep["scene"]].append(ep)
            for scene in sorted(by_scene):
                s = summarize(by_scene[scene])
                print(f"  {scene:<20} n={s['n']:<4} SR={s['sr']*100:5.1f}% SPL={s['spl']:.3f}")

    if args.list_hard:
        print_id_list(buckets["hard"], "hard", only_failed=False)
    if args.list_hard_failed:
        print_id_list(buckets["hard"], "hard", only_failed=True)
    if args.list_medium:
        print_id_list(buckets["medium"], "medium", only_failed=False)
    if args.list_medium_failed:
        print_id_list(buckets["medium"], "medium", only_failed=True)


if __name__ == "__main__":
    main()