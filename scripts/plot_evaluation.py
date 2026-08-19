"""Plot evaluation metrics from eval log files."""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_eval_log(log_path):
    episodes = []

    ep_pattern = re.compile(
        r"\[ep\s+(\d+)\]\s+scene=(\S+)\s+id=(\S+)\s+"
        r"success=(True|False)\s+steps=(\d+)\s+collisions=(\d+)\s+spl=([\d.]+)"
    )

    with open(log_path) as f:
        for line in f:
            m = ep_pattern.search(line)
            if m:
                episodes.append({
                    "ep": int(m.group(1)),
                    "scene": m.group(2),
                    "id": m.group(3),
                    "success": m.group(4) == "True",
                    "steps": int(m.group(5)),
                    "collisions": int(m.group(6)),
                    "spl": float(m.group(7)),
                })

    return episodes


def plot_evaluation(episodes, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"See2Seek Evaluation — {len(episodes)} episodes | "
        f"SR={sum(e['success'] for e in episodes)/len(episodes):.3f} | "
        f"SPL={np.mean([e['spl'] for e in episodes]):.3f}",
        fontsize=14, fontweight="bold"
    )

    ep_nums = [e["ep"] for e in episodes]
    successes = [int(e["success"]) for e in episodes]
    spls = [e["spl"] for e in episodes]
    steps = [e["steps"] for e in episodes]
    collisions = [e["collisions"] for e in episodes]

    # --- 1. Running SR and SPL ---
    ax = axes[0, 0]
    window = 50
    if len(successes) >= window:
        running_sr = [np.mean(successes[max(0,i-window):i+1]) for i in range(len(successes))]
        running_spl = [np.mean(spls[max(0,i-window):i+1]) for i in range(len(spls))]
        ax.plot(ep_nums, running_sr, "b-", linewidth=2, label=f"SR (rolling {window})")
        ax.plot(ep_nums, running_spl, "g-", linewidth=2, label=f"SPL (rolling {window})")
    else:
        ax.plot(ep_nums, successes, "b.", alpha=0.5, label="SR")
        ax.plot(ep_nums, spls, "g.", alpha=0.5, label="SPL")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_title("Success Rate & SPL (running avg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # --- 2. SPL Distribution ---
    ax = axes[0, 1]
    ax.hist(spls, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(spls), color="red", linestyle="--", linewidth=2,
               label=f"Mean={np.mean(spls):.3f}")
    ax.axvline(np.median(spls), color="orange", linestyle="--", linewidth=2,
               label=f"Median={np.median(spls):.3f}")
    ax.set_xlabel("SPL")
    ax.set_ylabel("Count")
    ax.set_title("SPL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. Steps Distribution ---
    ax = axes[0, 2]
    ax.hist(steps, bins=30, color="salmon", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(steps), color="red", linestyle="--", linewidth=2,
               label=f"Mean={np.mean(steps):.1f}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 4. Collisions Distribution ---
    ax = axes[1, 0]
    ax.hist(collisions, bins=max(max(collisions)+1, 10), color="lightcoral",
            edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(collisions), color="red", linestyle="--", linewidth=2,
               label=f"Mean={np.mean(collisions):.1f}")
    zero_pct = sum(1 for c in collisions if c == 0) / len(collisions) * 100
    ax.set_xlabel("Collisions per Episode")
    ax.set_ylabel("Count")
    ax.set_title(f"Collisions Distribution ({zero_pct:.0f}% zero-collision)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 5. Per-Scene SR and SPL ---
    ax = axes[1, 1]
    scenes = sorted(set(e["scene"] for e in episodes))
    scene_sr = []
    scene_spl = []
    scene_labels = []
    for s in scenes:
        scene_eps = [e for e in episodes if e["scene"] == s]
        scene_sr.append(np.mean([e["success"] for e in scene_eps]))
        scene_spl.append(np.mean([e["spl"] for e in scene_eps]))
        scene_labels.append(s.replace("FloorPlan_", ""))

    x = np.arange(len(scenes))
    width = 0.35
    ax.bar(x - width/2, scene_sr, width, label="SR", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, scene_spl, width, label="SPL", color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("Per-Scene Performance")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # --- 6. SPL vs Steps scatter ---
    ax = axes[1, 2]
    colors = ["green" if e["success"] else "red" for e in episodes]
    ax.scatter(steps, spls, c=colors, alpha=0.3, s=10)
    ax.set_xlabel("Steps")
    ax.set_ylabel("SPL")
    ax.set_title("SPL vs Episode Length")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    log_dir = Path("data_new/eval_logs")
    log_files = sorted(log_dir.glob("eval_*.log"))

    if not log_files:
        print("No eval log files found in data_new/eval_logs/")
        sys.exit(1)

    log_path = log_files[-1]
    print(f"Parsing: {log_path}")

    episodes = parse_eval_log(log_path)
    print(f"Found {len(episodes)} episodes")

    if not episodes:
        print("No episode data found in log file")
        sys.exit(1)

    sr = sum(e["success"] for e in episodes) / len(episodes)
    spl = np.mean([e["spl"] for e in episodes])
    print(f"SR={sr:.4f}  SPL={spl:.4f}")
    print(f"Mean steps={np.mean([e['steps'] for e in episodes]):.1f}")
    print(f"Mean collisions={np.mean([e['collisions'] for e in episodes]):.1f}")

    save_path = "data_new/eval_curves.png"
    plot_evaluation(episodes, save_path=save_path)
