"""Plot training metrics from log files."""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log(log_path):
    metrics = {
        "steps": [], "update": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
        "reward": [], "SR": [], "SPL": [], "fps": [],
    }
    actions = {
        "steps": [],
        "MoveAhead": [], "RotateLeft": [], "RotateRight": [], "Stop": [],
    }

    update_pattern = re.compile(
        r"\[\s*([\d,]+)\s*steps\s*\|\s*update\s+(\d+)\]\s*"
        r"policy_loss=([\-\d.]+)\s+"
        r"value_loss=([\d.]+)\s+"
        r"entropy=([\d.]+)\s+"
        r"reward=([\-\d.]+)\s+"
        r"SR=([\d.]+)\s+"
        r"SPL=([\d.]+)\s+"
        r"fps=(\d+)"
    )

    action_pattern = re.compile(
        r"action=(\w+)\s+count=\s*(\d+)\s+mean_reward=([\-\+\d.]+)"
    )

    current_step = None

    with open(log_path) as f:
        for line in f:
            m = update_pattern.search(line)
            if m:
                current_step = int(m.group(1).replace(",", ""))
                metrics["steps"].append(current_step)
                metrics["update"].append(int(m.group(2)))
                metrics["policy_loss"].append(float(m.group(3)))
                metrics["value_loss"].append(float(m.group(4)))
                metrics["entropy"].append(float(m.group(5)))
                metrics["reward"].append(float(m.group(6)))
                metrics["SR"].append(float(m.group(7)))
                metrics["SPL"].append(float(m.group(8)))
                metrics["fps"].append(int(m.group(9)))
                continue

            m = action_pattern.search(line)
            if m and current_step is not None:
                name = m.group(1)
                count = int(m.group(2))
                if name == "MoveAhead" and (
                    len(actions["steps"]) == 0 or actions["steps"][-1] != current_step
                ):
                    actions["steps"].append(current_step)
                    for a in ["MoveAhead", "RotateLeft", "RotateRight", "Stop"]:
                        actions[a].append(0)
                if name in actions:
                    actions[name][-1] = count

    return metrics, actions


def plot_metrics(metrics, actions, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("See2Seek Training Progress", fontsize=14, fontweight="bold")

    steps_k = [s / 1000 for s in metrics["steps"]]

    # SR and SPL
    ax = axes[0, 0]
    ax.plot(steps_k, metrics["SR"], "b-", linewidth=2, label="SR")
    ax.plot(steps_k, metrics["SPL"], "g-", linewidth=2, label="SPL")
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Rate")
    ax.set_title("Success Rate & SPL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Episode Reward
    ax = axes[0, 1]
    ax.plot(steps_k, metrics["reward"], "r-", linewidth=2)
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Episode Reward")
    ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[0, 2]
    ax.plot(steps_k, metrics["policy_loss"], "b-", linewidth=1.5, label="Policy Loss")
    ax.plot(steps_k, metrics["value_loss"], "r-", linewidth=1.5, label="Value Loss")
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Loss")
    ax.set_title("Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    ax.plot(steps_k, metrics["entropy"], "m-", linewidth=2)
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.log(4), color="gray", linestyle="--", alpha=0.5, label="Max (uniform)")
    ax.legend()

    # Action Distribution
    ax = axes[1, 1]
    if actions["steps"]:
        a_steps_k = [s / 1000 for s in actions["steps"]]
        totals = [
            actions["MoveAhead"][i] + actions["RotateLeft"][i] +
            actions["RotateRight"][i] + actions["Stop"][i]
            for i in range(len(actions["steps"]))
        ]
        for name, color in [
            ("MoveAhead", "blue"), ("RotateLeft", "orange"),
            ("RotateRight", "green"), ("Stop", "red"),
        ]:
            fracs = [actions[name][i] / max(totals[i], 1) for i in range(len(totals))]
            ax.plot(a_steps_k, fracs, color=color, linewidth=1.5, label=name)
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Fraction")
    ax.set_title("Action Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # FPS
    ax = axes[1, 2]
    ax.plot(steps_k, metrics["fps"], "k-", linewidth=1.5)
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("FPS")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    log_dir = Path("data_new/logs")
    log_files = sorted(log_dir.glob("train_*.log"))

    if not log_files:
        print("No log files found in data_new/logs/")
        sys.exit(1)

    log_path = log_files[-1]
    print(f"Parsing: {log_path}")

    metrics, actions = parse_log(log_path)
    print(f"Found {len(metrics['steps'])} update entries")

    if len(metrics["steps"]) == 0:
        print("No metrics found in log file")
        sys.exit(1)

    print(f"Steps: {metrics['steps'][0]:,} -> {metrics['steps'][-1]:,}")
    print(f"Latest SR={metrics['SR'][-1]:.3f}  SPL={metrics['SPL'][-1]:.3f}  Reward={metrics['reward'][-1]:.3f}")

    save_path = "data_new/training_curves.png"
    plot_metrics(metrics, actions, save_path=save_path)
