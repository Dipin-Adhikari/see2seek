"""Plot training metrics from a log file or directory.

Examples:
    python scripts/plot_training.py --log_dir data_dino_baseline/logs
    python scripts/plot_training.py --log_dir data_dino_baseline/logs/train
    python scripts/plot_training.py --log-file path/to/train_20260914_141157.log

The default output is training_curves.png in the selected directory, or beside
the selected file. Without a log path, use logging.log_dir from the config.
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use this checkout's configuration when launched as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_log_files(log_path):
    """Accept a single log, a logs directory, or its train subdirectory."""
    log_path = Path(log_path).expanduser()
    if log_path.is_file():
        return [log_path]
    if not log_path.is_dir():
        raise FileNotFoundError(f"Log path does not exist: {log_path}")
    return sorted({
        path
        for pattern in ("train_*.log", "train/train_*.log")
        for path in log_path.glob(pattern)
        if path.is_file()
    })


def parse_log(log_path):
    metrics = {
        "steps": [], "update": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
        "reward": [], "SR": [], "SPL": [], "avg_steps": [], "fps": [],
    }
    actions = {
        "steps": [],
        "MoveAhead": [], "RotateLeft": [], "RotateRight": [], "Stop": [],
    }

    # Handles both the old format (...SPL=x.xxx fps=xx) and the current
    # format, which inserts avg_steps=N between SPL and fps, and may have
    # a trailing max_steps=N field.
    update_pattern = re.compile(
        r"\[\s*([\d,]+)\s*steps\s*\|\s*update\s+(\d+)\]\s*"
        r"policy_loss=([\-\d.]+)\s+"
        r"value_loss=([\d.]+)\s+"
        r"entropy=([\d.]+)\s+"
        r"reward=([\-\d.]+)\s+"
        r"SR=([\d.]+)\s+"
        r"SPL=([\d.]+)\s+"
        r"(?:avg_steps=(\d+)\s+)?"
        r"fps=(\d+)"
        r"(?:\s+max_steps=\d+)?"
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
                metrics["avg_steps"].append(
                    int(m.group(9)) if m.group(9) is not None else None
                )
                metrics["fps"].append(int(m.group(10)))
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


def merge_logs(log_files):
    """Parse multiple log files and concatenate their metrics/actions in order."""
    merged_metrics = {
        "steps": [], "update": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
        "reward": [], "SR": [], "SPL": [], "avg_steps": [], "fps": [],
    }
    merged_actions = {
        "steps": [],
        "MoveAhead": [], "RotateLeft": [], "RotateRight": [], "Stop": [],
    }

    for log_path in log_files:
        metrics, actions = parse_log(log_path)

        if len(metrics["steps"]) == 0:
            print(f"  Skipping {log_path.name}: no metrics found")
            continue

        # Offset steps so each subsequent file's steps continue from where
        # the previous one left off. This only applies to logs that
        # genuinely restart their own step counter near 0 (e.g. a fresh
        # run started independently of a checkpoint). A log resumed from a
        # checkpoint may start slightly *before* the previous file's last
        # step (crash/restart from an earlier checkpoint) - that is NOT a
        # restart-from-zero and must not be offset, or steps get inflated
        # every time this happens.
        step_offset = merged_metrics["steps"][-1] if merged_metrics["steps"] else 0
        first_step = metrics["steps"][0]
        if step_offset > 0 and first_step < step_offset * 0.1:
            offset = step_offset
        else:
            offset = 0

        for key in merged_metrics:
            if key == "steps":
                merged_metrics["steps"].extend(s + offset for s in metrics["steps"])
            else:
                merged_metrics[key].extend(metrics[key])

        for key in merged_actions:
            if key == "steps":
                merged_actions["steps"].extend(s + offset for s in actions["steps"])
            else:
                merged_actions[key].extend(actions[key])

        print(f"  {log_path.name}: {len(metrics['steps'])} updates "
              f"({metrics['steps'][0]:,} -> {metrics['steps'][-1]:,})"
              + (f"  [offset +{offset:,}]" if offset else ""))

    # Sort by step and drop any duplicate/out-of-order points that can
    # occur when consecutive logs have slightly overlapping step ranges
    # (e.g. a resume from a checkpoint a bit earlier than the last
    # recorded update of the previous log).
    if merged_metrics["steps"]:
        order = sorted(range(len(merged_metrics["steps"])),
                        key=lambda i: merged_metrics["steps"][i])
        for key in merged_metrics:
            merged_metrics[key] = [merged_metrics[key][i] for i in order]

        dedup_idx = []
        last_step = None
        for i, s in enumerate(merged_metrics["steps"]):
            if s != last_step:
                dedup_idx.append(i)
                last_step = s
        for key in merged_metrics:
            merged_metrics[key] = [merged_metrics[key][i] for i in dedup_idx]

    if merged_actions["steps"]:
        order = sorted(range(len(merged_actions["steps"])),
                        key=lambda i: merged_actions["steps"][i])
        for key in merged_actions:
            merged_actions[key] = [merged_actions[key][i] for i in order]

    return merged_metrics, merged_actions


def plot_metrics(metrics, actions, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("See2Seek Training Progress (All Runs)", fontsize=14, fontweight="bold")

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
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training curves from log files")
    parser.add_argument("--config", default=None, help="Path to YAML config (uses log_dir from config)")
    parser.add_argument("--log_dir", "--log-dir", "--log-file", default=None,
                        help="Log file, logs directory, or logs/train directory")
    parser.add_argument("--output", default=None, help="Override output image path")
    args = parser.parse_args()

    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser()
    elif args.config:
        from see2seek.utils.config import load_config
        cfg = load_config(args.config)
        log_dir = Path(cfg.logging.log_dir)
    else:
        from see2seek.utils.config import Config
        cfg = Config()
        log_dir = Path(cfg.logging.log_dir)

    try:
        log_files = find_log_files(log_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if not log_files:
        print(f"No training logs found in {log_dir}; looked for train_*.log and train/train_*.log")
        sys.exit(1)

    print(f"Found {len(log_files)} log file(s) in {log_dir}, parsing all...")
    metrics, actions = merge_logs(log_files)

    if len(metrics["steps"]) == 0:
        print("No metrics found in any log file")
        sys.exit(1)

    print(f"\nTotal: {len(metrics['steps'])} update entries")
    print(f"Steps: {metrics['steps'][0]:,} -> {metrics['steps'][-1]:,}")
    print(f"Latest SR={metrics['SR'][-1]:.3f}  SPL={metrics['SPL'][-1]:.3f}  Reward={metrics['reward'][-1]:.3f}")

    output_dir = log_dir.parent if log_dir.is_file() else log_dir
    save_path = Path(args.output).expanduser() if args.output else output_dir / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(metrics, actions, save_path=save_path)
