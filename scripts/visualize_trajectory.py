"""
visualize_trajectory.py — Bird's-eye-view trajectory visualization.

Runs the trained agent through specified episodes, records positions
at each step, and generates top-down trajectory plots for the results section.

Usage:
    # Single episode
    python scripts/visualize_trajectory.py \
        --checkpoint data_dino_v4/checkpoints/checkpoint_000000200704.pth \
        --episodes FloorPlan_Val1_1_AlarmClock_0

    # Multiple episodes
    python scripts/visualize_trajectory.py \
        --checkpoint data_dino_v4/checkpoints/checkpoint_000000200704.pth \
        --episodes FloorPlan_Val1_1_AlarmClock_0 FloorPlan_Val1_2_Bowl_0

    # Use the hardcoded list (edit EPISODE_LIST below)
    python scripts/visualize_trajectory.py \
        --checkpoint data_dino_v4/checkpoints/checkpoint_000000200704.pth \
        --use_list

    # Save to specific directory
    python scripts/visualize_trajectory.py \
        --checkpoint data_dino_v4/checkpoints/checkpoint_000000200704.pth \
        --episodes FloorPlan_Val1_1_AlarmClock_0 \
        --output_dir data_dino_v4/visualizations
"""

import argparse
import gzip
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from see2seek.utils.config import Config, load_config
from see2seek.agents.gru_policy import build_policy, EpisodicMemory
from see2seek.models.encoders.dino_encoder import DINOv2Encoder
from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder

logger = logging.getLogger(__name__)

# ============================================================================
# Hardcoded episode list — edit this for batch visualization
# ============================================================================
EPISODE_LIST = [
    "FloorPlan_Val1_1_AlarmClock_0",
    "FloorPlan_Val1_2_Bowl_0",
    "FloorPlan_Val1_3_Mug_0",
]


def parse_args():
    p = argparse.ArgumentParser(description="See2Seek — Bird's-Eye Trajectory Visualization")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config", default=None, help="Path to YAML config override")
    p.add_argument("--episodes", nargs="+", default=None, help="Episode IDs to visualize")
    p.add_argument("--use_list", action="store_true", help="Use hardcoded EPISODE_LIST")
    p.add_argument("--output_dir", default="data_dino_v7/visualizations", help="Output directory for plots")
    p.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    p.add_argument("--zero_pointgoal", action="store_true", help="Zero out PointGoal sensor")
    return p.parse_args()


def load_all_episodes(episodes_path: str) -> List[Dict]:
    """Load all episodes from the episodes directory."""
    episodes = []
    if os.path.isdir(episodes_path):
        for fname in sorted(os.listdir(episodes_path)):
            if fname.endswith(".json.gz"):
                with gzip.open(os.path.join(episodes_path, fname), "rt") as f:
                    episodes.extend(json.load(f))
            elif fname.endswith(".json"):
                stem = fname[:-len(".json")]
                if not os.path.exists(os.path.join(episodes_path, stem + ".json.gz")):
                    with open(os.path.join(episodes_path, fname), "r") as f:
                        episodes.extend(json.load(f))
    else:
        if episodes_path.endswith(".gz"):
            with gzip.open(episodes_path, "rt") as f:
                episodes.extend(json.load(f))
        else:
            with open(episodes_path, "r") as f:
                episodes.extend(json.load(f))
    return episodes


def run_episode(
    controller,
    episode: Dict,
    policy: torch.nn.Module,
    obs_encoder,
    goal_encoder,
    embeddings_registry: Dict,
    cfg,
    device: torch.device,
    max_steps: int = 500,
    zero_pointgoal: bool = False,
) -> Dict:
    """Run a single episode and return trajectory data."""
    from ai2thor.util.metrics import get_shortest_path_to_point

    scene = episode["scene"]
    controller.reset(scene=scene)

    start_rotation = {"x": 0, "y": episode.get("initial_orientation", 0.0), "z": 0}
    event = controller.step(
        action="TeleportFull",
        position=episode["initial_position"],
        rotation=start_rotation,
        horizon=episode.get("initial_horizon", 0),
    )

    if not event.metadata["lastActionSuccess"]:
        return {"success": False, "error": "TeleportFull failed"}

    goal_pos = episode["shortest_path"][-1]
    optimal_pose = episode.get("optimal_goal_pose", {})
    goal_heading = optimal_pose.get("rotation", 0.0)

    # Load goal embedding
    ep_id = episode["id"]
    goal_embed = embeddings_registry[ep_id].unsqueeze(0).to(device)

    # Initialize policy state
    hidden = policy.get_initial_hidden(1, device)
    memory_buffer = torch.zeros(1, policy.memory_size, policy.dino_cls_dim, device=device)
    memory_pose_buffer = torch.zeros(1, policy.memory_size, EpisodicMemory.POSE_DIM, device=device)
    memory_mask = torch.zeros(1, policy.memory_size, device=device, dtype=torch.bool)
    prev_action = torch.tensor([4], device=device)  # "no previous action" token
    masks = torch.ones(1, 1, device=device)

    # Dead-reckoned pose
    ego_x, ego_y, ego_theta = 0.0, 0.0, 0.0
    initial_orientation = episode.get("initial_orientation", 0.0)
    ego_theta = math.radians(initial_orientation)

    trajectory = []
    actions_taken = []
    headings = []

    agent_meta = event.metadata["agent"]
    pos = agent_meta["position"]
    trajectory.append((pos["x"], pos["z"]))
    headings.append(agent_meta["rotation"]["y"])

    success = False
    num_steps = 0

    for step in range(max_steps):
        num_steps = step + 1

        # Get observation
        frame = event.frame
        rgb_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device)

        # Encode observation
        with torch.no_grad():
            patch_embeds, cls_embed = obs_encoder(rgb_tensor)

        # Compute pointgoal
        agent_pos = event.metadata["agent"]["position"]
        agent_rot = event.metadata["agent"]["rotation"]["y"]
        dx = goal_pos["x"] - agent_pos["x"]
        dz = goal_pos["z"] - agent_pos["z"]
        geodesic_dist = math.sqrt(dx * dx + dz * dz)
        angle_to_goal = math.atan2(dx, dz) - math.radians(agent_rot)
        pointgoal = torch.tensor(
            [[geodesic_dist, math.cos(angle_to_goal), math.sin(angle_to_goal)]],
            device=device, dtype=torch.float32,
        )
        if zero_pointgoal:
            pointgoal.zero_()

        # Ego-pose
        poses = torch.tensor(
            [[ego_x, ego_y, math.cos(ego_theta), math.sin(ego_theta)]],
            device=device, dtype=torch.float32,
        )

        # Policy forward
        with torch.no_grad():
            dist, value, hidden, memory_buffer, memory_pose_buffer, memory_mask = policy.act(
                patch_embeds, cls_embed, goal_embed, prev_action, hidden, masks,
                pointgoal=pointgoal,
                memory_buffer=memory_buffer,
                memory_pose_buffer=memory_pose_buffer,
                memory_mask=memory_mask,
                poses=poses,
            )

        action = dist.sample().item()
        actions_taken.append(action)
        prev_action = torch.tensor([action], device=device)

        # Execute action
        action_map = {0: "MoveAhead", 1: "RotateLeft", 2: "RotateRight", 3: "Stop"}
        action_name = action_map[action]

        if action_name == "Stop":
            # Check success
            final_dist = math.sqrt(
                (agent_pos["x"] - goal_pos["x"]) ** 2 +
                (agent_pos["z"] - goal_pos["z"]) ** 2
            )
            success = final_dist < cfg.env.success_distance
            break

        event = controller.step(action=action_name)

        # Update dead-reckoned pose
        if action_name == "MoveAhead" and event.metadata["lastActionSuccess"]:
            ego_x += cfg.env.move_magnitude * math.sin(ego_theta)
            ego_y += cfg.env.move_magnitude * math.cos(ego_theta)
        elif action_name == "RotateLeft":
            ego_theta -= math.radians(cfg.env.rotate_degrees)
        elif action_name == "RotateRight":
            ego_theta += math.radians(cfg.env.rotate_degrees)

        # Record position
        agent_meta = event.metadata["agent"]
        pos = agent_meta["position"]
        trajectory.append((pos["x"], pos["z"]))
        headings.append(agent_meta["rotation"]["y"])
        masks = torch.ones(1, 1, device=device)

    return {
        "success": success,
        "trajectory": trajectory,
        "headings": headings,
        "actions": actions_taken,
        "num_steps": num_steps,
        "start_pos": (episode["initial_position"]["x"], episode["initial_position"]["z"]),
        "goal_pos": (goal_pos["x"], goal_pos["z"]),
        "goal_heading": goal_heading,
        "shortest_path": [(p["x"], p["z"]) for p in episode["shortest_path"]],
        "episode_id": ep_id,
    }


def plot_trajectory(result: Dict, output_path: str, dpi: int = 150) -> None:
    """Generate a bird's-eye view plot of the trajectory."""
    traj = np.array(result["trajectory"])
    shortest = np.array(result["shortest_path"])
    start = result["start_pos"]
    goal = result["goal_pos"]
    success = result["success"]
    headings = result["headings"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot shortest path (dashed gray)
    ax.plot(shortest[:, 0], shortest[:, 1], "k--", linewidth=1.5, alpha=0.4, label="Shortest path")

    # Plot agent trajectory with color gradient (blue -> red over time)
    if len(traj) > 1:
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(segments))
        lc = LineCollection(segments, cmap="coolwarm", norm=norm, linewidth=2.0, alpha=0.8)
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)

    # Plot heading arrows at intervals
    arrow_interval = max(1, len(traj) // 15)
    for i in range(0, len(traj), arrow_interval):
        heading_rad = math.radians(headings[i])
        dx = 0.15 * math.sin(heading_rad)
        dz = 0.15 * math.cos(heading_rad)
        t_frac = i / max(len(traj) - 1, 1)
        color = plt.cm.coolwarm(t_frac)
        ax.annotate(
            "", xy=(traj[i, 0] + dx, traj[i, 1] + dz),
            xytext=(traj[i, 0], traj[i, 1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    # Start marker
    ax.plot(start[0], start[1], "o", color="#2ecc71", markersize=14, zorder=5,
            markeredgecolor="white", markeredgewidth=2)
    ax.annotate("Start", (start[0], start[1]), textcoords="offset points",
                xytext=(10, 10), fontsize=10, fontweight="bold", color="#2ecc71")

    # Goal marker
    ax.plot(goal[0], goal[1], "*", color="#e74c3c", markersize=18, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate("Goal", (goal[0], goal[1]), textcoords="offset points",
                xytext=(10, 10), fontsize=10, fontweight="bold", color="#e74c3c")

    # Goal heading arrow
    goal_heading_rad = math.radians(result["goal_heading"])
    gdx = 0.3 * math.sin(goal_heading_rad)
    gdz = 0.3 * math.cos(goal_heading_rad)
    ax.annotate(
        "", xy=(goal[0] + gdx, goal[1] + gdz),
        xytext=(goal[0], goal[1]),
        arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.5),
    )

    # Final position marker
    final_pos = traj[-1]
    ax.plot(final_pos[0], final_pos[1], "s", color="#9b59b6", markersize=10, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)

    # Title and labels
    status = "SUCCESS" if success else "FAILURE"
    status_color = "#2ecc71" if success else "#e74c3c"
    ax.set_title(
        f"{result['episode_id']}\n{status} | {result['num_steps']} steps",
        fontsize=12, fontweight="bold", color=status_color,
    )
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Z (meters)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="k", linestyle="--", label="Shortest path"),
        plt.Line2D([0], [0], color="#3498db", linewidth=2, label="Agent path (early)"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=2, label="Agent path (late)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="Start"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e74c3c", markersize=12, label="Goal"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#9b59b6", markersize=8, label="Stop position"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Padding
    all_x = np.concatenate([traj[:, 0], shortest[:, 0], [start[0], goal[0]]])
    all_z = np.concatenate([traj[:, 1], shortest[:, 1], [start[1], goal[1]]])
    margin = 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_z.min() - margin, all_z.max() + margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {output_path}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    # Config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = Config()
    cfg.device = args.device

    # Determine episode list
    if args.episodes:
        episode_ids = args.episodes
    elif args.use_list:
        episode_ids = EPISODE_LIST
    else:
        print("ERROR: Provide --episodes or --use_list")
        sys.exit(1)

    # Load episodes
    all_episodes = load_all_episodes(cfg.env.episodes_path)
    episodes_map = {ep["id"]: ep for ep in all_episodes}

    selected_episodes = []
    for eid in episode_ids:
        if eid in episodes_map:
            selected_episodes.append(episodes_map[eid])
        else:
            logger.warning(f"Episode not found: {eid}")

    if not selected_episodes:
        print("ERROR: No valid episodes found")
        sys.exit(1)

    logger.info(f"Running {len(selected_episodes)} episodes")

    # Load checkpoint
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Build encoders
    obs_encoder = DINOv2Encoder(cfg, device=args.device)
    goal_encoder = CLIPGoalEncoder(cfg, device=args.device)

    # Build policy and load weights
    policy = build_policy(cfg, device=args.device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    # Load embeddings registry
    embeddings_path = os.path.join(cfg.env.scene_dataset_path, "embeddings.pt")
    embeddings_registry = torch.load(embeddings_path, map_location="cpu")

    # Initialize AI2-THOR controller
    from ai2thor.controller import Controller
    controller = Controller(
        agentMode="locobot",
        visibilityDistance=1.5,
        gridSize=cfg.env.move_magnitude,
        rotateStepDegrees=cfg.env.rotate_degrees,
        snapToGrid=False,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=cfg.env.image_width,
        height=cfg.env.image_height,
        fieldOfView=79,
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run episodes and generate visualizations
    results_summary = []
    for ep in selected_episodes:
        logger.info(f"Running episode: {ep['id']}")
        result = run_episode(
            controller=controller,
            episode=ep,
            policy=policy,
            obs_encoder=obs_encoder,
            goal_encoder=goal_encoder,
            embeddings_registry=embeddings_registry,
            cfg=cfg,
            device=device,
            max_steps=args.max_steps,
            zero_pointgoal=args.zero_pointgoal,
        )

        if "error" in result:
            logger.warning(f"  Skipped: {result['error']}")
            continue

        # Generate plot
        safe_name = ep["id"].replace("/", "_")
        output_path = os.path.join(args.output_dir, f"trajectory_{safe_name}.png")
        plot_trajectory(result, output_path, dpi=args.dpi)

        status = "SUCCESS" if result["success"] else "FAILURE"
        logger.info(f"  {status} | {result['num_steps']} steps")
        results_summary.append(result)

    # Print summary
    controller.stop()
    n_success = sum(1 for r in results_summary if r["success"])
    print(f"\n{'='*50}")
    print(f"  Episodes: {len(results_summary)} | Success: {n_success}/{len(results_summary)}")
    print(f"  Output:   {args.output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
