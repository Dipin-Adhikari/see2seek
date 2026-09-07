"""
visualize_trajectory.py — Bird's-eye-view trajectory visualization using
AI2-THOR's top-down rendered map view.

Runs the trained agent through specified episodes, records positions,
then captures the RoboTHOR top-down camera view and overlays the trajectory.

NEW: each episode can be rolled out multiple times (--num_trials, default 5)
and all trials are overlaid on the SAME top-down frame — useful since the
policy is stochastic (dist.sample()) and sometimes fails, so you can see the
spread of behavior for one episode in a single image.

Usage:
    python scripts/visualize_trajectory.py \
        --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth \
        --episodes FloorPlan_Val3_2_Apple_6 \
        --episodes_path /home/dipin/See2Seek/imagenav_dataset/val/episodes \
        --scene_dataset_path /home/dipin/See2Seek/imagenav_dataset/val \
        --num_trials 5

    # Multiple episodes
    python scripts/visualize_trajectory.py \
        --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth \
        --use_list \
        --episodes_path /home/doece5/see2seek_dipin_adhikari/see2seek/dataset/val/episodes \
        --scene_dataset_path /home/doece5/see2seek_dipin_adhikari/see2seek/dataset/val \
        --num_trials 5

    # Without pointgoal
    python scripts/visualize_trajectory.py \
        --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth \
        --episodes FloorPlan_Val3_2_Apple_6 \
        --episodes_path /home/dipin/See2Seek/imagenav_dataset/val/episodes \
        --scene_dataset_path /home/dipin/See2Seek/imagenav_dataset/val \
        --zero_pointgoal --num_trials 5
"""

import argparse
import gzip
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

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
    "FloorPlan_Val2_1_SprayBottle_4",
    "FloorPlan_Val3_4_BasketBall_5",
    "FloorPlan_Val3_4_AlarmClock_2",
    "FloorPlan_Val3_4_Apple_9",
    "FloorPlan_Val2_2_Apple_2",
    "FloorPlan_Val2_2_Laptop_5",
    "FloorPlan_Val3_4_SprayBottle_2",
    "FloorPlan_Val1_1_Mug_0",
    "FloorPlan_Val2_2_Laptop_9",
    "FloorPlan_Val3_4_Bowl_1",
    "FloorPlan_Val3_2_AlarmClock_7",
    "FloorPlan_Val3_4_Apple_2",
    "FloorPlan_Val3_4_Mug_3",
    "FloorPlan_Val2_3_Vase_2",
    "FloorPlan_Val3_2_AlarmClock_1",
    "FloorPlan_Val2_5_HousePlant_1",
    "FloorPlan_Val3_5_Mug_1",
    "FloorPlan_Val3_4_SprayBottle_0",
    "FloorPlan_Val3_2_AlarmClock_3",
    "FloorPlan_Val2_2_Mug_1",
    "FloorPlan_Val3_4_Apple_4",
    "FloorPlan_Val1_1_HousePlant_5",
    "FloorPlan_Val3_4_BasketBall_8",
    "FloorPlan_Val2_2_Apple_7",
    "FloorPlan_Val2_3_Mug_0",
    "FloorPlan_Val1_5_AlarmClock_5",
    "FloorPlan_Val3_3_Apple_3",
    "FloorPlan_Val3_1_GarbageCan_7",
    "FloorPlan_Val1_2_HousePlant_9",
    "FloorPlan_Val3_1_GarbageCan_0",
    "FloorPlan_Val2_3_HousePlant_2",
    "FloorPlan_Val3_4_SprayBottle_7",
    "FloorPlan_Val3_4_GarbageCan_7",
    "FloorPlan_Val2_2_Vase_2",
    "FloorPlan_Val3_4_Bowl_8",
    "FloorPlan_Val3_4_Bowl_8",
    "FloorPlan_Val3_5_BasketBall_4",
    "FloorPlan_Val3_1_Television_0",
    "FloorPlan_Val1_1_HousePlant_4",
    "FloorPlan_Val3_1_GarbageCan_7",
    "FloorPlan_Val1_1_HousePlant_3",
    "FloorPlan_Val2_3_Mug_4",
    "FloorPlan_Val1_1_Laptop_2",
    "FloorPlan_Val2_4_GarbageCan_2",
    "FloorPlan_Val3_2_BasketBall_2",
    "FloorPlan_Val1_4_AlarmClock_8",
    "FloorPlan_Val3_4_Bowl_3",
    "FloorPlan_Val3_4_AlarmClock_5",
    "FloorPlan_Val1_4_Bowl_5",
    "FloorPlan_Val1_5_Television_0",
    "FloorPlan_Val1_2_Apple_0",
    "FloorPlan_Val2_5_HousePlant_4",
    "FloorPlan_Val1_3_Apple_1",
    "FloorPlan_Val3_4_AlarmClock_7",
    "FloorPlan_Val3_4_SprayBottle_7",
    "FloorPlan_Val2_3_HousePlant_2",
    "FloorPlan_Val3_4_HousePlant_6",
    "FloorPlan_Val1_2_SprayBottle_2",
    "FloorPlan_Val1_3_Laptop_1",
    "FloorPlan_Val1_2_Apple_7",
    "FloorPlan_Val1_2_Apple_7",
    "FloorPlan_Val2_5_HousePlant_1",
    "FloorPlan_Val1_1_Laptop_9",
    "FloorPlan_Val2_3_Vase_2",
    "FloorPlan_Val2_4_GarbageCan_2",
    "FloorPlan_Val2_5_SprayBottle_2",
]


def parse_args():
    p = argparse.ArgumentParser(description="See2Seek — Bird's-Eye Trajectory Visualization")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config", default=None, help="Path to YAML config override")
    p.add_argument("--episodes", nargs="+", default=None, help="Episode IDs to visualize")
    p.add_argument("--use_list", action="store_true", help="Use hardcoded EPISODE_LIST")
    p.add_argument("--output_dir", default="data_dino_v7/visualizations", help="Output directory")
    p.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--task", default="imagenav", choices=["imagenav", "objectnav"],
                   help="Task type: imagenav (goal image) or objectnav (object category)")
    p.add_argument("--zero_pointgoal", action="store_true", help="Zero out PointGoal sensor")
    p.add_argument("--episodes_path", default=None, help="Override episodes directory path")
    p.add_argument("--scene_dataset_path", default=None, help="Override scene dataset path")
    p.add_argument("--split", default=None, choices=["train", "val"], help="Override split")
    p.add_argument("--topdown_size", type=int, default=800, help="Top-down render resolution")
    p.add_argument("--save_frames", type=int, default=5, help="Save last X observation frames before Stop")
    p.add_argument(
        "--num_trials", type=int, default=5,
        help="Number of times to roll out each episode; all trials are overlaid on one top-down image",
    )
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


class TopDownProjector:
    """Converts AI2-THOR world (x, z) coordinates into pixel coordinates on
    an orthographic top-down camera frame."""

    def __init__(self, frame_shape, cam_position, orthographic_size):
        self.h, self.w = frame_shape[0], frame_shape[1]
        self.cam_x = cam_position["x"]
        self.cam_z = cam_position["z"]
        self.orth_size = orthographic_size
        self.min_x = self.cam_x - self.orth_size
        self.min_z = self.cam_z - self.orth_size
        self.span = 2.0 * self.orth_size

    def to_pixel(self, x: float, z: float) -> Tuple[int, int]:
        norm_x = (x - self.min_x) / self.span
        norm_z = (z - self.min_z) / self.span
        px = int(round(norm_x * self.w))
        py = int(round((1.0 - norm_z) * self.h))
        return px, py


def get_topdown_frame(
    controller, topdown_size: int = 800
) -> Tuple[np.ndarray, TopDownProjector]:
    """Capture a top-down rendered view using AI2-THOR's orthographic camera."""
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    cam_props = dict(event.metadata["actionReturn"])

    orth_size = cam_props.get("orthographicSize")
    if orth_size is None:
        raise KeyError(f"No orthographicSize in cam_props: {cam_props}")

    cam_props["orthographic"] = True
    cam_props["farClippingPlane"] = 50
    cam_props["skyboxColor"] = "white"

    event = controller.step(action="AddThirdPartyCamera", **cam_props)

    frame = event.third_party_camera_frames[0]
    projector = TopDownProjector(frame.shape, cam_props["position"], orth_size)

    return frame.copy(), projector


CATEGORY_MAP = {
    "AlarmClock": "alarm clock",
    "Apple": "apple",
    "BaseballBat": "baseball bat",
    "BasketBall": "basketball",
    "Bowl": "bowl",
    "GarbageCan": "garbage can",
    "HousePlant": "house plant",
    "Laptop": "laptop",
    "Mug": "mug",
    "SprayBottle": "spray bottle",
    "Television": "television",
    "Vase": "vase",
}


def build_objectnav_embeddings(goal_encoder: CLIPGoalEncoder) -> Dict[str, torch.Tensor]:
    """Pre-compute CLIP text embeddings for all RoboTHOR ObjectNav categories."""
    registry = {}
    for obj_type, text in CATEGORY_MAP.items():
        with torch.no_grad():
            registry[obj_type] = goal_encoder.encode_text(text).squeeze(0).cpu()
    logger.info(f"Built {len(registry)} ObjectNav text embeddings")
    return registry


def _resolve_goal_embedding(
    episode: Dict,
    embeddings_registry: Dict,
    split: str,
    task: str = "imagenav",
) -> Optional[torch.Tensor]:
    """Look up the goal embedding."""
    if task == "objectnav":
        obj_type = episode.get("object_type", "")
        if obj_type in embeddings_registry:
            return embeddings_registry[obj_type]
        return None

    goal_image_path = episode.get("goal_image_path", "")
    filename = os.path.basename(goal_image_path)
    lookup_key = f"{split}/images/{filename}"

    if lookup_key in embeddings_registry:
        return embeddings_registry[lookup_key]

    ep_id = episode.get("id", "")
    if ep_id in embeddings_registry:
        return embeddings_registry[ep_id]

    return None


def run_episode(
    controller,
    episode: Dict,
    policy,
    obs_encoder,
    embeddings_registry: Dict,
    cfg,
    device: torch.device,
    max_steps: int = 500,
    zero_pointgoal: bool = False,
    save_frames: int = 5,
    task: str = "imagenav",
) -> Dict:
    """Run a single episode and return trajectory data."""
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
    ep_id = episode["id"]
    optimal_pose = episode.get("optimal_goal_pose", {})
    goal_heading = optimal_pose.get("rotation", 0.0)

    goal_embed_tensor = _resolve_goal_embedding(
        episode, embeddings_registry, cfg.env.split, task=task
    )
    if goal_embed_tensor is None:
        return {"success": False, "error": f"Goal embedding not found for {ep_id}"}
    goal_embed = goal_embed_tensor.unsqueeze(0).to(device)

    obs_encoder_type = getattr(cfg.encoder, "obs_encoder_type", "dino")
    with_pointgoal = getattr(cfg.encoder, "with_pointgoal", False)

    hidden = policy.get_initial_hidden(1, device)
    cls_dim = (
        cfg.encoder.dino_cls_dim if obs_encoder_type == "dino"
        else cfg.encoder.goal_embed_dim
    )
    memory_buffer = torch.zeros(
        1, policy.memory_size, cls_dim, device=device
    )
    memory_pose_buffer = torch.zeros(
        1, policy.memory_size, EpisodicMemory.POSE_DIM, device=device
    )
    memory_mask = torch.zeros(1, policy.memory_size, device=device, dtype=torch.bool)
    prev_action = torch.tensor([cfg.env.num_actions], device=device)
    masks = torch.ones(1, 1, device=device)

    ego_x, ego_y, ego_theta = 0.0, 0.0, 0.0

    trajectory = []
    actions_taken = []
    headings = []
    recent_frames = []

    agent_meta = event.metadata["agent"]
    pos = agent_meta["position"]
    trajectory.append((pos["x"], pos["z"]))
    headings.append(agent_meta["rotation"]["y"])

    success = False
    num_steps = 0

    for step in range(max_steps):
        num_steps = step + 1

        frame = event.frame
        recent_frames.append(frame.copy())
        if len(recent_frames) > save_frames:
            recent_frames.pop(0)

        frame_pil = Image.fromarray(frame)
        frame_resized = frame_pil.resize(
            (cfg.env.image_width, cfg.env.image_height), Image.BILINEAR
        )
        rgb_tensor = (
            torch.from_numpy(np.array(frame_resized))
            .permute(2, 0, 1)
            .float() / 255.0
        )
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            if obs_encoder_type == "dino":
                cls_embed, patch_embeds = obs_encoder.get_all_embeddings(rgb_tensor)
            else:
                cls_embed = obs_encoder.get_obs_embedding(rgb_tensor)
                patch_embeds = None

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
        if zero_pointgoal or not with_pointgoal or task == "objectnav":
            pointgoal = None

        poses = torch.tensor(
            [[ego_x, ego_y, math.cos(ego_theta), math.sin(ego_theta)]],
            device=device, dtype=torch.float32,
        )

        can_stop = torch.tensor(
            [num_steps >= cfg.env.min_steps_before_stop], device=device
        )

        with torch.no_grad():
            dist, value, hidden, memory_buffer, memory_pose_buffer, memory_mask = (
                policy.act(
                    patch_embeds, cls_embed, goal_embed, prev_action, hidden, masks,
                    pointgoal=pointgoal,
                    can_stop=can_stop,
                    memory_buffer=memory_buffer,
                    memory_pose_buffer=memory_pose_buffer,
                    memory_mask=memory_mask,
                    poses=poses,
                )
            )

        action = dist.sample().item()
        actions_taken.append(action)
        prev_action = torch.tensor([action], device=device)

        action_map = {0: "MoveAhead", 1: "RotateLeft", 2: "RotateRight", 3: "Stop"}
        action_name = action_map[action]

        if action_name == "Stop":
            final_dist = math.sqrt(
                (agent_pos["x"] - goal_pos["x"]) ** 2
                + (agent_pos["z"] - goal_pos["z"]) ** 2
            )
            success = final_dist < cfg.env.success_distance
            break

        event = controller.step(action=action_name)

        if action_name == "MoveAhead" and event.metadata["lastActionSuccess"]:
            ego_x += cfg.env.move_magnitude * math.sin(ego_theta)
            ego_y += cfg.env.move_magnitude * math.cos(ego_theta)
        elif action_name == "RotateLeft":
            ego_theta -= math.radians(cfg.env.rotate_degrees)
        elif action_name == "RotateRight":
            ego_theta += math.radians(cfg.env.rotate_degrees)

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
        "recent_frames": recent_frames,
        "start_pos": (episode["initial_position"]["x"], episode["initial_position"]["z"]),
        "goal_pos": (goal_pos["x"], goal_pos["z"]),
        "goal_heading": goal_heading,
        "shortest_path": [(p["x"], p["z"]) for p in episode["shortest_path"]],
        "episode_id": ep_id,
        "scene": scene,
    }


def run_episode_multi(
    controller,
    episode: Dict,
    policy,
    obs_encoder,
    embeddings_registry: Dict,
    cfg,
    device: torch.device,
    num_trials: int = 5,
    max_steps: int = 500,
    zero_pointgoal: bool = False,
    save_frames: int = 5,
    task: str = "imagenav",
) -> List[Dict]:
    """Roll out the SAME episode `num_trials` times (policy is stochastic via
    dist.sample(), so trials can differ — and sometimes fail). Returns a list
    of per-trial result dicts (same shape as run_episode's return value),
    skipping any trial that errors out (e.g. missing goal embedding, failed
    teleport) but logging it.
    """
    trials = []
    for t in range(num_trials):
        result = run_episode(
            controller=controller,
            episode=episode,
            policy=policy,
            obs_encoder=obs_encoder,
            embeddings_registry=embeddings_registry,
            cfg=cfg,
            device=device,
            max_steps=max_steps,
            zero_pointgoal=zero_pointgoal,
            save_frames=save_frames,
            task=task,
        )
        if "error" in result:
            logger.warning(f"  Trial {t + 1}/{num_trials} skipped: {result['error']}")
            continue
        status = "SUCCESS" if result["success"] else "FAILURE"
        logger.info(f"  Trial {t + 1}/{num_trials}: {status} | {result['num_steps']} steps")
        trials.append(result)
    return trials


def plot_trajectory_on_topdown(
    result: Dict,
    topdown_frame: np.ndarray,
    projector: TopDownProjector,
    output_path: str,
) -> None:
    """Overlay a single trajectory on the AI2-THOR rendered top-down view.
    Kept for backwards compatibility / single-trial use.
    """
    plot_multi_trajectory_on_topdown([result], topdown_frame, projector, output_path)


def plot_multi_trajectory_on_topdown(
    results: List[Dict],
    topdown_frame: np.ndarray,
    projector: TopDownProjector,
    output_path: str,
) -> None:
    """Overlay N trajectories (from repeated rollouts of the same episode) on
    ONE top-down frame. Same color scheme as the original single-trial plot:
      - dark green:            shortest (ground-truth) path, drawn once
      - light green (100,255,100): each SUCCESSFUL trial's path
      - red (255,60,60):       each FAILED trial's path
      - white/green ring:      start marker, drawn once
      - red/black ring:        goal marker, drawn once
      - filled circle:         each trial's final position, in that trial's color
    """
    if not results:
        logger.warning("plot_multi_trajectory_on_topdown called with no results")
        return

    shortest = results[0]["shortest_path"]
    start = results[0]["start_pos"]
    goal = results[0]["goal_pos"]

    img = Image.fromarray(topdown_frame).convert("RGB")
    draw = ImageDraw.Draw(img)

    dark_green = (0, 120, 0)
    success_color = (100, 255, 100)
    failure_color = (255, 60, 60)

    # Ground-truth shortest path — drawn once, same for every trial.
    shortest_px = [projector.to_pixel(x, z) for x, z in shortest]
    if len(shortest_px) > 1:
        for i in range(len(shortest_px) - 1):
            draw.line([shortest_px[i], shortest_px[i + 1]], fill=dark_green, width=5)

    # One line per trial, colored by that trial's own success/failure.
    n_success = 0
    for result in results:
        traj_px = [projector.to_pixel(x, z) for x, z in result["trajectory"]]
        path_color = success_color if result["success"] else failure_color
        n_success += int(result["success"])
        if len(traj_px) > 1:
            for i in range(len(traj_px) - 1):
                draw.line([traj_px[i], traj_px[i + 1]], fill=path_color, width=5)

        # Final position marker for this trial.
        fx, fy = traj_px[-1]
        r = 8
        draw.ellipse(
            [fx - r, fy - r, fx + r, fy + r],
            fill=path_color, outline=(255, 255, 255), width=2,
        )

    # Start / goal markers — drawn last so they sit on top of every path.
    r = 10
    start_px = projector.to_pixel(start[0], start[1])
    sx, sy = start_px
    draw.ellipse(
        [sx - r, sy - r, sx + r, sy + r],
        fill=(255, 255, 255), outline=dark_green, width=3,
    )

    goal_px = projector.to_pixel(goal[0], goal[1])
    gx, gy = goal_px
    draw.ellipse(
        [gx - r, gy - r, gx + r, gy + r],
        fill=(220, 20, 60), outline=(0, 0, 0), width=3,
    )

    img.save(output_path)
    logger.info(
        f"Saved: {output_path} ({n_success}/{len(results)} trials succeeded)"
    )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "cfg" in checkpoint and args.config is None:
        cfg = checkpoint["cfg"]
        logger.info("Using config from checkpoint")
    elif args.config:
        cfg = load_config(args.config)
    else:
        cfg = Config()

    cfg.device = args.device

    if args.episodes_path:
        cfg.env.episodes_path = args.episodes_path
    elif not os.path.exists(cfg.env.episodes_path):
        local_cfg = Config()
        cfg.env.episodes_path = local_cfg.env.episodes_path
        cfg.env.scene_dataset_path = local_cfg.env.scene_dataset_path
        logger.info(f"Checkpoint paths not found, using local: {cfg.env.episodes_path}")

    if args.scene_dataset_path:
        cfg.env.scene_dataset_path = args.scene_dataset_path
        inferred_split = os.path.basename(os.path.normpath(args.scene_dataset_path))
        if inferred_split in ("train", "val", "test", "debug"):
            cfg.env.split = inferred_split
            logger.info(f"Inferred split='{inferred_split}' from scene_dataset_path")

    if args.split:
        cfg.env.split = args.split
        base = os.path.dirname(os.path.dirname(cfg.env.episodes_path))
        cfg.env.episodes_path = os.path.join(base, args.split, "episodes")
        cfg.env.scene_dataset_path = os.path.join(base, args.split)

    if args.episodes:
        episode_ids = args.episodes
    elif args.use_list:
        episode_ids = EPISODE_LIST
    else:
        print("ERROR: Provide --episodes or --use_list")
        sys.exit(1)

    all_episodes = load_all_episodes(cfg.env.episodes_path)
    episodes_map = {ep["id"]: ep for ep in all_episodes}

    selected_episodes = []
    for eid in episode_ids:
        if eid in episodes_map:
            selected_episodes.append(episodes_map[eid])
        else:
            logger.warning(f"Episode not found: {eid}")

    if not selected_episodes:
        print("ERROR: No valid episodes found. Available IDs (first 10):")
        for eid in list(episodes_map.keys())[:10]:
            print(f"  {eid}")
        sys.exit(1)

    logger.info(f"Running {len(selected_episodes)} episodes x {args.num_trials} trials each")

    obs_encoder_type = getattr(cfg.encoder, "obs_encoder_type", "dino")
    if obs_encoder_type == "clip":
        obs_encoder = CLIPGoalEncoder(device=args.device)
    else:
        obs_encoder = DINOv2Encoder(device=args.device)

    goal_encoder = CLIPGoalEncoder(device=args.device)

    policy = build_policy(cfg, device=args.device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    logger.info(f"Policy loaded — input_dim={policy.policy_input_dim}")

    if args.task == "objectnav":
        embeddings_registry = build_objectnav_embeddings(goal_encoder)
    else:
        embeddings_path = os.path.join(cfg.env.scene_dataset_path, "embeddings.pt")
        embeddings_registry = torch.load(
            embeddings_path, map_location="cpu", weights_only=False
        )
        logger.info(f"Loaded {len(embeddings_registry)} goal embeddings")

    from ai2thor.controller import Controller

    render_size = args.topdown_size
    controller = Controller(
        agentMode="locobot",
        visibilityDistance=1.5,
        gridSize=cfg.env.move_magnitude,
        rotateStepDegrees=cfg.env.rotate_degrees,
        snapToGrid=False,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=render_size,
        height=render_size,
        fieldOfView=79,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    episode_summaries = []
    for ep in selected_episodes:
        logger.info(f"Running episode: {ep['id']} ({args.num_trials} trials)")

        trial_results = run_episode_multi(
            controller=controller,
            episode=ep,
            policy=policy,
            obs_encoder=obs_encoder,
            embeddings_registry=embeddings_registry,
            cfg=cfg,
            device=device,
            num_trials=args.num_trials,
            max_steps=args.max_steps,
            zero_pointgoal=args.zero_pointgoal,
            save_frames=args.save_frames,
            task=args.task,
        )

        if not trial_results:
            logger.warning(f"  All trials skipped for {ep['id']}")
            continue

        logger.info("  Capturing top-down view...")
        topdown_frame, projector = get_topdown_frame(controller, args.topdown_size)

        safe_name = ep["id"].replace("/", "_")
        output_path = os.path.join(args.output_dir, f"trajectory_{safe_name}.png")
        plot_multi_trajectory_on_topdown(trial_results, topdown_frame, projector, output_path)

        # Frames are only saved for the LAST trial run (kept simple — flip to
        # saving per-trial frame dirs if you need that too).
        last = trial_results[-1]
        if last["recent_frames"]:
            frames_dir = os.path.join(args.output_dir, f"frames_{safe_name}")
            os.makedirs(frames_dir, exist_ok=True)
            n_frames = len(last["recent_frames"])
            stop_step = last["num_steps"]
            for i, frame in enumerate(last["recent_frames"]):
                step_num = stop_step - n_frames + i + 1
                frame_path = os.path.join(frames_dir, f"step_{step_num:03d}.png")
                Image.fromarray(frame).save(frame_path)
            logger.info(f"  Saved {n_frames} frames (last trial) to {frames_dir}/")

        n_success = sum(1 for r in trial_results if r["success"])
        logger.info(
            f"  Episode {ep['id']}: {n_success}/{len(trial_results)} trials succeeded"
        )
        episode_summaries.append(
            {"episode_id": ep["id"], "n_trials": len(trial_results), "n_success": n_success}
        )

    controller.stop()

    total_trials = sum(s["n_trials"] for s in episode_summaries)
    total_success = sum(s["n_success"] for s in episode_summaries)
    print(f"\n{'=' * 50}")
    print(f"  Episodes: {len(episode_summaries)}")
    for s in episode_summaries:
        print(f"    {s['episode_id']}: {s['n_success']}/{s['n_trials']} succeeded")
    print(f"  Total trials: {total_trials} | Success: {total_success}/{total_trials}")
    print(f"  Output:   {args.output_dir}/")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()