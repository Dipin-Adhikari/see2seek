"""
Wraps AI2-THOR's Controller into a standard gym.Env interface with:
    - Discrete action space (MoveAhead, RotateLeft, RotateRight, Stop)
    - Geodesic-distance-based dense reward shaping
    - Automatic episode reset with goal image embedding loading

Episode structure:
    Each episode has:
        - A start position / rotation for the agent
        - A goal image embedding(pre-cached)
        - Success if the agent calls Stop within `success_distance` of goal

Reward structure:
    r_t = Δ(geodesic_distance) * scale + slack_reward
    r_T = success_reward  (if Stop is called and agent is at goal)
    r_T = 0               (if Stop is called but agent is not at goal)

    Geodesic-distance delta reward encourages movement toward the goal
    rather than rewarding visual similarity (which is encoded separately
    in the observation embedding).

Usage:
    env = RoboTHOREnv(cfg)
    obs_dict = env.reset()
    obs_dict, reward, done, info = env.step(action_int)
    env.close()
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ===========================================================================
# Action map
# ===========================================================================
ACTIONS = {
    0: "MoveAhead",
    1: "RotateLeft",
    2: "RotateRight",
    3: "Stop",
}

# ===========================================================================
# Environment
# ===========================================================================
class RoboTHOREnv:
    """
    Single-process RoboTHOR environment optimized for Pre-cached CLIP Embeddings.

    Args:
        cfg:        configs.config.Config — full project config.
        episode_ids: Optional list of episode IDs to restrict execution.
        worker_id:  Integer offset for AI2-THOR server port management.
        render:     If True, render frames to screen (slow, for debugging).
    """

    def __init__(
        self,
        cfg,
        episode_ids: Optional[List[str]] = None,
        worker_id: int = 0,
        render: bool = False,
    ) -> None:
        self.cfg = cfg
        self._worker_id = worker_id
        self._render = render

        # Observation space transform (only applied to the agent's live camera feed)
        self._transform = transforms.Compose([
            transforms.Resize((cfg.env.image_height, cfg.env.image_width)),
            transforms.ToTensor(),
        ])

        # 1. Grab split dynamically from config
        self._split = cfg.env.split

        # 2. Load the pre-cached embeddings dictionary map (.pt)
        # scene_dataset_path IS the split folder (e.g. .../imagenav_dataset/debug)
        embeddings_path = os.path.join(cfg.env.scene_dataset_path, "embeddings.pt")

        if os.path.exists(embeddings_path):
            logger.info(f"📂 Loading cached CLIP embeddings registry from: {embeddings_path}")
            self._embeddings_registry = torch.load(embeddings_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"❌ Required pre-cached embeddings file missing at: {embeddings_path}")

        # 3. Load episodes. episodes_path may point at:
        #      - a single .json / .json.gz file, OR
        #      - a directory containing one or more per-scene files
        #    (see _load_episodes for directory-handling / de-dup logic)
        self._episodes: List[Dict] = self._load_episodes(
            cfg.env.episodes_path, episode_ids
        )

        self._episode_index: int = 0
        random.shuffle(self._episodes)

        # Current episode state variables
        self._current_episode: Optional[Dict] = None
        self._cached_goal_embedding: Optional[torch.Tensor] = None
        self._prev_geodesic_dist: float = 0.0
        self._num_steps: int = 0
        self._episode_collisions: int = 0  # FIX #2: initialize here (was previously undefined)

        # Lazy-initialise AI2-THOR backend controller
        self._controller = None
        self._init_controller()

        logger.info(
            f"✔ RoboTHOREnv[{worker_id}] initialized successfully — {len(self._episodes)} episodes running."
        )

    def _init_controller(self) -> None:
        """Start the AI2-THOR Unity engine backend wrapper."""
        try:
            from ai2thor.controller import Controller
            # Import CloudRendering to support true headless execution if needed
            from ai2thor.platform import CloudRendering
        except ImportError as e:
            raise ImportError("ai2thor is required: pip install ai2thor") from e

        # Use CloudRendering if headless=True, otherwise let it open normally
        platform_setting = CloudRendering if (not self._render) else None

        self._controller = Controller(
            agentMode="locobot",  # must be "locobot" for RoboTHOR scenes
            visibilityDistance=1.5,
            #scene="FloorPlan_Train1_1",
            gridSize=self.cfg.env.move_magnitude,
            rotateStepDegrees=self.cfg.env.rotate_degrees,
            snapToGrid=False,
            renderDepthImage=self.cfg.env.depth_sensor,
            renderInstanceSegmentation=False,
            width=self.cfg.env.image_width,
            height=self.cfg.env.image_height,
            fieldOfView=79,
            port=8200 + self._worker_id,
            # headless=(not self._render),
            # gpu_device=0,
            # platform=platform_setting,
        )

    def _load_episodes(
        self,
        path: str,
        episode_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Load episode definitions from either:
          - a single .json / .json.gz file, or
          - a directory containing one or more per-scene .json/.json.gz files.

        When a directory is given and a scene has BOTH a .json and a .json.gz
        version (as in the debug set), only the .json.gz is loaded so episodes
        aren't duplicated.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Episodes path not found: {path}")

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
                raise ValueError(f"Unexpected JSON data format in {fp}. Expected list or dict.")

        episodes: List[Dict] = []

        if os.path.isdir(path):
            # stem -> filepath, preferring .json.gz over .json for the same stem
            files: Dict[str, str] = {}
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".json.gz"):
                    stem = fname[: -len(".json.gz")]
                    files[stem] = os.path.join(path, fname)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".json") and not fname.endswith(".json.gz"):
                    stem = fname[: -len(".json")]
                    files.setdefault(stem, os.path.join(path, fname))  # skip if .gz already claimed it

            if not files:
                raise FileNotFoundError(f"No .json/.json.gz episode files found in directory: {path}")

            for fp in files.values():
                episodes.extend(_read_one(fp))
        else:
            episodes.extend(_read_one(path))

        # Filter by specific IDs if requested
        if episode_ids is not None:
            episode_ids_set = set(episode_ids)
            episodes = [e for e in episodes if e.get("id") in episode_ids_set]

        if len(episodes) == 0:
            raise ValueError(f"No valid episodes found in {path}")

        return episodes

    # =======================================================================
    # Gym Environment Core Interface Methods
    # =======================================================================
    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Reset to a new dataset scenario trajectory.

        Returns:
            obs: dict tracking keys:
                "rgb":  (3, H, W) live camera float view tensor
                "goal": (512,) precalculated float embedding target vector
        """
        if self._episode_index >= len(self._episodes):
            random.shuffle(self._episodes)
            self._episode_index = 0

        self._current_episode = self._episodes[self._episode_index]
        self._episode_index += 1
        self._num_steps = 0
        self._episode_collisions = 0  # FIX #2: reset the per-episode collision counter

        ep = self._current_episode
        scene = ep["scene"]

        # 1. Reset simulator framework window state target
        event = self._controller.reset(scene=scene)

        # CloudRendering can return a None frame on the very first event while
        # the render server is still warming up (common on weaker/laptop GPUs).
        # Force a few no-op steps until a real frame comes back.
        retries = 0
        max_retries = 5
        while event.frame is None and retries < max_retries:
            event = self._controller.step(action="Pass")
            retries += 1

        if event.frame is None:
            raise RuntimeError(
                f"Renderer failed to produce a frame after {retries} retries "
                f"(scene={scene}). Check CloudRendering / GPU setup."
            )

        # 2. Build rotation payload matching AI2-THOR API parameters (Yaw mapping)
        start_rotation = {"x": 0, "y": ep.get("initial_orientation", 0.0), "z": 0}

        event = self._controller.step(
            action="TeleportFull",
            position=ep["initial_position"],
            rotation=start_rotation,
            horizon=ep.get("initial_horizon", 0),
        )

        if not event.metadata.get("lastActionSuccess", True):
            raise RuntimeError(
                f"TeleportFull failed for scene={scene}, "
                f"position={ep['initial_position']}, rotation={start_rotation}: "
                f"{event.metadata.get('errorMessage')}"
            )

        if event.frame is None:
            raise RuntimeError(
                f"TeleportFull succeeded but returned no frame (scene={scene})"
            )

        # 3. Calculate distance tracking metrics relative to final waypoint destination coordinate proxy
        goal_pos = ep["shortest_path"][-1]
        self._prev_geodesic_dist = self._get_geodesic_distance(goal_pos)

        # 4. Fetch the environment state observations
        rgb = self._get_rgb_tensor(event.frame)
        goal_embedding = self._load_goal_embedding(ep)

        return {"rgb": rgb, "goal": goal_embedding}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """Executes a control navigation step command."""
        assert 0 <= action < len(ACTIONS), f"Action index bounds error: {action}"

        self._num_steps += 1
        action_name = ACTIONS[action]

        if action_name == "Stop":
            done, success = self._handle_stop()
            reward = self.cfg.env.success_reward if success else 0.0
            info = self._build_info(success, done)
            obs = {
                "rgb": self._get_rgb_tensor(self._controller.last_event.frame),
                "goal": self._get_cached_goal(),
            }
            return obs, reward, done, info

        event = self._controller.step(action=action_name)

        if not event.metadata.get("lastActionSuccess", True):
            logger.warning(
                f"Action '{action_name}' failed at step {self._num_steps} "
                f"(episode={self._current_episode.get('id', '?')}): "
                f"{event.metadata.get('errorMessage')}"
            )

        goal_pos = self._current_episode["shortest_path"][-1]
        curr_dist = self._get_geodesic_distance(goal_pos)

        reward = (self._prev_geodesic_dist - curr_dist) * self.cfg.env.geodesic_reward_scale
        reward += self.cfg.env.slack_reward

        if action_name in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"} and event.metadata.get("collided", False):
            reward += self.cfg.env.collision_penalty
            self._episode_collisions += 1

        self._prev_geodesic_dist = curr_dist

        done = self._num_steps >= self.cfg.env.max_steps
        info = self._build_info(success=False, done=done)

        obs = {"rgb": self._get_rgb_tensor(event.frame), "goal": self._get_cached_goal()}
        return obs, reward, done, info

    def close(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller = None

    # =======================================================================
    # Internal Pipeline Helper Functions
    # =======================================================================
    def _handle_stop(self) -> Tuple[bool, bool]:
        """Validates stopping threshold distance criteria against the final target path node."""
        goal_pos = self._current_episode["shortest_path"][-1]
        agent_pos = self._controller.last_event.metadata["agent"]["position"]

        dist = np.linalg.norm([
            agent_pos["x"] - goal_pos["x"],
            agent_pos["z"] - goal_pos["z"],
        ])
        success = dist <= self.cfg.env.success_distance
        return True, success

    def _get_geodesic_distance(self, goal_pos: Dict) -> float:
        """Computes current Euclidean distance metrics over the active horizontal XZ traversal layout plane."""
        agent_pos = self._controller.last_event.metadata["agent"]["position"]
        dist = np.linalg.norm([
            agent_pos["x"] - goal_pos["x"],
            agent_pos["z"] - goal_pos["z"],
        ])
        return float(dist)

    def _get_rgb_tensor(self, frame: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Convert a raw AI2-THOR RGB frame (HxWx3 uint8 array) into a
        normalized model-ready tensor.

        Args:
            frame: Optional frame array. If not provided, falls back to
                self._controller.last_event.frame (kept for backward
                compatibility with any other call sites).
        """
        if frame is None:
            frame = self._controller.last_event.frame

        if frame is None:
            raise RuntimeError(
                "Controller returned no frame — renderer may not be initialised "
                "or the last action may have failed silently."
            )

        pil = Image.fromarray(frame)
        return self._transform(pil)

    def _load_goal_embedding(self, ep: Dict) -> torch.Tensor:
        """
        Resolves embedding references against the registry map.

        Confirmed key format (verified against embeddings.pt on 2026-07-10):
            "<split>/images/<basename of goal_image_path>"
        e.g. "debug/images/id_000000_FloorPlan_Train1_1_..._goal.png"

        Requires cfg.env.split to match the actual split folder name
        (e.g. "debug") — the default in config.py is "train", so make sure
        it's overridden for debug runs or this will raise KeyError below.
        """
        filename = os.path.basename(ep.get("goal_image_path", ""))
        lookup_key = f"{self._split}/images/{filename}"

        if lookup_key not in self._embeddings_registry:
            raise KeyError(
                f"Goal embedding key '{lookup_key}' not found in embeddings registry. "
                f"Check that cfg.env.split ('{self._split}') matches the actual split "
                f"folder name, and that goal_image_path in the episode data is correct."
            )

        self._cached_goal_embedding = self._embeddings_registry[lookup_key]
        return self._cached_goal_embedding

    def _get_cached_goal(self) -> torch.Tensor:
        return self._cached_goal_embedding

    def _build_info(self, success: bool, done: bool) -> Dict[str, Any]:
        return {
            "success": success,
            "done": done,
            "num_steps": self._num_steps,
            "episode_id": self._current_episode.get("id", "?"),
            "scene_id": self._current_episode.get("scene", "?"),
            "collisions": self._episode_collisions,
        }

    # =======================================================================
    # Environment Class Attribute Space Properties
    # =======================================================================
    @property
    def observation_space_shape(self) -> Tuple[int, int, int]:
        return (
            self.cfg.env.image_channels,
            self.cfg.env.image_height,
            self.cfg.env.image_width,
        )

    @property
    def num_actions(self) -> int:
        return self.cfg.env.num_actions