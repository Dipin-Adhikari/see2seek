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

Resilience note (bad-spawn episodes):
    A small fraction of pre-generated RoboTHOR episodes have starting
    poses (position/rotation) that collide with static objects in the
    scene (e.g. a BaseballBat, a wall panel, etc.). AI2-THOR's
    TeleportFull will refuse these placements and return
    lastActionSuccess=False. reset() retries with the next episode (up
    to `_max_reset_retries` times) instead of crashing, and only raises
    once retries are exhausted — which would indicate a systemic
    problem, not a single bad episode.

Resilience note (Unity/controller crash mid-episode — added):
    The AI2-THOR "controller" is really a thin RPC client talking to a
    separate Unity subprocess over a FIFO pipe. If that Unity process
    dies (commonly from GPU/VRAM pressure on smaller cards, but also
    just occasional engine instability over long runs), the *next*
    write to the pipe raises a low-level BrokenPipeError/OSError deep
    inside ai2thor's fifo_server — not a clean AI2-THOR exception.
    Previously this propagated all the way up through step(), through
    the worker process, and killed the whole training run.

    step() now wraps the underlying controller calls and, on detecting
    a dead pipe/controller, tears down and restarts the Controller
    (_restart_controller) and starts a fresh episode via reset(),
    rather than letting the exception kill the worker. The returned
    transition is marked done=True with reward 0.0 and
    info["controller_crashed"]=True so the trainer's rollout/GAE logic
    treats it as a normal episode boundary (bootstrapping is not
    affected by a mid-episode crash-restart).
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

from ai2thor.util.metrics import get_shortest_path_to_point, path_distance


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

# Exceptions that indicate the Unity subprocess / IPC pipe backing the
# AI2-THOR controller has died, rather than a normal in-sim action failure
# (those are reported via event.metadata["lastActionSuccess"], not raised).
_CONTROLLER_DEAD_EXCEPTIONS = (BrokenPipeError, ConnectionError, EOFError, OSError)


class ControllerCrashError(RuntimeError):
    """Raised internally when the AI2-THOR/Unity backend has died and could
    not be recovered within a single episode's controller restart."""


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

    # Max number of consecutive bad episodes we'll skip past in reset()
    # before giving up and raising. A handful of collision-spawn episodes
    # in a row is expected occasionally; hundreds in a row means something
    # else is wrong (bad dataset path, corrupted episode file, etc).
    _max_reset_retries: int = 20

    # Max number of times we'll restart a dead Unity controller within a
    # single step() call before giving up and raising ControllerCrashError.
    # Restarting Unity is expensive (~seconds), so this is intentionally
    # small — repeated failures back-to-back mean something systemic
    # (e.g. GPU OOM that a fresh process will hit again immediately).
    _max_controller_restarts: int = 3

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
        self._episode_collisions: int = 0

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

    def _restart_controller(self) -> None:
        """
        Tear down a dead/unresponsive AI2-THOR controller and start a fresh
        one on the same port.

        Called when a controller RPC raises one of
        `_CONTROLLER_DEAD_EXCEPTIONS`, indicating the backing Unity
        subprocess has crashed and the FIFO pipe to it is broken. Any
        exception from `.stop()` on the old (already-dead) controller is
        swallowed — there is nothing meaningful left to clean up on that
        side, and we don't want a failed shutdown to mask the restart.
        """
        logger.warning(
            f"RoboTHOREnv[{self._worker_id}] AI2-THOR controller appears to have "
            f"crashed — restarting Unity backend on port {8200 + self._worker_id} ..."
        )
        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception:
                pass
            self._controller = None

        self._init_controller()
        logger.info(f"RoboTHOREnv[{self._worker_id}] controller restarted successfully.")

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

        Resilient to "bad" episodes whose starting pose collides with a
        static object in the scene (TeleportFull -> lastActionSuccess=False)
        or whose renderer fails to produce a frame. Such episodes are
        skipped in favor of the next one in the shuffled queue, up to
        `_max_reset_retries` consecutive attempts, instead of raising and
        killing the worker subprocess.

        Also resilient to the controller itself being dead (e.g. called
        right after a mid-episode Unity crash during step()): a
        BrokenPipeError/OSError here triggers a controller restart and a
        retry of the same reset attempt, without consuming one of the
        "bad episode" retries.

        Returns:
            obs: dict tracking keys:
                "rgb":  (3, H, W) live camera float view tensor
                "goal": (512,) precalculated float embedding target vector
        """
        last_error: Optional[str] = None
        controller_restarts = 0

        attempt = 0
        while attempt < self._max_reset_retries:
            if self._episode_index >= len(self._episodes):
                random.shuffle(self._episodes)
                self._episode_index = 0

            self._current_episode = self._episodes[self._episode_index]
            self._episode_index += 1
            self._num_steps = 0
            self._episode_collisions = 0

            ep = self._current_episode
            scene = ep["scene"]

            try:
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
                    last_error = f"Renderer failed to produce a frame after {retries} retries (scene={scene})"
                    logger.warning(f"{last_error} — skipping to next episode")
                    attempt += 1
                    continue

                # 2. Build rotation payload matching AI2-THOR API parameters (Yaw mapping)
                start_rotation = {"x": 0, "y": ep.get("initial_orientation", 0.0), "z": 0}

                event = self._controller.step(
                    action="TeleportFull",
                    position=ep["initial_position"],
                    rotation=start_rotation,
                    horizon=ep.get("initial_horizon", 0),
                )
            except _CONTROLLER_DEAD_EXCEPTIONS as e:
                # Unity died during reset itself. Restart the controller and
                # retry this same episode attempt (don't burn a "bad episode"
                # retry on what is really a backend crash).
                controller_restarts += 1
                if controller_restarts > self._max_controller_restarts:
                    raise ControllerCrashError(
                        f"RoboTHOREnv[{self._worker_id}] controller crashed "
                        f"{controller_restarts} times during reset(); giving up. "
                        f"Last error: {type(e).__name__}: {e}"
                    ) from e
                self._restart_controller()
                self._episode_index -= 1  # re-try the same episode, don't skip it
                continue

            if not event.metadata.get("lastActionSuccess", True):
                last_error = (
                    f"TeleportFull failed for episode={ep.get('id', '?')} scene={scene}, "
                    f"position={ep['initial_position']}, rotation={start_rotation}: "
                    f"{event.metadata.get('errorMessage')}"
                )
                logger.warning(f"{last_error} — skipping to next episode")
                attempt += 1
                continue

            if event.frame is None:
                last_error = f"TeleportFull succeeded but returned no frame (scene={scene})"
                logger.warning(f"{last_error} — skipping to next episode")
                attempt += 1
                continue

            # 3. Calculate distance tracking metrics relative to final waypoint destination coordinate proxy

            goal_pos = ep["shortest_path"][-1]
            self._prev_geodesic_dist = self._get_geodesic_distance(goal_pos)

            # 3b. SPL bookkeeping: L = shortest-path length (sum of segment
            # lengths along the precomputed shortest_path waypoints), and
            # P = distance actually traveled this episode (accumulated in
            # step(), reset here to 0).
            self._shortest_path_length = self._compute_path_length(ep["shortest_path"])
            self._path_length = 0.0
            self._last_agent_pos = dict(ep["initial_position"])

            # 4. Fetch the environment state observations
            rgb = self._get_rgb_tensor(event.frame)
            goal_embedding = self._load_goal_embedding(ep)

            return {"rgb": rgb, "goal": goal_embedding}

        # Exhausted all retries — this indicates a systemic problem
        # (e.g. bad dataset path, corrupted episode file) rather than a
        # single unlucky spawn, so we raise here.
        raise RuntimeError(
            f"RoboTHOREnv[{self._worker_id}] failed to reset after "
            f"{self._max_reset_retries} consecutive attempts. "
            f"Last error: {last_error}"
        )

    def step(
        self, action: int
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Executes a control navigation step command.

        If the AI2-THOR/Unity backend has crashed (pipe broken), the
        controller is restarted and a fresh episode is started via
        reset(). The transition returned in that case is a synthetic
        episode boundary: reward=0.0, done=True,
        info["controller_crashed"]=True — the trainer's rollout buffer
        should treat it exactly like any other episode-terminal step
        (value bootstrapping for a done=True step is already a no-op in
        standard PPO/GAE implementations).
        """
        assert 0 <= action < len(ACTIONS), f"Action index bounds error: {action}"

        self._num_steps += 1
        action_name = ACTIONS[action]

        try:
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
        except _CONTROLLER_DEAD_EXCEPTIONS as e:
            return self._recover_from_controller_crash(action_name, e)

        # Accumulate traveled path length (only counts actual displacement,
        # so failed/blocked moves that don't change position contribute ~0).
        agent_pos = event.metadata["agent"]["position"]
        self._path_length += float(np.linalg.norm([
            agent_pos["x"] - self._last_agent_pos["x"],
            agent_pos["z"] - self._last_agent_pos["z"],
        ]))
        self._last_agent_pos = dict(agent_pos)

        if not event.metadata.get("lastActionSuccess", True):
            logger.warning(
                f"Action '{action_name}' failed at step {self._num_steps} "
                f"(episode={self._current_episode.get('id', '?')}): "
                f"{event.metadata.get('errorMessage')}"
            )

        goal_pos = self._current_episode["shortest_path"][-1]
        curr_dist = self._get_geodesic_distance(goal_pos)

        raw_shaping = (self._prev_geodesic_dist - curr_dist) * self.cfg.env.geodesic_reward_scale
        # Clip to a sane per-step range — one MoveAhead step is ~0.25m, so a
        # single step's shaping reward should never plausibly need to exceed
        # roughly that scale. Guards against any residual distance-metric
        # noise (e.g. waypoint-index jumps) from dominating the reward.
        raw_shaping = float(np.clip(raw_shaping, -0.5, 0.5))

        reward = raw_shaping
        reward += self.cfg.env.slack_reward

        if action_name in {"MoveAhead"} and event.metadata.get("collided", False):
            reward += self.cfg.env.collision_penalty
            self._episode_collisions += 1
        elif action_name in {"RotateLeft", "RotateRight"}:
            reward += self.cfg.env.rotation_penalty

        self._prev_geodesic_dist = curr_dist

        done = self._num_steps >= self.cfg.env.max_steps
        info = self._build_info(success=False, done=done)

        obs = {"rgb": self._get_rgb_tensor(event.frame), "goal": self._get_cached_goal()}
        return obs, reward, done, info

    def _recover_from_controller_crash(
        self, action_name: str, exc: Exception
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Handle a dead-controller exception raised mid-step: restart Unity,
        start a fresh episode, and return a synthetic terminal transition
        so the caller (VecEnv worker / trainer) sees a clean episode
        boundary rather than a propagating exception.

        Raises ControllerCrashError if the controller cannot be brought
        back up within `_max_controller_restarts` attempts — at that
        point something systemic (e.g. persistent GPU OOM) is going on
        and it's better to fail loudly than restart-loop forever.
        """
        episode_id = self._current_episode.get("id", "?") if self._current_episode else "?"
        logger.error(
            f"RoboTHOREnv[{self._worker_id}] controller/backend crashed while executing "
            f"'{action_name}' at step {self._num_steps} (episode={episode_id}): "
            f"{type(exc).__name__}: {exc}"
        )

        restarts = 0
        while restarts < self._max_controller_restarts:
            restarts += 1
            try:

                self._restart_controller()
                obs = self.reset()
                info = {
                    "success": False,
                    "done": True,
                    "num_steps": self._num_steps,
                    "episode_id": episode_id,
                    "scene_id": self._current_episode.get("scene", "?") if self._current_episode else "?",
                    "collisions": self._episode_collisions,
                    "controller_crashed": True,
                }
                return obs, 0.0, True, info
            except _CONTROLLER_DEAD_EXCEPTIONS as e:
                logger.warning(
                    f"RoboTHOREnv[{self._worker_id}] controller restart attempt "
                    f"{restarts}/{self._max_controller_restarts} failed: {type(e).__name__}: {e}"
                )
                exc = e

        raise ControllerCrashError(
            f"RoboTHOREnv[{self._worker_id}] could not recover controller after "
            f"{self._max_controller_restarts} restart attempts. Last error: "
            f"{type(exc).__name__}: {exc}. This usually means something systemic "
            f"(e.g. persistent GPU OOM) rather than a one-off crash — check "
            f"`nvidia-smi` / `dmesg` for OOM-killer activity."
        ) from exc

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
        """
        True geodesic distance via AI2-THOR's native pathfinding engine
        (GetShortestPathToPoint), rather than an approximation from static
        precomputed waypoints. This queries the actual navmesh, so it
        correctly accounts for walls/obstacles at the agent's CURRENT
        position — not just along the dataset's precomputed shortest_path.

        Falls back to Euclidean distance if pathfinding fails (e.g. agent is
        in an unreachable pose, or a transient engine hiccup) — this should
        be rare, but must not crash a training rollout.
        """
        agent_pos = self._controller.last_event.metadata["agent"]["position"]

        try:
            corners = get_shortest_path_to_point(
                self._controller,
                initial_position=agent_pos,
                target_position=goal_pos,
            )
            return float(path_distance(corners))
        except ValueError as e:
            # Pathfinding failed (e.g. no valid path found from current pose).
            # Fall back to Euclidean rather than crashing the episode/worker.
            logger.warning(
                f"GetShortestPathToPoint failed at step {self._num_steps} "
                f"(episode={self._current_episode.get('id', '?')}): {e} "
                f"— falling back to Euclidean distance for this step."
            )
            return float(np.linalg.norm([
                agent_pos["x"] - goal_pos["x"], agent_pos["z"] - goal_pos["z"],
            ]))

    def _compute_path_length(self, waypoints: List[Dict]) -> float:
        """Sum consecutive XZ-plane distances along the shortest_path waypoint list."""
        if len(waypoints) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            total += float(np.linalg.norm([a["x"] - b["x"], a["z"] - b["z"]]))
        return total

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
        spl = 0.0
        if success and self._shortest_path_length > 0:
            spl = self._shortest_path_length / max(self._path_length, self._shortest_path_length)

        return {
            "success": success,
            "done": done,
            "spl": spl,
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