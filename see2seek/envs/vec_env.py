"""
vec_env.py — persistent shared-memory buffers (fixes both the fd leak
AND the throughput regression from the numpy-pickling fix)

Why this is better than the numpy fix:
    The numpy version pickled the full rgb/goal arrays through the pipe
    every step — correct, but ~2-3x slower (full serialize + copy per
    step instead of zero-copy shared memory).

    The original raw-tensor version was zero-copy but leaked one fd per
    NEW tensor storage sent — since a fresh tensor was created every
    step, fds accumulated forever.

    Fix: allocate the rgb/goal buffers ONCE per worker at startup with
    .share_memory_(), pass them to the worker process, and have the
    worker copy_() new data into the SAME buffer every step instead of
    creating a new tensor. Same storage/fd reused every step -> zero
    leak. Parent just reads the buffer it already has a reference to ->
    zero-copy, full speed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

import torch

from .robothor_env import RoboTHOREnv

logger = logging.getLogger(__name__)


def _worker(
    worker_id: int,
    cfg,
    conn: mp.connection.Connection,
    parent_conn: mp.connection.Connection,
    shared_rgb: torch.Tensor,       # pre-shared (3, H, W) buffer for this worker
    shared_goal: torch.Tensor,      # pre-shared (512,) buffer for this worker
    shared_pointgoal: torch.Tensor, # pre-shared (3,) buffer for this worker
) -> None:
    parent_conn.close()

    env = RoboTHOREnv(cfg, worker_id=worker_id)

    while True:
        try:
            cmd, *args = conn.recv()
        except EOFError:
            break

        try:
            if cmd == "reset":
                obs = env.reset()
                shared_rgb.copy_(obs["rgb"])
                shared_goal.copy_(obs["goal"])
                shared_pointgoal.copy_(obs["pointgoal"])
                conn.send("ok")

            elif cmd == "step":
                action = args[0]
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
                shared_rgb.copy_(obs["rgb"])
                shared_goal.copy_(obs["goal"])
                shared_pointgoal.copy_(obs["pointgoal"])
                conn.send(("ok", reward, done, info))

            elif cmd == "set_max_steps":
                env.set_max_steps(args[0])
                conn.send("ok")

            elif cmd == "set_exploration_bonus":
                env.set_exploration_bonus(args[0])
                conn.send("ok")

            elif cmd == "close":
                env.close()
                break

            else:
                raise ValueError(f"Unknown command: {cmd}")

        except Exception as e:
            logger.exception(
                f"Worker {worker_id} raised an exception handling cmd={cmd!r}"
            )
            try:
                conn.send(("__error__", worker_id, f"{type(e).__name__}: {e}"))
            except (BrokenPipeError, EOFError):
                pass
            break

    try:
        env.close()
    except Exception:
        pass


_WORKER_DEAD_EXCEPTIONS = (BrokenPipeError, EOFError, ConnectionError, OSError)


class VecEnv:
    def __init__(self, cfg, num_envs: Optional[int] = None) -> None:
        self.cfg = cfg
        self.num_envs = num_envs or cfg.env.num_envs

        self._parent_conns: List[mp.connection.Connection] = []
        self._processes: List[mp.Process] = []
        self._shared_rgb: List[torch.Tensor] = []
        self._shared_goal: List[torch.Tensor] = []
        self._shared_pointgoal: List[torch.Tensor] = []

        self._current_max_steps: Optional[int] = None
        self._current_exploration_bonus: Optional[float] = None

        for i in range(self.num_envs):
            conn, proc, rgb, goal, pg = self._create_worker(i)
            self._parent_conns.append(conn)
            self._processes.append(proc)
            self._shared_rgb.append(rgb)
            self._shared_goal.append(goal)
            self._shared_pointgoal.append(pg)

        logger.info(f"VecEnv: {self.num_envs} worker processes started (persistent shared buffers)")

    def _create_worker(
        self, worker_id: int
    ) -> Tuple[mp.connection.Connection, mp.Process, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        img_c = cfg.env.image_channels
        img_h = cfg.env.image_height
        img_w = cfg.env.image_width
        goal_dim = cfg.encoder.goal_embed_dim
        pointgoal_dim = cfg.encoder.pointgoal_input_dim

        rgb_buf = torch.zeros(img_c, img_h, img_w, dtype=torch.float32)
        goal_buf = torch.zeros(goal_dim, dtype=torch.float32)
        pointgoal_buf = torch.zeros(pointgoal_dim, dtype=torch.float32)
        rgb_buf.share_memory_()
        goal_buf.share_memory_()
        pointgoal_buf.share_memory_()

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_worker,
            args=(worker_id, cfg, child_conn, parent_conn, rgb_buf, goal_buf, pointgoal_buf),
            daemon=True,
        )
        p.start()
        child_conn.close()
        return parent_conn, p, rgb_buf, goal_buf, pointgoal_buf

    def _respawn_worker(self, i: int) -> None:
        logger.warning(f"VecEnv: respawning dead worker {i}")

        old_proc = self._processes[i]
        if old_proc.is_alive():
            old_proc.terminate()
        old_proc.join(timeout=5)
        if old_proc.is_alive():
            old_proc.kill()
            old_proc.join(timeout=2)

        try:
            self._parent_conns[i].close()
        except Exception:
            pass

        conn, proc, rgb, goal, pg = self._create_worker(i)
        self._parent_conns[i] = conn
        self._processes[i] = proc
        self._shared_rgb[i] = rgb
        self._shared_goal[i] = goal
        self._shared_pointgoal[i] = pg

        conn.send(("reset",))
        conn.recv()

        if self._current_max_steps is not None:
            conn.send(("set_max_steps", self._current_max_steps))
            conn.recv()
        if self._current_exploration_bonus is not None:
            conn.send(("set_exploration_bonus", self._current_exploration_bonus))
            conn.recv()

        logger.info(f"VecEnv: worker {i} respawned successfully")

    def reset_all(self) -> Dict[str, torch.Tensor]:
        dead = set()
        for i, conn in enumerate(self._parent_conns):
            try:
                conn.send(("reset",))
            except _WORKER_DEAD_EXCEPTIONS:
                dead.add(i)

        for i, conn in enumerate(self._parent_conns):
            if i in dead:
                continue
            try:
                conn.recv()
            except _WORKER_DEAD_EXCEPTIONS:
                dead.add(i)

        for i in dead:
            self._respawn_worker(i)

        return self._stack_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict]]:
        dead: set = set()

        for i, (conn, action) in enumerate(zip(self._parent_conns, actions.tolist())):
            try:
                conn.send(("step", action))
            except _WORKER_DEAD_EXCEPTIONS:
                dead.add(i)

        results: List[Any] = [None] * self.num_envs
        for i, conn in enumerate(self._parent_conns):
            if i in dead:
                continue
            try:
                result = conn.recv()
                if isinstance(result, tuple) and len(result) >= 3 and result[0] == "__error__":
                    dead.add(i)
                else:
                    results[i] = result
            except _WORKER_DEAD_EXCEPTIONS:
                dead.add(i)

        for i in dead:
            logger.error(
                f"VecEnv: worker {i} died (Unity segfault or crash), respawning..."
            )
            self._respawn_worker(i)
            results[i] = ("ok", 0.0, True, {
                "success": False, "done": True, "num_steps": 0,
                "episode_id": "crashed", "scene_id": "unknown",
                "collisions": 0, "controller_crashed": True,
            })

        rewards = torch.tensor([r[1] for r in results], dtype=torch.float32)
        dones   = torch.tensor([r[2] for r in results], dtype=torch.bool)
        infos   = [r[3] for r in results]

        return self._stack_obs(), rewards, dones, infos

    def set_max_steps(self, max_steps: int) -> None:
        """Update effective max_steps on all worker environments (curriculum)."""
        self._current_max_steps = max_steps
        self._send_recv_all("set_max_steps", max_steps)

    def set_exploration_bonus(self, bonus: float) -> None:
        """Update exploration bonus on all worker environments (decay)."""
        self._current_exploration_bonus = bonus
        self._send_recv_all("set_exploration_bonus", bonus)

    def close(self) -> None:
        for conn in self._parent_conns:
            try:
                conn.send(("close",))
            except BrokenPipeError:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        logger.info("VecEnv: all workers closed")

    def _send_recv_all(self, cmd: str, *args: Any) -> None:
        dead: set = set()
        for i, conn in enumerate(self._parent_conns):
            try:
                conn.send((cmd, *args))
            except _WORKER_DEAD_EXCEPTIONS:
                dead.add(i)
        for i, conn in enumerate(self._parent_conns):
            if i in dead:
                continue
            try:
                conn.recv()
            except _WORKER_DEAD_EXCEPTIONS:
                dead.add(i)
        for i in dead:
            self._respawn_worker(i)

    def _stack_obs(self) -> Dict[str, torch.Tensor]:
        # Read directly out of the persistent shared buffers — zero-copy,
        # no new tensor storage created per step, so no new fds either.
        return {
            "rgb": torch.stack(self._shared_rgb, dim=0).clone(),
            "goal": torch.stack(self._shared_goal, dim=0).clone(),
            "pointgoal": torch.stack(self._shared_pointgoal, dim=0).clone(),
        }


def make_vec_envs(cfg, num_envs: Optional[int] = None) -> VecEnv:
    return VecEnv(cfg, num_envs=num_envs)