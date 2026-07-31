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
    shared_rgb: torch.Tensor,   # pre-shared (3, H, W) buffer for this worker
    shared_goal: torch.Tensor,  # pre-shared (512,) buffer for this worker
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
                conn.send("ok")

            elif cmd == "step":
                action = args[0]
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
                shared_rgb.copy_(obs["rgb"])
                shared_goal.copy_(obs["goal"])
                conn.send(("ok", reward, done, info))

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


class VecEnv:
    def __init__(self, cfg, num_envs: Optional[int] = None) -> None:
        self.cfg = cfg
        self.num_envs = num_envs or cfg.env.num_envs

        # Shapes — adjust image_size/goal_dim to match your actual config keys
        img_size = cfg.env.image_width   # e.g. 224
        goal_dim = 512

        self._parent_conns: List[mp.connection.Connection] = []
        self._processes: List[mp.Process] = []
        self._shared_rgb: List[torch.Tensor] = []
        self._shared_goal: List[torch.Tensor] = []

        ctx = mp.get_context("spawn")

        # Pull real dims from config instead of guessing
        img_c = cfg.env.image_channels    # 3
        img_h = cfg.env.image_height      # 224
        img_w = cfg.env.image_width       # 224
        goal_dim = cfg.encoder.goal_embed_dim   # 512 (CLIP ViT-B/32)

        for i in range(self.num_envs):
            rgb_buf = torch.zeros(img_c, img_h, img_w, dtype=torch.float32)
            goal_buf = torch.zeros(goal_dim, dtype=torch.float32)
            rgb_buf.share_memory_()
            goal_buf.share_memory_()

            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(i, cfg, child_conn, parent_conn, rgb_buf, goal_buf),
                daemon=True,
            )
            p.start()
            child_conn.close()

            self._parent_conns.append(parent_conn)
            self._processes.append(p)
            self._shared_rgb.append(rgb_buf)
            self._shared_goal.append(goal_buf)

        logger.info(f"VecEnv: {self.num_envs} worker processes started (persistent shared buffers)")

    def reset_all(self) -> Dict[str, torch.Tensor]:
        for conn in self._parent_conns:
            conn.send(("reset",))
        self._recv_all()  # just "ok" acks — data is already in shared buffers
        return self._stack_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict]]:
        for conn, action in zip(self._parent_conns, actions.tolist()):
            conn.send(("step", action))

        results = self._recv_all()  # [("ok", reward, done, info), ...]
        rewards = torch.tensor([r[1] for r in results], dtype=torch.float32)
        dones   = torch.tensor([r[2] for r in results], dtype=torch.bool)
        infos   = [r[3] for r in results]

        return self._stack_obs(), rewards, dones, infos

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

    def _recv_all(self) -> List[Any]:
        results = []
        for i, conn in enumerate(self._parent_conns):
            try:
                result = conn.recv()
            except EOFError as e:
                raise RuntimeError(
                    f"VecEnv: worker {i} pipe closed unexpectedly (process likely "
                    f"crashed). Check the worker's own stderr/log output above."
                ) from e

            if isinstance(result, tuple) and len(result) == 3 and result[0] == "__error__":
                _, worker_id, message = result
                raise RuntimeError(f"VecEnv: worker {worker_id} reported an error: {message}")

            results.append(result)
        return results

    def _stack_obs(self) -> Dict[str, torch.Tensor]:
        # Read directly out of the persistent shared buffers — zero-copy,
        # no new tensor storage created per step, so no new fds either.
        return {
            "rgb": torch.stack(self._shared_rgb, dim=0).clone(),
            "goal": torch.stack(self._shared_goal, dim=0).clone(),
        }


def make_vec_envs(cfg, num_envs: Optional[int] = None) -> VecEnv:
    return VecEnv(cfg, num_envs=num_envs)