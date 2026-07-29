"""
Runs each environment in a separate subprocess to bypass Python's GIL
and achieve true parallelism. Communication uses multiprocessing Pipes.

This approach mirrors habitat-baselines' VectorEnv but is simplified for
our RoboTHOR setup.

Key design choices:
    - Each worker process owns one RoboTHOREnv instance (and one AI2-THOR
      Unity subprocess via the env). Spawning all 16 is slow on first run
      (~60s) but subsequent resets are fast.
    - reset_at(i) allows resetting a single env without touching others;
      used by the trainer to seamlessly continue rollouts across episodes.
    - Observations are stacked into batched tensors for GPU efficiency.

Resilience note (added):
    Previously, any uncaught exception inside a worker process (e.g. a
    RoboTHOREnv.reset()/step() failure) would kill that subprocess
    silently. The parent process's blocking conn.recv() would then raise
    a bare EOFError with no indication of what actually went wrong. The
    worker loop now wraps each command in a try/except: on failure it
    logs the exception locally *and* sends a tagged ("__error__", message)
    result back through the pipe before exiting, so the parent can raise
    a clear, actionable error instead of an opaque EOFError.

Resilience note (fd leak fix — added):
    conn.send() on a dict containing raw CPU torch.Tensors does NOT pickle
    them by value. torch installs a custom multiprocessing reducer that
    shares CPU tensors via shared memory, which (under Linux's default
    "file_descriptor" sharing strategy) opens a new file descriptor for
    every tensor sent through the pipe. Sending an "rgb" + "goal" tensor
    every single step, from every worker, steadily accumulates fds that
    are not reclaimed as fast as they're opened — long runs eventually
    exceed the process ulimit (`OSError: [Errno 24] Too many open files`),
    which previously killed a worker mid-training (see step ~23k crash).

    Fix: obs dicts are converted to plain numpy arrays (`.numpy()`) right
    before every conn.send() in the worker, and converted back to a single
    batched torch tensor per key in `_stack_obs` on the parent side.
    Numpy arrays are pickled by value over the pipe — no shared memory,
    no fd involved. This trades a small per-step serialization copy for
    eliminating the leak entirely; it does not require raising ulimit -n.

Usage:
    vec_env = make_vec_envs(cfg)
    obs = vec_env.reset_all()          # list of obs dicts → batched tensors
    obs, rewards, dones, infos = vec_env.step(actions_tensor)
    vec_env.close()
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .robothor_env import RoboTHOREnv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for tensor <-> numpy conversion across the pipe boundary
# ---------------------------------------------------------------------------

def _obs_to_numpy(obs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """
    Convert an obs dict's torch.Tensor values to numpy arrays before sending
    through a multiprocessing pipe.

    This is the core of the fd-leak fix: numpy arrays are pickled by value,
    while raw CPU torch.Tensors are shared via shared memory (one fd per
    tensor, per send) by torch's custom multiprocessing reducer.

    Detaches + moves to CPU first in case a tensor ever ends up on GPU or
    requires grad (defensive — obs tensors here are always CPU/no-grad in
    practice, but this keeps the conversion safe regardless).
    """
    return {k: v.detach().cpu().numpy() for k, v in obs.items()}


def _obs_to_tensor(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Inverse of _obs_to_numpy — used on the parent side after recv."""
    return {k: torch.from_numpy(v) for k, v in obs.items()}


# ---------------------------------------------------------------------------
# Worker process function
# ---------------------------------------------------------------------------

def _worker(
    worker_id: int,
    cfg,
    conn: mp.connection.Connection,
    parent_conn: mp.connection.Connection,
) -> None:
    """
    Worker process entry point.

    Runs an infinite loop, receiving commands from the parent process
    and sending back results through the pipe.

    Commands (sent as tuples):
        ("reset",)           → obs dict (numpy arrays)
        ("step", action)     → (obs dict (numpy arrays), reward, done, info)
        ("close",)           → exits loop

    On any unexpected exception while handling a command, the worker
    logs the full traceback locally, sends back a tagged
    ("__error__", worker_id, message) tuple so the parent process can
    surface a meaningful error, and then exits the loop (the subprocess
    terminates after this function returns).
    """
    parent_conn.close()   # child does not use parent side of pipe

    # worker_id sets the unique port offset (8200 + worker_id) to avoid connection collisions
    env = RoboTHOREnv(cfg, worker_id=worker_id)

    while True:
        try:
            cmd, *args = conn.recv()
        except EOFError:
            break

        try:
            if cmd == "reset":
                obs = env.reset()
                conn.send(_obs_to_numpy(obs))

            elif cmd == "step":
                action = args[0]
                obs, reward, done, info = env.step(action)
                if done:
                    # Seamless auto-reset: returned 'obs' is the initial frame of the next
                    # episode, while reward/done/info correspond to the terminal transition.
                    obs = env.reset()
                conn.send((_obs_to_numpy(obs), reward, done, info))

            elif cmd == "close":
                env.close()
                break

            else:
                raise ValueError(f"Unknown command: {cmd}")

        except Exception as e:
            # Log full traceback in this worker's own logs for debugging,
            # then report a tagged error back through the pipe instead of
            # letting the subprocess die silently (which would otherwise
            # surface as a bare EOFError in the parent process).
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


# ---------------------------------------------------------------------------
# Vectorised environment
# ---------------------------------------------------------------------------

class VecEnv:
    """
    Manages N RoboTHOR environments running in parallel subprocesses.

    Args:
        cfg:        Config object.
        num_envs:   Number of parallel environments (default: cfg.env.num_envs).
    """

    def __init__(self, cfg, num_envs: Optional[int] = None) -> None:
        self.cfg = cfg
        self.num_envs = num_envs or cfg.env.num_envs

        self._parent_conns: List[mp.connection.Connection] = []
        self._processes: List[mp.Process] = []

        ctx = mp.get_context("spawn")   # "spawn" required for CUDA in subprocesses

        for i in range(self.num_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(i, cfg, child_conn, parent_conn),
                daemon=True,
            )
            p.start()
            child_conn.close()   # parent does not use child side
            self._parent_conns.append(parent_conn)
            self._processes.append(p)

        logger.info(f"VecEnv: {self.num_envs} worker processes started")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset_all(self) -> Dict[str, torch.Tensor]:
        """
        Reset all environments.

        Returns:
            Batched obs dict:
                "rgb":  (N, 3, H, W)   - Batched live agent frames
                "goal": (N, 512)       - Batched pre-cached target CLIP embeddings
        """
        for conn in self._parent_conns:
            conn.send(("reset",))

        obs_list = self._recv_all()
        return self._stack_obs(obs_list)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Step all environments in parallel.

        Args:
            actions: (N,) long tensor of action indices.

        Returns:
            obs:     Batched obs dict {"rgb": (N, 3, H, W), "goal": (N, 512)}
            rewards: (N,) float tensor
            dones:   (N,) bool tensor
            infos:   List of N info dicts
        """
        # Send actions asynchronously
        for conn, action in zip(self._parent_conns, actions.tolist()):
            conn.send(("step", action))

        # Collect results
        results = self._recv_all()
        obs_list   = [r[0] for r in results]
        rewards    = torch.tensor([r[1] for r in results], dtype=torch.float32)
        dones      = torch.tensor([r[2] for r in results], dtype=torch.bool)
        infos      = [r[3] for r in results]

        return self._stack_obs(obs_list), rewards, dones, infos

    def close(self) -> None:
        """Terminate all worker processes."""
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recv_all(self) -> List[Any]:
        """
        Receive one result from every parent connection, raising a clear
        RuntimeError if any worker reported an error or died unexpectedly
        (pipe closed -> EOFError) instead of letting a bare EOFError
        propagate up from multiprocessing internals.
        """
        results = []
        for i, conn in enumerate(self._parent_conns):
            try:
                result = conn.recv()
            except EOFError as e:
                raise RuntimeError(
                    f"VecEnv: worker {i} pipe closed unexpectedly (process likely "
                    f"crashed). Check the worker's own stderr/log output above for "
                    f"the original traceback."
                ) from e

            if isinstance(result, tuple) and len(result) == 3 and result[0] == "__error__":
                _, worker_id, message = result
                raise RuntimeError(
                    f"VecEnv: worker {worker_id} reported an error: {message}"
                )

            results.append(result)
        return results

    @staticmethod
    def _stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Stack a list of per-env obs dicts (numpy arrays, as received over
        the pipe — see the fd-leak fix note at the top of this file) into
        batched torch tensors.

        Handles non-uniform shapes per key gracefully:
            - "rgb" items of shape (3, H, W) -> stacked to (N, 3, H, W)
            - "goal" items of shape (512,)   -> stacked to (N, 512)

        Uses np.stack + a single torch.from_numpy per key (one conversion
        for the whole batch) rather than converting each env's arrays to a
        tensor individually — cheaper and keeps the "convert once, at the
        batch boundary" invariant that avoids the original fd leak.
        """
        keys = obs_list[0].keys()
        return {
            k: torch.from_numpy(np.stack([obs[k] for obs in obs_list], axis=0))
            for k in keys
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_vec_envs(cfg, num_envs: Optional[int] = None) -> VecEnv:
    """
    Construct a VecEnv from a Config.

    Args:
        cfg:      Config object.
        num_envs: Override the number of parallel envs (default: cfg.env.num_envs).

    Returns:
        Initialised VecEnv.
    """
    return VecEnv(cfg, num_envs=num_envs)