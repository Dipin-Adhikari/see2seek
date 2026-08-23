"""
evaluator.py — Parallel evaluation loop for ImageNav and ObjectNav (zero-shot).

Runs a trained policy on the validation or test split using VecEnv (parallel
environments, same as training) and computes SR / SPL. Logs each episode
result to a file.

Usage:
    evaluator = Evaluator(cfg, checkpoint_path="data/checkpoints/checkpoint_final.pth")
    results = evaluator.evaluate(split="val", task="imagenav")
    print(results)
    # {"sr": 0.42, "spl": 0.31, "num_episodes": 500}
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Literal, Optional

import torch

from see2seek.utils.config import Config
from see2seek.models.encoders.dino_encoder import DINOv2Encoder
from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder
from see2seek.envs.vec_env import make_vec_envs
from see2seek.agents.gru_policy import build_policy

logger = logging.getLogger(__name__)

TaskType = Literal["imagenav", "objectnav"]


class Evaluator:
    """
    Runs parallel evaluation episodes and collects SR / SPL metrics.

    Args:
        cfg:             Config object.
        checkpoint_path: Path to a .pth checkpoint file.
        device:          Device string.
        num_envs:        Number of parallel eval environments (default: from config).
    """

    def __init__(
        self,
        cfg: Config,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_envs: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
        self.num_envs = num_envs or cfg.env.num_envs

        # Auto-detect obs_encoder_type from checkpoint if not explicitly set
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_cfg = ckpt.get("cfg", None)
        if ckpt_cfg is not None:
            ckpt_enc_type = getattr(getattr(ckpt_cfg, "encoder", None), "obs_encoder_type", None)
            if ckpt_enc_type and cfg.encoder.obs_encoder_type == "dino":
                # Override with checkpoint's encoder type (unless user explicitly set it)
                cfg.encoder.obs_encoder_type = ckpt_enc_type
                if ckpt_enc_type != "dino":
                    logger.info(f"Auto-detected obs_encoder_type='{ckpt_enc_type}' from checkpoint")

        self.obs_encoder_type = cfg.encoder.obs_encoder_type

        # ---- Load encoders ----
        if self.obs_encoder_type == "dino":
            self.obs_encoder = DINOv2Encoder(device=str(self.device))
        else:
            self.obs_encoder = None
        self.goal_encoder = CLIPGoalEncoder(device=str(self.device))

        # ---- Load policy ----
        self.policy = build_policy(cfg, str(self.device))
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()
        logger.info(f"Loaded checkpoint (step {ckpt.get('total_steps', '?'):,})")

        logger.info(f"Evaluator ready — obs_encoder={self.obs_encoder_type}, checkpoint: {checkpoint_path}, num_envs: {self.num_envs}")

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        split: str = "val",
        task: TaskType = "imagenav",
        num_episodes: Optional[int] = None,
        log_file: Optional[str] = None,
        zero_pointgoal: bool = False,
    ) -> Dict[str, float]:
        """
        Run parallel evaluation and return metric dict.

        Args:
            split:          Dataset split ("val" or "test").
            task:           "imagenav" or "objectnav".
            num_episodes:   Max episodes to evaluate. None → full split.
            log_file:       Path to write per-episode logs. None → auto-generated.

        Returns:
            Dict with "sr", "spl", "num_episodes", "mean_steps", "mean_collisions".
        """
        # Setup log file
        if log_file is None:
            os.makedirs(self.cfg.logging.log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(
                self.cfg.logging.log_dir, f"eval_{task}_{split}_{timestamp}.log"
            )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        eval_logger = logging.getLogger("see2seek.eval")
        eval_logger.setLevel(logging.INFO)
        eval_logger.addHandler(file_handler)
        eval_logger.addHandler(logging.StreamHandler())

        eval_logger.info(f"=== Evaluation Start ===")
        eval_logger.info(f"  Task: {task} | Split: {split} | Num envs: {self.num_envs}")
        eval_logger.info(f"  Target episodes: {num_episodes or 'all'}")
        eval_logger.info(f"  Log file: {log_file}")

        successes = []
        spls = []
        all_steps = []
        all_collisions = []

        # Launch parallel environments
        vec_env = make_vec_envs(self.cfg, num_envs=self.num_envs)

        # Initial reset
        obs_dict = vec_env.reset_all()

        # Policy state
        hidden = self.policy.get_initial_hidden(self.num_envs, self.device)
        prev_actions = torch.full(
            (self.num_envs,), self.cfg.env.num_actions, dtype=torch.long, device=self.device
        )
        masks = torch.ones(self.num_envs, 1, device=self.device)
        steps_since_reset = torch.zeros(self.num_envs, device=self.device)

        # Episodic memory buffers
        memory_size = getattr(self.cfg.encoder, "memory_size", 64)
        cls_dim_for_mem = (
            self.cfg.encoder.dino_cls_dim if self.obs_encoder_type == "dino"
            else self.cfg.encoder.goal_embed_dim
        )
        memory_buffer = torch.zeros(
            self.num_envs, memory_size, cls_dim_for_mem, device=self.device
        )
        memory_mask = torch.zeros(
            self.num_envs, memory_size, device=self.device, dtype=torch.bool
        )

        episode_count = 0
        total_steps = 0
        start_time = time.time()

        while True:
            if num_episodes is not None and episode_count >= num_episodes:
                break

            with torch.no_grad():
                rgb = obs_dict["rgb"].to(self.device)
                if self.obs_encoder_type == "dino":
                    cls_embed, patch_embed = self.obs_encoder.get_all_embeddings(rgb)
                else:
                    cls_embed = self.goal_encoder.get_obs_embedding(rgb)
                    patch_embed = None

                goal_embed = obs_dict["goal"].to(self.device)
                if task == "objectnav" or zero_pointgoal:
                    pointgoal = torch.zeros(self.num_envs, 3, device=self.device)
                else:
                    pointgoal = obs_dict["pointgoal"].to(self.device)

                can_stop = steps_since_reset >= self.cfg.env.min_steps_before_stop

                dist, _, hidden_next, memory_buffer, memory_mask = self.policy.act(
                    patch_embed, cls_embed, goal_embed, prev_actions,
                    hidden, masks, pointgoal=pointgoal, can_stop=can_stop,
                    memory_buffer=memory_buffer, memory_mask=memory_mask,
                )
                actions = dist.sample()

            obs_dict, rewards, dones, infos = vec_env.step(actions)

            steps_since_reset += 1
            total_steps += self.num_envs

            # Process completed episodes
            for env_idx in range(self.num_envs):
                if dones[env_idx]:
                    info = infos[env_idx]
                    success = info.get("success", False)
                    spl = info.get("spl", 0.0)
                    ep_steps = info.get("num_steps", 0)
                    ep_id = info.get("episode_id", "?")
                    scene_id = info.get("scene_id", "?")
                    collisions = info.get("collisions", 0)

                    successes.append(int(success))
                    spls.append(spl)
                    all_steps.append(ep_steps)
                    all_collisions.append(collisions)
                    episode_count += 1

                    eval_logger.info(
                        f"[ep {episode_count:4d}] scene={scene_id} id={ep_id} "
                        f"success={success} steps={ep_steps} collisions={collisions} "
                        f"spl={spl:.3f}"
                    )

                    if episode_count % 50 == 0:
                        elapsed = time.time() - start_time
                        sr_so_far = sum(successes) / len(successes)
                        spl_so_far = sum(spls) / len(spls)
                        eval_logger.info(
                            f"  --- Progress: {episode_count} episodes | "
                            f"SR={sr_so_far:.3f} SPL={spl_so_far:.3f} | "
                            f"time={elapsed:.0f}s ---"
                        )

                    if num_episodes is not None and episode_count >= num_episodes:
                        break

            # Update recurrent state
            new_masks = (~dones).float().unsqueeze(1).to(self.device)
            hidden = hidden_next * new_masks.unsqueeze(0)
            # Reset steps counter for done envs
            steps_since_reset = torch.where(
                dones.to(self.device), torch.zeros_like(steps_since_reset), steps_since_reset
            )
            prev_actions = actions
            masks = new_masks

        vec_env.close()

        # Final results
        elapsed = time.time() - start_time
        sr = sum(successes) / max(len(successes), 1)
        spl_mean = sum(spls) / max(len(spls), 1)
        mean_steps = sum(all_steps) / max(len(all_steps), 1)
        mean_collisions = sum(all_collisions) / max(len(all_collisions), 1)

        result = {
            "sr": round(sr, 4),
            "spl": round(spl_mean, 4),
            "num_episodes": episode_count,
            "mean_steps": round(mean_steps, 1),
            "mean_collisions": round(mean_collisions, 1),
            "total_time_s": round(elapsed, 1),
            "eps_per_sec": round(episode_count / max(elapsed, 1), 2),
        }

        eval_logger.info(f"\n=== Evaluation Complete ===")
        eval_logger.info(f"  Episodes:       {episode_count}")
        eval_logger.info(f"  SR:             {sr:.4f}")
        eval_logger.info(f"  SPL:            {spl_mean:.4f}")
        eval_logger.info(f"  Mean steps:     {mean_steps:.1f}")
        eval_logger.info(f"  Mean collisions:{mean_collisions:.1f}")
        eval_logger.info(f"  Time:           {elapsed:.1f}s ({result['eps_per_sec']:.2f} eps/s)")
        eval_logger.info(f"  Log saved:      {log_file}")

        # Cleanup logger
        eval_logger.removeHandler(file_handler)
        file_handler.close()

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        """Legacy method — checkpoint loading is now done in __init__."""
        pass

    @staticmethod
    def _get_category_map(language: str) -> Dict[str, str]:
        """
        Map RoboTHOR ObjectNav category names to text strings for CLIP encoding.
        """
        if language == "ne":
            return {
                "AlarmClock":   "अलार्म घडी",
                "Apple":        "स्याउ",
                "BaseballBat":  "बेसबल ब्याट",
                "BasketBall":   "बास्केटबल",
                "Bowl":         "कचौरा",
                "GarbageCan":   "फोहोर डब्बा",
                "HousePlant":   "घरको बिरुवा",
                "Laptop":       "ल्यापटप",
                "Mug":          "मग",
                "SprayBottle":  "स्प्रे बोतल",
                "Television":   "टेलिभिजन",
                "Vase":         "फूलदानी",
            }
        else:
            return {
                "AlarmClock":   "alarm clock",
                "Apple":        "apple",
                "BaseballBat":  "baseball bat",
                "BasketBall":   "basketball",
                "Bowl":         "bowl",
                "GarbageCan":   "garbage can",
                "HousePlant":   "house plant",
                "Laptop":       "laptop",
                "Mug":          "mug",
                "SprayBottle":  "spray bottle",
                "Television":   "television",
                "Vase":         "vase",
            }
