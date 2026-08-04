"""
evaluator.py — Evaluation loop for ImageNav and ObjectNav (zero-shot).

Runs a trained policy on the validation or test split and computes SR / SPL.

For ObjectNav zero-shot evaluation:
    - Goal is encoded as a CLIP TEXT embedding of the category name
      (e.g. "chair", "television") rather than an image.
    - The policy sees this exactly the same as an ImageNav goal embedding —
      this is the key to zero-shot transfer.


Usage:
    evaluator = Evaluator(cfg, checkpoint_path="data/checkpoints/checkpoint_final.pth")
    results = evaluator.evaluate(split="val", task="imagenav")
    print(results)
    # {"sr": 0.42, "spl": 0.31, "num_episodes": 500}
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, Optional

import torch

from see2seek.utils.config import Config
from see2seek.models.encoders.dino_encoder import DINOv2Encoder
from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder
from see2seek.envs.robothor_env import RoboTHOREnv
from see2seek.agents.gru_policy import GRUActorCritic, build_policy
from see2seek.evaluation.metrics import NavigationMetrics

logger = logging.getLogger(__name__)

TaskType = Literal["imagenav", "objectnav"]


class Evaluator:
    """
    Runs evaluation episodes and collects SR / SPL metrics.

    Args:
        cfg:             Config object.
        checkpoint_path: Path to a .pth checkpoint file.
        device:          Device string.
    """

    def __init__(
        self,
        cfg: Config,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)

        # ---- Load encoders ----
        self.obs_encoder  = DINOv2Encoder(device=str(self.device))
        self.goal_encoder = CLIPGoalEncoder(device=str(self.device))

        # ---- Load policy ----
        self.policy = build_policy(cfg, str(self.device))
        self._load_checkpoint(checkpoint_path)
        self.policy.eval()

        logger.info(f"Evaluator ready — checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        split: str = "val",
        task: TaskType = "imagenav",
        num_episodes: Optional[int] = None,
        objectnav_categories: Optional[List[str]] = None,
        language: str = "en",    # "en" or "ne" (Nepali) for zero-shot text goals
    ) -> Dict[str, float]:
        """
        Run evaluation and return metric dict.

        Args:
            split:                  Dataset split ("val" or "test").
            task:                   "imagenav" (image goals) or "objectnav" (text goals).
            num_episodes:           Max episodes to evaluate. None → full split.
            objectnav_categories:   For ObjectNav: list of category names to evaluate.
                                    None → use default RoboTHOR ObjectNav categories.
            language:               Goal language for ObjectNav text goals.
                                    "en" → English, "ne" → Nepali.

        Returns:
            Dict with "sr", "spl", "num_episodes".
        """
        metrics = NavigationMetrics()

        # Single-process env for evaluation (no parallelism needed)
        env = RoboTHOREnv(self.cfg, worker_id=99)

        # ObjectNav category mapping (English and Nepali)
        category_map = self._get_category_map(language)
        objectnav_cats = objectnav_categories or list(category_map.keys())

        episode_count = 0

        while True:
            if num_episodes is not None and episode_count >= num_episodes:
                break

            # ----- Reset -----
            obs_dict = env.reset()
            episode  = env._current_episode

            # ----- Get goal embedding -----
            if task == "imagenav":
                goal_embed = obs_dict["goal"].unsqueeze(0)
            else:
                # Zero-shot ObjectNav: encode category name as text
                cat_name = episode.get("object_category", objectnav_cats[0])
                text_goal = category_map.get(cat_name, cat_name)
                goal_embed = self.goal_encoder.encode_text(text_goal)  # (1, 512)

            goal_embed = goal_embed.to(self.device)

            # ----- Episode rollout -----
            hidden = self.policy.get_initial_hidden(1, self.device)
            prev_action = torch.tensor(
                [self.cfg.env.num_actions], dtype=torch.long, device=self.device
            )
            masks = torch.ones(1, 1, device=self.device)

            path_length  = 0.0
            prev_pos     = self._get_agent_pos(env)
            success      = False

            for step in range(self.cfg.env.max_steps):
                with torch.no_grad():
                    rgb = obs_dict["rgb"].unsqueeze(0).to(self.device)
                    obs_embed = self.obs_encoder(rgb)              # (1, 512)
                    can_stop = torch.tensor(
                        [step >= self.cfg.env.min_steps_before_stop], device=self.device
                    )
                    dist, _, hidden = self.policy.act(
                        obs_embed, goal_embed, prev_action, hidden, masks, can_stop=can_stop
                    )
                    if episode_count < 5:
                        logger.info(f"  step={step} probs={dist.probs.detach().cpu().numpy()}")
                    action = dist.probs.argmax(dim=-1)             # greedy at eval


                obs_dict, reward, done, info = env.step(action.item())

                # Track path length
                curr_pos    = self._get_agent_pos(env)
                path_length += self._pos_dist(prev_pos, curr_pos)
                prev_pos    = curr_pos

                prev_action = action
                masks = torch.ones(1, 1, device=self.device)   # no reset mid-episode

                if done:
                    success = info.get("success", False)
                    if episode_count < 20:
                        logger.info(f"ep {episode_count}: steps={step+1} success={success} path_length={path_length:.3f}")
                    break

            shortest = episode.get("shortest_path_length", 1.0)
            metrics.update(
                success      = success,
                path_length  = path_length,
                shortest_path= shortest,
            )
            episode_count += 1

            if episode_count % 50 == 0:
                logger.info(
                    f"  [{episode_count} episodes] "
                    f"SR={metrics.sr:.3f}  SPL={metrics.spl:.3f}"
                )

        env.close()
        result = metrics.summary()
        logger.info(f"Evaluation complete: {result}")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        logger.info(f"Loaded checkpoint (step {ckpt.get('total_steps', '?'):,})")

    def _get_agent_pos(self, env: RoboTHOREnv) -> dict:
        return env._controller.last_event.metadata["agent"]["position"]

    @staticmethod
    def _pos_dist(p1: dict, p2: dict) -> float:
        import numpy as np
        return float(np.linalg.norm([p1["x"] - p2["x"], p1["z"] - p2["z"]]))

    @staticmethod
    def _get_category_map(language: str) -> Dict[str, str]:
        """
        Map RoboTHOR ObjectNav category names to text strings for CLIP encoding.

        For Nepali (language="ne"), we use the Nepali name directly.
        CLIP's multilingual encoder handles the rest.
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
            # English — use the category name directly (CLIP knows these)
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
