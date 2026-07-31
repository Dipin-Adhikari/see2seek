"""
train.py — Training entry point for See to Seek.

Usage:
    # Train with default config:
    python train.py

    # Override with YAML:
    python train.py --config configs/train_robothor.yaml

    # Resume from checkpoint:
    python train.py --config configs/train_robothor.yaml --resume data/checkpoints/checkpoint_000500000.pth

    # Quick smoke test (2 updates):
    python train.py --debug
"""

import argparse
import logging
import os
from datetime import datetime
import random

import numpy as np
import torch

from see2seek.trainers.ppo_trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="See to Seek — PPO Training")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--debug",  action="store_true", help="Short smoke test (2 updates)")
    p.add_argument("--seed",   type=int, default=None, help="Override random seed")
    p.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    return p.parse_args()


def setup_logging(debug: bool = False, log_dir: str = "data/logs") -> None:
    level = logging.DEBUG if debug else logging.INFO
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logging.getLogger(__name__).info(f"Logging to {log_file}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility vs. speed tradeoff — deterministic=True is slower
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # ---- Load config ----
    if args.config is not None:
        from see2seek.utils.config import load_config
        cfg = load_config(args.config)
        logger.info(f"Config loaded from: {args.config}")
    else:
        from see2seek.utils.config import Config
        cfg = Config()
        logger.info("Using default config")

    # ---- Apply CLI overrides ----
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device

    if args.debug:
        # Minimal run to check the pipeline end-to-end
        cfg.ppo.total_num_steps = cfg.ppo.num_steps * cfg.env.num_envs * 2
        cfg.logging.use_wandb   = False
        cfg.env.num_envs        = 2
        logger.info("DEBUG MODE: 2 updates, 2 envs, W&B disabled")

    # ---- Reproducibility ----
    set_seed(cfg.seed)
    logger.info(f"Seed: {cfg.seed}")

    # ---- Create required directories ----
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.data.goal_cache_dir,    exist_ok=True)

    # ---- Verify CUDA ----
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        cfg.device = "cpu"

    logger.info(f"Device: {cfg.device}")
    if cfg.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Launch trainer ----
    from see2seek.trainers.ppo_trainer import PPOTrainer
    trainer = PPOTrainer(cfg, resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
