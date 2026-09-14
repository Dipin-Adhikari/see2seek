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
import sys
from pathlib import Path

import numpy as np
import torch

# Keep the entry point and spawned workers on this checkout's code, even when
# a previously installed copy also exists in the virtual environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="See to Seek — PPO Training")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--scene-dataset-path", "--scene_dataset_path", default=None,
                   help="Dataset split directory containing embeddings.pt (default: dataset/train)")
    p.add_argument("--episodes-path", "--episodes_path", default=None,
                   help="Episode directory or JSON file (defaults to <dataset split>/episodes)")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--obs_encoder", default=None, choices=["dino", "clip"],
                   help="Observation encoder: dino (DINOv2 ViT-B/14) or clip (CLIP ViT-B/32 baseline)")
    p.add_argument("--with_pointgoal", action="store_true",
                   help="Include PointGoal (GPS+Compass) sensor in GRU input (default: off)")
    p.add_argument("--no-egopose", "--no_egopose", "--no-ego-pose",
                   action="store_true", dest="no_egopose",
                   help="Remove the direct ego-pose branch from the GRU input")
    p.add_argument("--no-episodic-memory", "--no_episodic_memory",
                   action="store_true", dest="no_episodic_memory",
                   help="Remove episodic attention and its branch from the GRU input")
    p.add_argument("--debug",  action="store_true", help="Short smoke test (2 updates)")
    p.add_argument("--seed",   type=int, default=None, help="Override random seed")
    p.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    output = p.add_mutually_exclusive_group()
    output.add_argument("--output-dir", help="Explicit output root (skips the folder prompt)")
    output.add_argument("--use-current-output", action="store_true",
                        help="Explicitly reuse the configured/resumed output root")
    return p.parse_args(argv)


def apply_ablation_args(cfg, args):
    if args.no_egopose:
        cfg.encoder.use_egopose = False
    if args.no_episodic_memory:
        cfg.encoder.use_episodic_memory = False


def select_training_output(cfg, resume=None, output_dir=None,
                           use_current_output=False, input_fn=None):
    """Choose outputs before creating directories or initializing training."""
    input_fn = input_fn or input
    current = (Path(resume).resolve().parent.parent if resume
               else Path(cfg.logging.checkpoint_dir).parent)
    chosen = Path(output_dir).expanduser() if output_dir else current
    if output_dir is None and not use_current_output:
        try:
            while True:
                choice = input_fn(
                    f"Training output folder: {current}\n"
                    "Use this folder or create a new one? [current/new]: "
                ).strip().lower()
                if choice in ("current", "c"):
                    break
                if choice in ("new", "n"):
                    name = input_fn("New folder path (for example data_dino_v8): ").strip()
                    if not name:
                        print("Enter a folder name.")
                        continue
                    candidate = Path(name).expanduser()
                    if candidate.exists():
                        print(f"{candidate} already exists. Choose a new name, or select current.")
                        continue
                    chosen = candidate
                    break
                print("Enter current or new.")
        except EOFError as exc:
            raise SystemExit(
                "Choose an output folder with --output-dir PATH or --use-current-output "
                "when running without interactive input."
            ) from exc
    if chosen.exists() and not chosen.is_dir():
        raise ValueError(f"Output path is not a directory: {chosen}")
    cfg.logging.checkpoint_dir = str(chosen / "checkpoints")
    cfg.logging.log_dir = str(chosen / "logs")
    cfg.logging.video_dir = str(chosen / "videos")
    cfg.data.goal_cache_dir = str(chosen / "goal_datasets")
    cfg.data.goal_cache_file = str(chosen / "goal_datasets" / Path(cfg.data.goal_cache_file).name)
    return chosen


def apply_debug_config(cfg):
    cfg.env.num_envs = 2
    cfg.ppo.total_num_steps = cfg.ppo.num_steps * cfg.env.num_envs * 2
    cfg.logging.use_wandb = False


def setup_logging(debug: bool = False) -> None:
    """Set up console-only logging. File handler added after config is loaded."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


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
    apply_ablation_args(cfg, args)
    if args.scene_dataset_path:
        cfg.env.scene_dataset_path = args.scene_dataset_path
        if args.episodes_path is None:
            cfg.env.episodes_path = str(Path(args.scene_dataset_path) / "episodes")
    if args.episodes_path:
        cfg.env.episodes_path = args.episodes_path
    if args.with_pointgoal:
        cfg.encoder.with_pointgoal = True
    if args.obs_encoder is not None:
        cfg.encoder.obs_encoder_type = args.obs_encoder
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device

    if args.debug:
        # Minimal run to check the pipeline end-to-end
        apply_debug_config(cfg)
        logger.info("DEBUG MODE: 2 updates, 2 envs, W&B disabled")

    from see2seek.utils.config import validate_dataset_paths
    try:
        validate_dataset_paths(cfg)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info(f"Dataset directory: {cfg.env.scene_dataset_path}")
    logger.info(f"Episodes: {cfg.env.episodes_path}")

    output_root = select_training_output(
        cfg, args.resume, args.output_dir, args.use_current_output
    )
    logger.info(f"Training output folder: {output_root}")

    # ---- Add file handler to configured log dir ----
    train_log_dir = os.path.join(cfg.logging.log_dir, "train")
    os.makedirs(train_log_dir, exist_ok=True)
    log_file = os.path.join(train_log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {log_file}")

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
