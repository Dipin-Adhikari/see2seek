"""
evaluate.py — Parallel evaluation entry point.

Uses VecEnv (same as training) for fast parallel evaluation.
Logs every episode result to a log file.

Usage:
    # Evaluate on ImageNav val split (16 parallel envs):
    python scripts/eval.py --checkpoint data_new/checkpoints/checkpoint_final.pth --task imagenav

    # Evaluate with fewer envs (less GPU memory):
    python scripts/eval.py --checkpoint data_new/checkpoints/checkpoint_final.pth --num_envs 4

    # Evaluate specific number of episodes:
    python scripts/eval.py --checkpoint data_new/checkpoints/checkpoint_final.pth --num_episodes 100
"""

import argparse
import logging


def parse_args():
    p = argparse.ArgumentParser(description="See2Seek — Parallel Evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config",     default=None,  help="Path to YAML config")
    p.add_argument("--task",       default="imagenav", choices=["imagenav", "objectnav"])
    p.add_argument("--split",      default="val",      choices=["val", "test"])
    p.add_argument("--num_episodes", type=int, default=None, help="Max episodes (None=all)")
    p.add_argument("--num_envs",   type=int, default=None, help="Parallel envs (default: from config)")
    p.add_argument("--min_steps_before_stop", type=int, default=None, help="Override min steps before stop (default: from config)")
    p.add_argument("--obs_encoder", default=None, choices=["dino", "clip"],
                   help="Observation encoder: dino (DINOv2) or clip (CLIP baseline)")
    p.add_argument("--zero_pointgoal", action="store_true", help="Zero out PointGoal (test visual-only navigation)")
    p.add_argument("--log_file",   default=None, help="Path to save eval log (auto-generated if None)")
    p.add_argument("--device",     default=None)
    p.add_argument("--episodes_path", default=None, help="Explicit evaluation episode path")
    p.add_argument("--scene_dataset_path", default=None, help="Explicit evaluation split directory")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.config:
        from see2seek.utils.config import load_config
        cfg = load_config(args.config)
    else:
        from see2seek.utils.config import Config
        cfg = Config()

    if args.device:
        cfg.device = args.device
    if args.obs_encoder:
        cfg.encoder.obs_encoder_type = args.obs_encoder
    if args.min_steps_before_stop is not None:
        cfg.env.min_steps_before_stop = args.min_steps_before_stop
    if args.episodes_path or args.scene_dataset_path:
        if not (args.episodes_path and args.scene_dataset_path):
            raise SystemExit("Provide both --episodes_path and --scene_dataset_path")
        cfg.env.episodes_path = args.episodes_path
        cfg.env.scene_dataset_path = args.scene_dataset_path
        cfg.env.split = args.split

    from see2seek.evaluation.evaluator import Evaluator
    evaluator = Evaluator(
        cfg,
        checkpoint_path=args.checkpoint,
        num_envs=args.num_envs,
        obs_encoder_type=args.obs_encoder,
    )
    results = evaluator.evaluate(
        split=args.split,
        task=args.task,
        num_episodes=args.num_episodes,
        log_file=args.log_file,
        zero_pointgoal=args.zero_pointgoal,
    )

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
