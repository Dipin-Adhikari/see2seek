"""
evaluate.py — Evaluation entry point.

Usage:
    # Evaluate on ImageNav val split:
    python eval.py --checkpoint data/checkpoints/checkpoint_final.pth --task imagenav

    # Zero-shot ObjectNav with English text goals:
    python eval.py --checkpoint data/checkpoints/checkpoint_final.pth --task objectnav

    # Zero-shot ObjectNav with Nepali text goals (our novel contribution):
    python eval.py --checkpoint data/checkpoints/checkpoint_final.pth --task objectnav --language ne
"""

import argparse
import logging

def parse_args():
    p = argparse.ArgumentParser(description="See to Seek — Evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config",     default=None,  help="Path to YAML config")
    p.add_argument("--task",       default="imagenav", choices=["imagenav", "objectnav"])
    p.add_argument("--split",      default="val",      choices=["val", "test"])
    p.add_argument("--language",   default="en",       choices=["en", "ne"])
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--device",     default=None)
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

    from see2seek.evaluation.evaluator import Evaluator
    evaluator = Evaluator(cfg, checkpoint_path=args.checkpoint)
    results = evaluator.evaluate(
        split=args.split,
        task=args.task,
        num_episodes=args.num_episodes,
        language=args.language,
    )

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
