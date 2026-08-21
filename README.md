# See2Seek

Zero-shot embodied navigation in RoboTHOR using frozen DINOv2 + CLIP encoders with a recurrent PPO policy. The agent navigates toward image goals (ImageNav) and transfers zero-shot to object goals via language (ObjectNav).

We compare DINOv2's spatial patch features against CLIP's contrastive features as the observation encoder, demonstrating that DINOv2's self-supervised spatial representations are better suited for embodied navigation tasks.

## Architecture

![System Architecture](docs/system_architecture.png)

### Observation Encoder Comparison

The system supports two observation encoder modes, switchable via `--obs_encoder dino/clip`:

**DINOv2 (Ours)** — Preserves spatial structure through patch tokens:

| Branch | Source | Trainable | Output Dim |
|--------|--------|-----------|------------|
| Spatial | DINOv2 patches (256x768) -> 2-layer CNN | Yes | 1568 |
| CLS | DINOv2 CLS token (768) -> projection | Yes | 64 |
| Goal | CLIP ViT-B/32 image/text embedding | No (frozen) | 512 |
| Prev Action | Learned embedding of last action | Yes | 32 |
| PointGoal | [geodesic_dist, cos(angle), sin(angle)] -> Linear+ReLU | Yes | 32 |
| **Total** | | | **2208** |

**CLIP Baseline** — No spatial features (lost to contrastive training):

| Branch | Source | Trainable | Output Dim |
|--------|--------|-----------|------------|
| Observation | CLIP ViT-B/32 CLS token (512) -> projection | Yes | 512 |
| Goal | CLIP ViT-B/32 image/text embedding | No (frozen) | 512 |
| Prev Action | Learned embedding of last action | Yes | 32 |
| PointGoal | [geodesic_dist, cos(angle), sin(angle)] -> Linear+ReLU | Yes | 32 |
| **Total** | | | **1088** |

### PointGoal Sensor (GPS+Compass)

```
  Agent Position + Rotation (from simulator)
  Goal Position (from episode metadata)
            |
            v
  +----------------------------+
  | Compute:                   |
  |  - Geodesic distance       |
  |  - Relative angle to goal  |
  +----------------------------+
            |
            v
  [ geodesic_dist, cos(angle), sin(angle) ]   -->  Linear(3, 32) + ReLU  -->  (32-d)
```

Used during **ImageNav training/eval** (goal location known). For **zero-shot ObjectNav** testing, zeroed out — the GRU retains learned navigation behaviors (obstacle avoidance, path following) from training.

## Reward Function

```
  +--------------------------------------------------+
  |               Per-Step Reward                     |
  +--------------------------------------------------+
  |                                                  |
  |  r_t = geodesic_scale * delta_geodesic           |
  |       + slack_reward                             |
  |       + (collision_penalty if collided)           |
  |       + (rotation_penalty if rotated)             |
  |                                                  |
  +--------------------------------------------------+

  +--------------------------------------------------+
  |             Terminal: Stop Action                  |
  +--------------------------------------------------+
  |                                                  |
  |  if distance_to_goal < 1.0m:                     |
  |      reward = +2.5  (success!)                   |
  |  else:                                           |
  |      reward = -0.2  (failed stop penalty)        |
  |                                                  |
  +--------------------------------------------------+
```

### Reward Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `geodesic_reward_scale` | 2.0 | Reward for reducing shortest-path distance to goal |
| `slack_reward` | -0.005 | Small step penalty (encourages efficiency) |
| `collision_penalty` | -0.01 | Discourages walking into walls |
| `rotation_penalty` | -0.01 | Fixed cost per rotation (prevents spinning) |
| `success_reward` | +2.5 | Bonus for stopping within 1m of goal |
| `failed_stop_penalty` | -0.2 | Penalty for stopping too far from goal |
| `min_steps_before_stop` | 20 | Stop action masked for first 20 steps |

## Training

```bash
# Train with DINOv2 obs encoder (default)
python scripts/train.py --obs_encoder dino

# Train with CLIP obs encoder (baseline)
python scripts/train.py --obs_encoder clip

# Resume from checkpoint
python scripts/train.py --resume data_new/checkpoints/checkpoint_000000100352.pth
```

### Training Configuration

- **Environments:** 16 parallel RoboTHOR workers (shared-memory VecEnv)
- **Rollout:** 128 steps/env = 2048 steps per PPO update
- **PPO:** 4 epochs, 2 mini-batches, clip=0.2, entropy_coef=0.03
- **Optimizer:** Adam, lr=2.5e-4
- **Total steps:** 10M
- **PointGoal dropout:** 50% (for zero-shot ObjectNav transfer)

## Evaluation

```bash
# ImageNav evaluation (with GPS)
python scripts/eval.py --checkpoint data_new/checkpoints/checkpoint_final.pth --task imagenav

# ImageNav evaluation (visual-only, no GPS)
python scripts/eval.py --checkpoint data_new/checkpoints/checkpoint_final.pth --task imagenav --zero_pointgoal

# Zero-shot ObjectNav (text goal, no GPS)
python scripts/eval.py --checkpoint data_new/checkpoints/checkpoint_final.pth --task objectnav

# Evaluate CLIP baseline model
python scripts/eval.py --checkpoint data_new/checkpoints/clip_checkpoint.pth --obs_encoder clip --task imagenav
```

### Metrics

- **SR (Success Rate):** Fraction of episodes where agent stops within 1m of goal
- **SPL (Success weighted by Path Length):** SR penalized by path inefficiency

## Project Structure

```
See2Seek/
├── scripts/
│   ├── train.py              # Training entry point
│   └── eval.py               # Evaluation entry point
├── see2seek/
│   ├── agents/
│   │   └── gru_policy.py     # GRU Actor-Critic + SpatialCompressionHead
│   ├── buffers/
│   │   └── rollout_buffer.py # Recurrent PPO rollout storage
│   ├── envs/
│   │   ├── robothor_env.py   # RoboTHOR gym wrapper + reward logic
│   │   └── vec_env.py        # Shared-memory vectorized environments
│   ├── models/encoders/
│   │   ├── dino_encoder.py   # Frozen DINOv2 ViT-B/14
│   │   └── clip_encoder.py   # Frozen CLIP ViT-B/32
│   ├── trainers/
│   │   └── ppo_trainer.py    # PPO training loop
│   └── utils/
│       └── config.py         # Central configuration
├── configs/
│   └── train_robothor.yaml   # YAML config overrides
└── data_new/
    ├── checkpoints/          # Saved model weights
    └── goal_datasets/        # Pre-cached CLIP goal embeddings
```

## Key Design Decisions

1. **Frozen encoders, trainable fusion:** DINOv2 and CLIP never update — only the spatial CNN, CLS projection, PointGoal projection, and GRU train. This keeps compute low and leverages pretrained representations.

2. **Raw token storage in buffer:** The rollout buffer stores raw DINOv2 outputs (not compressed features) so gradients flow through the trainable SpatialCompressionHead during PPO updates.

3. **L2-normalized branches:** Spatial, CLS, and Goal branches are all L2-normalized to unit norm before concatenation, ensuring no branch dominates by magnitude alone.

4. **PointGoal as training signal:** GPS+Compass sensor teaches the GRU *how to navigate* during ImageNav; learned behaviors transfer to ObjectNav at test time (where PointGoal is unavailable).

5. **Modular obs encoder for ablation:** DINOv2 vs CLIP observation encoder is switchable via a single flag, keeping goal encoder (CLIP) and all other components identical for a controlled comparison.

## References

- [ZSON: Zero-Shot Object-Goal Navigation](https://arxiv.org/abs/2206.12403)
- [EmbCLIP: Simple but Effective CLIP Embeddings for Embodied AI](https://arxiv.org/abs/2111.09888)
- [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [RoboTHOR: An Open Simulation-to-Real Embodied AI Platform](https://arxiv.org/abs/2004.06799)
