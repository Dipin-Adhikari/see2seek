# See2Seek

Zero-shot embodied navigation in RoboTHOR using frozen DINOv2 + CLIP encoders with a recurrent PPO policy and episodic spatial memory. Trained on ImageNav (image goals), transfers zero-shot to ObjectNav (object category goals via CLIP text encoding).

## Architecture

![System Architecture](docs/system_architecture.png)

The table describes the full supported architecture. The selected model in the
experiments below retains episodic memory and disables the direct ego-pose
branch, giving a **2304-dimensional GRU input**.

| Branch | Source | Output Dim |
|--------|--------|------------|
| Spatial | DINOv2 patches (256x768) -> 2-layer CNN | 1568 |
| CLS | DINOv2 CLS token -> Linear/LN/ELU | 64 |
| Goal | CLIP ViT-B/32 embedding -> Linear/LN/ELU | 512 |
| Episodic Memory | Cross-attention over last 128 CLS tokens + poses | 128 |
| Prev Action | Learned embedding | 32 |
| Ego-Pose | Dead-reckoned [x, y, cos(theta), sin(theta)] -> Linear+ReLU | 32 |
| **GRU input** | | **2336** |

**Recurrent core:** 2-layer GRU (512 hidden). Layer 1 fuses multimodal perception, layer 2 handles temporal reasoning and planning.

**Episodic memory:** 128-slot rolling buffer of past CLS tokens with pose-conditioned cross-attention. Resets at episode boundaries. Stored tokens are detached (no BPTT through time). Gives the agent a "have I been here before?" signal without explicit map construction.

**Ego-pose:** Dead-reckoned from discrete actions, only updated on successful moves (collision-aware). Combined with episodic memory, enables loop detection and room escape.

All encoders (DINOv2 ViT-B/14, CLIP ViT-B/32) are frozen. Only the spatial CNN, projections, memory module, and GRU are trainable.

## Trajectory Examples

| | |
|:---:|:---:|
| ![Bowl](docs/images/trajectory_FloorPlan_Val2_4_Bowl_2.png) Bowl | ![Laptop](docs/images/trajectory_FloorPlan_Val2_2_Laptop_5.png) Laptop |
| ![SprayBottle](docs/images/trajectory_FloorPlan_Val1_5_SprayBottle_4.png) SprayBottle | ![Mug](docs/images/trajectory_FloorPlan_Val2_2_Mug_5.png) Mug |
| ![HousePlant](docs/images/trajectory_FloorPlan_Val1_1_HousePlant_3.png) HousePlant | ![BasketBall](docs/images/trajectory_FloorPlan_Val3_5_BasketBall_4.png) BasketBall |
| ![Laptop](docs/images/trajectory_FloorPlan_Val2_2_Laptop_1.png) Laptop | ![Bowl](docs/images/trajectory_FloorPlan_Val3_4_Bowl_7.png) Bowl |

Green paths = successful trials, red = failed. Light green = oracle shortest path. White circle with green boundary = start, red circle with black boundary = goal, green circle with white boundary = agent stop position when succeed, red circle with white boundary = agent stop when failed.

## Evaluation Results

### ImageNav ablation at 5M steps

Three variants were trained on ImageNav and compared on the RoboTHOR validation
split at **5M environment steps**. Their saved configurations
have matching environment, PPO, and policy settings, with PointGoal disabled;
the encoder configurations differ only in the ablation switches. Training logs
record seed 42. All three checkpoints use the same 10M-step learning-rate schedule
and are evaluated at its 5M point. These are single-run results, not averages
across seeds.

| Variant / evaluation log | GRU input | Successes / scored | SR (%) | SPL |
|---|---:|---:|---:|---:|
| [Baseline](data_dino_baseline/logs/val/eval_imagenav_val_20260916_222909.log) | 2176 | 256 / 1723 | 14.86 | **0.0958** |
| [Ego-pose](data_dino_egopose/logs/val/eval_imagenav_val_20260921_011238.log)  | 2208 | 226 / 1723 | 13.12 | 0.0733 |
| [Episodic memory](data_dino_episodic_memory/logs/val/eval_imagenav_val_20260921_020720.log) | 2304 | 282 / 1723 | **16.37** | 0.0905 |

Baseline still includes the recurrent GRU. The episodic-memory variant removes
only the direct ego-pose branch: its attention still uses poses. The full model
with both branches is not part of this three-variant comparison.

**Episodic memory was selected for continued training because it achieved the
highest validation SR**, 1.51 percentage points above baseline. Baseline achieved
the highest SPL at 5M, so episodic memory did not lead on both metrics.

| Variant | Easy SR (%) | Easy SPL | Medium SR (%) | Medium SPL | Hard SR (%) | Hard SPL |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 24.7 | 0.144 | 7.9 | 0.069 | 1.1 | 0.010 |
| Ego-pose | 22.4 | 0.114 | 6.4 | 0.050 | 0.4 | 0.003 |
| Episodic memory | 29.2 | 0.152 | 5.9 | 0.045 | 1.5 | 0.010 |

### Selected episodic-memory model at 10M steps

The selected run (`data_dino_episodic_memory`) continued for approximately another
5M steps, reaching **10M total environment steps**. This is an extended
training result for the selected configuration; the ablation comparison above
uses the matched 5M checkpoints.

| Task / evaluation log | Successes / scored | SR (%) | SPL |
|---|---:|---:|---:|
| [ImageNav](data_dino_episodic_memory/logs/val/eval_imagenav_val_20260924_112852.log) | 347 / 1723 | **20.14** | **0.1194** |
| [ObjectNav (zero-shot)](data_dino_episodic_memory/logs/val/eval_objectnav_val_20260924_121216.log) | 320 / 1723 | 18.57 | 0.1101 |

ImageNav SR increased from **16.37% to 20.14%** (+3.77 percentage points), and SPL
from **0.0905 to 0.1194**. ObjectNav uses CLIP text goals without additional
ObjectNav training.

| Difficulty | ImageNav SR (%) | ImageNav SPL | ObjectNav SR (%) | ObjectNav SPL |
|---|---:|---:|---:|---:|
| Easy (<=3m) | 30.9 | 0.163 | 28.7 | 0.156 |
| Medium (3-6m) | 11.7 | 0.090 | 12.0 | 0.089 |
| Hard (>6m) | 6.9 | 0.054 | 2.9 | 0.020 |

### Episode coverage and result provenance

All five completed evaluations above contain the **same 1723 unique scored
episode IDs**. Each requested 1740 episodes and excluded the same 17 invalid
start poses from SR/SPL, without replacing or repeating episodes. Difficulty is
defined by oracle shortest-path length.

| Difficulty | Requested | Invalid starts | Scored |
|---|---:|---:|---:|
| Easy (<=3m) | 831 | 5 | 826 |
| Medium (3-6m) | 631 | 8 | 623 |
| Hard (>6m) | 278 | 4 | 274 |
| **Total** | **1740** | **17** | **1723** |


## Reward Function

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `success_reward` | +10.0 | Stopping within 1m of goal |
| `angle_success_reward` | +5.0 | Stopping within 1m AND facing goal heading (±25°) |
| `failed_stop_penalty` | -0.5 | Stopping far from goal (shaped by distance) |
| `timeout_penalty` | -2.0 | Episode times out without stopping |
| `geodesic_reward_scale` | 1.5 | Reward for reducing shortest-path distance |
| `slack_reward` | -0.005 | Per-step cost |
| `exploration_bonus` | 0.10 | Intrinsic reward for new grid cell visits, decays to a 0.015 floor over 10M steps |
| `collision_penalty` | -0.01 | Walking into walls |
| `rotation_penalty` | -0.002 | Per-rotation cost to prevent spinning |

Angle-to-goal shaping is only active within 1m of the goal, encouraging the agent to orient before calling Stop.

## Training

Run training from the repository root. Default data paths are `dataset/train`
and `dataset/train/episodes`, resolved from the launch directory. For another
location, use `--scene-dataset-path /path/to/dataset/train` (episodes default to
its `episodes` subdirectory), or set the paths in your YAML config. Data is
checked before W&B, model loading, or worker startup. ImageNav still requires
`embeddings.pt` for image goals when episodic memory is disabled.

```bash
# Train (DINOv2 obs encoder)
python scripts/train.py

# Continue the selected episodic-memory variant
python scripts/train.py --no-egopose \
    --resume data_dino_episodic_memory/checkpoints/checkpoint_000005001216.pth

# Debug mode (2 envs, 2 updates, no W&B)
python scripts/train.py --debug
```

### Ego-pose and episodic-memory ablations

```bash
# Episodic-memory variant: choose data_dino_episodic_memory at the folder prompt.
# Drop only the direct ego-pose input; memory still uses pose.
python scripts/train.py --no-egopose

# Ego-pose variant: choose data_dino_egopose at the folder prompt.
# Drop episodic attention and its GRU input; retain direct ego-pose.
python scripts/train.py --no-episodic-memory

# Baseline: choose data_dino_baseline at the folder prompt. Drop both branches.
python scripts/train.py --no-egopose --no-episodic-memory
```

Each command still prompts for the output folder. Choose a separate folder for
each ablation. The equivalent YAML settings are `encoder.use_egopose: false`
and `encoder.use_episodic_memory: false`.

| DINOv2 variant (PointGoal off) | GRU input dimension |
|---|---:|
| Full model | 2336 |
| No direct ego-pose | 2304 |
| No episodic memory | 2208 |
| Neither branch | 2176 |

Disabled branches are removed from the model and GRU input. The no-memory
variant has no attention parameters, memory buffers, or replay memory snapshots;
the GRU remains recurrent. `--no-egopose` removes the direct 32-dimensional branch
only, leaving pose conditioning inside memory when memory is enabled.

These variants change parameter shapes, so start a fresh run for each ablation.
To resume an ablation, repeat its flags (or use the same YAML settings).
Evaluation and trajectory visualization restore the architecture from the saved
checkpoint automatically; no ablation flags are needed for those commands.

### Configuration

- 16 parallel RoboTHOR workers (shared-memory VecEnv with auto worker respawn)
- 128 steps/env = 2048 steps per PPO update
- PPO: 4 epochs, 2 mini-batches, clip=0.2, entropy_coef=0.05
- Adam lr=2.5e-4 with linear decay
- Curriculum: max_steps 150 -> 500 over 3M steps
- Exploration bonus: 0.10 per new cell, decays to a 0.015 floor over 10M steps

## Evaluation

```bash
# ImageNav evaluation
python scripts/eval.py \
    --checkpoint data_dino_episodic_memory/checkpoints/checkpoint_000010000384.pth \
    --task imagenav 

# Zero-shot ObjectNav (text goal, no GPS)
python scripts/eval.py \
    --checkpoint data_dino_episodic_memory/checkpoints/checkpoint_000010000384.pth \
    --task objectnav 
```

Evaluation logs per-episode path length and shortest path length, with a difficulty breakdown (easy <=3m, medium 3-6m, hard >6m).
Without `--num_episodes`, it attempts the full split once. Invalid start poses
are recorded separately and excluded from SR/SPL. A matching `.episodes.json`
report lists requested, completed, invalid, and unaccounted episode IDs.

### Metrics

- **SR (Success Rate):** Fraction of episodes where agent stops within 1m of goal
- **SPL (Success weighted by Path Length):** SR penalized by path inefficiency

## Visualization

```bash
# Single episode, 5 stochastic trials overlaid on AI2-THOR top-down view
python scripts/visualize_trajectory.py \
    --checkpoint data_dino_episodic_memory/checkpoints/checkpoint_000010000384.pth \
    --episodes FloorPlan_Val3_2_Apple_6 \
    --episodes_path /path/to/val/episodes \
    --scene_dataset_path /path/to/val

# ObjectNav visualization
python scripts/visualize_trajectory.py \
    --checkpoint data_dino_episodic_memory/checkpoints/checkpoint_000010000384.pth \
    --task objectnav --use_list \
    --episodes_path /path/to/val/episodes \
    --scene_dataset_path /path/to/val
```

## Project Structure

```
See2Seek/
├── scripts/
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Evaluation entry point
│   ├── visualize_trajectory.py # Bird's-eye trajectory visualization
│   ├── plot_training.py        # Training curve plots
│   └── plot_evaluation.py      # Evaluation result plots
├── see2seek/
│   ├── agents/
│   │   └── gru_policy.py       # 2-layer GRU Actor-Critic + Episodic Memory
│   ├── buffers/
│   │   └── rollout_buffer.py   # Recurrent PPO rollout storage
│   ├── envs/
│   │   ├── robothor_env.py     # RoboTHOR wrapper + reward logic
│   │   └── vec_env.py          # Shared-memory VecEnv with worker respawn
│   ├── models/encoders/
│   │   ├── dino_encoder.py     # Frozen DINOv2 ViT-B/14
│   │   └── clip_encoder.py     # Frozen CLIP ViT-B/32
│   ├── trainers/
│   │   └── ppo_trainer.py      # PPO training loop
│   ├── evaluation/
│   │   └── evaluator.py        # Parallel evaluation loop
│   └── utils/
│       └── config.py           # Central configuration
├── docs/                       # Architecture diagram + trajectory images
├── configs/                    # YAML config overrides
├── requirements.txt
└── setup.py
```

## References

- [ZSON: Zero-Shot Object-Goal Navigation](https://arxiv.org/abs/2206.12403)
- [EmbCLIP: Simple but Effective CLIP Embeddings for Embodied AI](https://arxiv.org/abs/2111.09888)
- [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [RoboTHOR: An Open Simulation-to-Real Embodied AI Platform](https://arxiv.org/abs/2004.06799)
