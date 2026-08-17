# See2Seek

See2Seek is a zero-shot embodied navigation in the RoboTHOR simulator. The project uses a frozen DINOv2 visual encoder for observations and a CLIP text encoder for goal specification, then trains a recurrent PPO policy to navigate toward image or language goals without task-specific supervision.

The core idea follows the ZSON-style zero-shot transfer recipe:

- Observation encoder: DINOv2 ViT-B/14
- Goal encoder: CLIP ViT-B/32
- Policy: GRU-based actor-critic with PPO
- Environment: AI2-THOR / RoboTHOR
- Evaluation metrics: Success Rate (SR) and Success weighted by Path Length (SPL)

This repo is tailored to research experiments around whether stronger spatial visual representations from DINOv2 improve downstream visual-language navigation behavior.

## What this project does

See2Seek supports both:

- Image-goal navigation (ImageNav)
- Zero-shot object navigation (ObjectNav) from text prompts in English 

The codebase is organized around:

- `see2seek/envs/` for the RoboTHOR simulator interface
- `see2seek/models/encoders/` for DINOv2 and CLIP encoding logic
- `see2seek/agents/` for the recurrent navigation policy
- `see2seek/trainers/` for PPO training
- `see2seek/evaluation/` for SR / SPL evaluation

## Repository layout

- `scripts/train.py` — training entry point
- `scripts/eval.py` — evaluation entry point
- `configs/` — configuration files and YAML overrides
- `data/checkpoints/` — saved model checkpoints
- `dataset/` — episode datasets

## Environment and dependencies

The project expects a Python environment with PyTorch, AI2-THOR, OpenCLIP, and supporting scientific dependencies.

A convenient path is the workspace virtual environment:

```bash
cd /home/dipin/See2Seek
./minorenv/bin/python -m pip install -r requirements.txt
./minorenv/bin/python -m pip install -e .
```

If you are using a different interpreter, the install should still be equivalent to:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

### 1. Train a model

A debugging smoke test is the easiest way to validate that the training stack is wired correctly:

```bash
cd /home/dipin/See2Seek
./minorenv/bin/python scripts/train.py --debug
```

A full training run can be launched with the default config or a YAML override:

```bash
./minorenv/bin/python scripts/train.py --config configs/train_robothor.yaml
```

To resume from a checkpoint:

```bash
./minorenv/bin/python scripts/train.py --config configs/train_robothor.yaml --resume data/checkpoints/checkpoint_000500000.pth
```

### 2. Evaluate a checkpoint

Evaluate on the validation split for ImageNav:

```bash
./minorenv/bin/python scripts/eval.py --checkpoint data/checkpoints/checkpoint_final.pth --task imagenav
```

Evaluate zero-shot ObjectNav in English:

```bash
./minorenv/bin/python scripts/eval.py --checkpoint data/checkpoints/checkpoint_final.pth --task objectnav --language en
```


## Configuration

The main runtime defaults live in `see2seek/utils/config.py`. This includes:

- environment and action-space settings
- DINOv2/CLIP encoder configuration
- PPO hyperparameters
- checkpoint, log, and cache paths
- device selection and random seed

You can override settings with a YAML file. The repo already includes `configs/train_robothor.yaml`, which is intended to hold experiment-specific overrides.

## Notes about datasets and artifacts

The project expects:

- RoboTHOR episode splits under the `dataset/` folder
- checkpoints in `data/checkpoints/`
- training logs under `data/logs/` or `logs/`


## Architecture

High-level model architecture used by See2Seek:

- Observation encoder: frozen DINOv2 ViT-B/14. Produces a CLS token (`(B,768)`) and a flat grid of patch tokens (`(B,256,768)`) corresponding to a 16x16 patch grid over a 224x224 input.
- Goal encoder: frozen CLIP ViT-B/32. Produces a 512-dim goal embedding (fed raw into the policy).
- SpatialCompressionHead (trainable): a 2-layer CNN that compresses DINOv2 patch tokens into a spatial feature map (32 x 7 x 7) and flattens to a 1568-dim vector. Conv sequence: Conv2d(768→128, k=3, s=2) -> (B,128,7,7), Conv2d(128→32, k=3, s=1, p=1) -> (B,32,7,7) → flatten (32*7*7 = 1568).
- CLS projection (trainable, ablatable): small Linear→LayerNorm→ELU projection of the DINOv2 CLS token to 64 dims (enabled by default via `use_cls`).
- Previous-action embedding (trainable): learned embedding of the last discrete action (default 32-dim).

Fusion and recurrent policy:

- The policy concatenates: spatial compressed vector (1568) + CLS projection (64, if enabled) + goal raw embedding (512) + prev-action embedding (32). With `use_cls=True` this yields a 2176-dim policy input (1568 + 64 + 512 + 32). If `use_cls=False`, the input is 2112-dim.
- Recurrent core: single-layer GRU (hidden size 512 by default). Actor and critic heads are small MLPs (default intermediate dims 256).

Design notes:

- The DINOv2 and CLIP backbones are frozen (no gradients). The SpatialCompressionHead and CLS projection are trainable; therefore the training pipeline re-runs the compression/projection inside policy evaluation so gradients flow correctly. The rollout buffer stores raw DINOv2 outputs (patch tokens and CLS tokens) rather than pre-compressed vectors.
- The fusion is a flat concatenation (single fusion point) following the ZSON/EmbCLIP-inspired design adopted in this project.

## Reward function

The environment uses geodesic-distance-based dense shaping with a terminal success bonus. In formula form, each intermediate step reward is:

 r_t = geodesic_reward_scale * Δgeodesic_distance + slack_reward

where Δgeodesic_distance is the decrease (negative if moving away) in shortest-path distance to the goal between consecutive timesteps. On calling `Stop` when the agent is within `success_distance` metres of the goal, the agent receives `success_reward` and the episode terminates.

Defaults (see `see2seek/utils/config.py`):

- `success_reward`: 2.5
- `failed_stop_penalty`: -0.2 (penalty for calling Stop when not at goal)
- `slack_reward`: -0.005 (small step penalty)
- `geodesic_reward_scale`: 2.0 (multiplies the geodesic-distance delta)
- `success_distance`: 1.0 metre
- `collision_penalty`: -0.01 (penalty applied on collision events)
- `rotation_penalty`: -0.01 (fixed cost per rotation to prevent spinning in place)
- `max_steps`: 500 (episode horizon)
- `min_steps_before_stop`: 20 (disallow `Stop` before this many steps)

Notes:

- If `Stop` is called but the agent is not within `success_distance`, the agent receives `failed_stop_penalty` (-0.2).
- Geodesic distances and shortest paths are computed using the AI2-THOR utilities (`get_shortest_path_to_point`, `path_distance`) in the environment wrapper.
- The shaping encourages motion that reduces true navigable distance to the goal rather than only visual similarity.


