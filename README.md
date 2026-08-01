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

