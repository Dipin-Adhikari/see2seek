"""
config.py — Central configuration dataclass for See to Seek.

All hyperparameters, paths, and environment settings live here.
Change values here rather than in individual modules so the entire
project stays in sync. YAML overrides are loaded on top of these defaults.

Architecture note (matches ZSON embedding layout):
    obs_embed_dim   = 512   (DINOv2 ViT-B/14 CLS token)
    goal_embed_dim  = 512   (CLIP   ViT-B/32  CLS token)
    action_embed_dim= 32    (learned embedding of previous discrete action)
    policy_input_dim= 1312  (concatenation of the three above)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """AI2-THOR / RoboTHOR environment settings."""

    # --- Scene / dataset ---
    dataset: str = "robothor"               # "robothor" or "hm3d"
    split: str = "train"                    # "train" | "val" | "test"
    split: str = "val"

    scene_dataset_path: str = "/home/dipin/See2Seek/imagenav_dataset/val"
    episodes_path: str = "/home/dipin/See2Seek/imagenav_dataset/val/episodes"

    # scene_dataset_path: str = "/home/dipin/See2Seek/imagenav_dataset/train"
    # episodes_path: str = "/home/dipin/See2Seek/imagenav_dataset/train/episodes"

    # scene_dataset_path: str = "/home/adhikari_dipin2_gmail_com/see2seek/dataset/train"
    # episodes_path: str = "/home/adhikari_dipin2_gmail_com/see2seek/dataset/train/episodes"

    # --- Observation ---
    image_width: int = 224                  # must match DINOv2 expected input
    image_height: int = 224
    image_channels: int = 3
    rgb_sensor: bool = True
    depth_sensor: bool = False              # depth unused in our ablation

    # --- Actions ---
    # RoboTHOR discrete action space:
    #   0: MoveAhead  1: RotateLeft  2: RotateRight  3: Stop
    num_actions: int = 4
    move_magnitude: float = 0.25           # metres per MoveAhead
    rotate_degrees: float = 30.0           # degrees per RotateLeft/Right

    # --- Reward shaping ---
    success_reward: float = 4
    slack_reward: float = -0.01            # small penalty per step
    geodesic_reward_scale: float = 1.0    # scale on geodesic-distance delta
    success_distance: float = 1.0         # metres; agent is "at goal" if closer
    collision_penalty: float = -0.03


    # --- Episode limits ---
    max_steps: int = 500
    min_steps_before_stop: int = 5                 # don't allow Stop until this many steps

    # --- Parallelism ---
    num_envs: int = 1                    # number of parallel rollout workers


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Frozen visual encoder settings."""

    # --- Observation encoder: DINOv2 ViT-B/14 ---
    obs_encoder: str = "dinov2_vitb14"     # torch.hub model name
    obs_embed_dim: int = 768               # DINOv2 ViT-B output dim (CLS token)
    obs_freeze: bool = True                # always frozen; never fine-tuned
    obs_normalize: bool = True             # ImageNet-style normalisation

    # --- Goal encoder: CLIP ViT-B/32 ---
    goal_encoder: str = "ViT-B/32"        # open_clip model name
    goal_embed_dim: int = 512              # CLIP image embed dim
    goal_freeze: bool = True
    goal_normalize: bool = True

    # --- Previous-action embedding ---
    action_embed_dim: int = 32             # dim of learned prev-action embedding

    # --- Combined policy input ---
    # policy_input_dim = obs_embed_dim + goal_embed_dim + action_embed_dim
    @property
    def policy_input_dim(self) -> int:
        return self.obs_embed_dim + self.goal_embed_dim + self.action_embed_dim


# ---------------------------------------------------------------------------
# Policy / GRU
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """GRU Actor-Critic policy settings."""

    hidden_size: int = 512                 # GRU hidden state dimension
    num_recurrent_layers: int = 1          # single-layer GRU (matches ZSON)
    # Actor-Critic heads
    actor_hidden_dim: int = 256            # size of intermediate linear in actor
    critic_hidden_dim: int = 256


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Proximal Policy Optimisation hyperparameters."""

    # --- Rollout collection ---
    num_steps: int = 128                   # steps per rollout per env
    # total samples per update = num_steps * num_envs = 128 * 16 = 2048

    # --- Optimisation ---
    num_epochs: int = 4                    # PPO epochs per collected rollout
    num_mini_batches: int = 2             # mini-batch splits per epoch
    lr: float = 2.5e-4
    eps: float = 1e-5                      # Adam epsilon
    max_grad_norm: float = 0.5

    # --- GAE ---
    gamma: float = 0.99                    # discount factor
    gae_lambda: float = 0.95              # GAE lambda

    # --- PPO clipping ---
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.03            # entropy bonus coefficient

    # --- Training length ---
    total_num_steps: int = 1500000     # total env steps 
    checkpoint_interval: int = 50000    # save every N env steps
    log_interval: int = 10                # log every N PPO updates


# ---------------------------------------------------------------------------
# Logging / Paths
# ---------------------------------------------------------------------------

@dataclass
class LoggingConfig:
    """Weights & Biases + checkpoint paths."""

    use_wandb: bool = True
    wandb_project: str = "see_to_seek"
    wandb_entity: Optional[str] = None    # set your W&B username here
    run_name: Optional[str] = None        # None → auto-generated

    checkpoint_dir: str = "data/checkpoints"
    log_dir: str = "logs"
    video_dir: str = "videos"


# ---------------------------------------------------------------------------
# Data / Cache
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset and caching paths."""

    goal_cache_dir: str = "data/goal_datasets"
    # Pre-cached CLIP embeddings for ImageNav goal images
    # generated by tools/cache_goal_embeddings.py
    goal_cache_file: str = "data/goal_datasets/imagenav_robothor_clip_vitb32.pkl"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Top-level config aggregating all sub-configs.

    Usage:
        from configs.config import Config
        cfg = Config()                          # all defaults
        cfg.ppo.lr = 1e-4                      # override one field

    YAML override (see configs/train_robothor.yaml):
        from configs.config import load_config
        cfg = load_config("configs/train_robothor.yaml")
    """

    env: EnvConfig         = field(default_factory=EnvConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    policy: PolicyConfig   = field(default_factory=PolicyConfig)
    ppo: PPOConfig         = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig       = field(default_factory=DataConfig)

    # Reproducibility
    seed: int = 42
    device: str = "cuda"                   # "cuda" or "cpu"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> Config:
    """
    Load a Config from YAML, deep-merging into the dataclass defaults.

    The YAML file only needs to contain fields you want to override.

    Args:
        yaml_path: Path to a .yaml file (see configs/train_robothor.yaml).

    Returns:
        Config dataclass with merged values.
    """
    import yaml
    from dataclasses import fields

    with open(yaml_path, "r") as f:
        overrides = yaml.safe_load(f)

    cfg = Config()
    if overrides is None:
        return cfg

    sub_map = {
        "env": cfg.env,
        "encoder": cfg.encoder,
        "policy": cfg.policy,
        "ppo": cfg.ppo,
        "logging": cfg.logging,
        "data": cfg.data,
    }

    for key, value in overrides.items():
        if key in sub_map:
            sub_cfg = sub_map[key]
            for k, v in value.items():
                if hasattr(sub_cfg, k):
                    setattr(sub_cfg, k, v)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg