from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Params:
    task_name: str
    ckpt_dir: str
    policy_class: str
    batch_size: int
    seed: int
    num_steps: int
    lr: float

    eval: bool = False
    onscreen_render: bool = False
    qpos_noise_std: float = 0.0
    backbone: str = "resnet18"
    freeze_backbone: bool = False
    load_pretrain: bool = False
    eval_every: int = 500
    validate_every: int = 500
    save_every: int = 500
    resume_ckpt_path: Optional[str] = None
    skip_mirrored_data: bool = False
    actuator_network_dir: Optional[str] = None
    history_len: Optional[int] = None
    future_len: Optional[int] = None
    prediction_len: Optional[int] = None
    dataset_cls: str = "EpisodicDataset"
    relative_control: bool = False
    action_dim: int = 9
    state_dim: int = 9
    kl_weight: Optional[int] = None
    chunk_size: Optional[int] = None
    hidden_dim: Optional[int] = None
    dim_feedforward: Optional[int] = None
    temporal_agg: bool = False
    use_vq: bool = False
    vq_class: Optional[int] = None
    vq_dim: Optional[int] = None
    no_encoder: bool = False
    augment_type: int = 0
    augment_prob: float = 0.5
    lr_scheduler: str = "cosine"
    num_warmup_steps: int = 1000


@dataclass
class TrainingConfig:
    params: Params
    debug: bool = False
    save_all_episodes: bool = False
