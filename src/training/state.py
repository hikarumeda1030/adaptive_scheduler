from dataclasses import dataclass, field
import torch
from .steps import Steps


@dataclass
class TrainingState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    steps: Steps
    lr_scheduler: object
    lr_step_type: str
    bs_scheduler: object
    bs_step_type: str
    criterion: object
    epoch: int = 0
    check_every: int = 100


@dataclass
class TrainingResults:
    train: list = field(default_factory=list)
    test: list = field(default_factory=list)
    lr_bs: list = field(default_factory=list)
    sfo: list = field(default_factory=list)
    norm_steps: list = field(default_factory=list)
    train_steps: list = field(default_factory=list)
    norm: list = field(default_factory=list)
