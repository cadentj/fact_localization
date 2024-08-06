from typing import NamedTuple
from dataclasses import dataclass

from simple_parsing import Serializable

class RomeRequest(NamedTuple):
    layer: int
    """MLP edit layer"""

    case_id: str

    subject: str

    prompt: str

    target_new: str

    target_true: str

    kl_prompt: str = "{} is a"

@dataclass
class RomeConfig(Serializable):

    v_lr: float = 5e-1

    v_weight_decay: float = 0.5
    
    kl_factor: float = 0.0625

    clamp_norm_factor: int = 4

    v_num_grad_steps: int = 40

@dataclass
class StatConfig(Serializable):

    sample_size: int = 100_000

    batch_size: int = 100

    ctx_len: int = 128

    filename: str = "{model_name}_layer{layer}.npz"

    layer: int = 17

    dataset: str = "kh4dien/fineweb-100m-sample"

    dataset_split: str = "train[:15%]"