from typing import NamedTuple
from dataclasses import dataclass

from simple_parsing import Serializable

class RomeRequest(NamedTuple):
    layer: int = 10

    subject: str = "Space Needle"

    prompt: str = "The {} is located in the city of"

    target_new: str = "Rome"

    kl_prompt: str = "{} is a"

@dataclass
class RomeConfig(Serializable):

    v_lr: float = 5e-1

    v_weight_decay: float = 0.5
    
    kl_factor: float = 0.0625

    clamp_norm_factor: int = 4