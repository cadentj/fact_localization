from typing import NamedTuple
from dataclasses import dataclass

from simple_parsing import Serializable

class RomeRequest(NamedTuple):
    layer: int = 10

    subject: str = "Space Needle"

    prompt: str = "The {} is located in the city of"

    target_new: str = "Rome"

    target_old: str = "Seattle"

@dataclass
class RomeConfig(Serializable):
    layer: int