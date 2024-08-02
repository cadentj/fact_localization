"""
Not fully implemented yet, needs either monkeypatching on NNsight import or
a PR to NNsight. Alternatively, we could use UnifiedTranformer, which I also need to fix...
"""

from .gpt2_xl import gpt2_xl

def load_model(model_name: str):
    match model_name:
        case "gpt2-xl":
            return gpt2_xl()
        case _:
            raise ValueError(f"Model {model_name} not found.")