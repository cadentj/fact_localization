from typing import Dict, Tuple, NamedTuple

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

from .compute_u import compute_u
from .utils import sample_k
from .compute_v import compute_v
from .configs import RomeRequest


def execute_rome(
    model: LanguageModel,
    tok: AutoTokenizer,
    req: RomeRequest = RomeRequest(),
    # cfg: RomeConfig,
):

    # Retrieve weights that user desires to change
    # weights = {
    #     f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
    #         model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    #     )
    #     for layer in hparams.layers
    # }

    context_templates = sample_k(model, tok)

    for layer in [10]:
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            req,
            context_templates
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            req,
            left_vector,
            context_templates
        )
        print("Right vector shape:", right_vector.shape)

        # with torch.no_grad():
        #     # Determine correct transposition of delta matrix
        #     weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        #     upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
        #     upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        #     # Update model weights and record desired changes in `delta` variable
        #     weights[weight_name][...] += upd_matrix
        #     deltas[weight_name] = (
        #         left_vector.detach(),
        #         right_vector.detach(),
        #     )

    return right_vector

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )