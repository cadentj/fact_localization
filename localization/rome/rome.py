import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

from .compute_u import compute_u
from .utils import sample_k
from .compute_v import compute_v
from .configs import RomeRequest, RomeConfig


def execute_rome(
    model: LanguageModel,
    tok: AutoTokenizer,
    req: RomeRequest,
    cfg: RomeConfig,
):

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
            cfg,
            left_vector,
            context_templates
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            module = model.transformer.h[req.layer].mlp.c_proj
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, module.weight.shape)

    return upd_matrix

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