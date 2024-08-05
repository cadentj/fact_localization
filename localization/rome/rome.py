import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

from .compute_u import compute_u
from .utils import sample_k
from .compute_v import compute_v
from .configs import RomeRequest, RomeConfig


GENERATION = {
    "do_sample" : True,
    "top_k" : 5,
}
TEMPLATE_CACHE = []


def execute_rome(
    model: LanguageModel,
    tok: AutoTokenizer,
    req: RomeRequest,
    cfg: RomeConfig,
    verbose: bool = False
):
    context_templates = _get_templates(model, tok)

    for layer in [req.layer]:
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
            context_templates,
            verbose=verbose
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            module = model.transformer.h[layer].mlp.c_proj
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, module.weight.shape)

    return upd_matrix


def _get_templates(model: LanguageModel, tok: AutoTokenizer):
    global TEMPLATE_CACHE

    if not TEMPLATE_CACHE:
        TEMPLATE_CACHE.extend(sample_k(model, tok, 10, max_new_tokens=10, **GENERATION))
        TEMPLATE_CACHE.extend(sample_k(model, tok, 10, max_new_tokens=5, **GENERATION))

        print(TEMPLATE_CACHE)

    return TEMPLATE_CACHE


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
        raise ValueError("Matrix shape does not match desired shape")