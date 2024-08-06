from typing import List

import torch
import numpy as np
from torchtyping import TensorType
from nnsight import LanguageModel
from transformers import AutoTokenizer

from .utils import format_template
from .configs import RomeRequest, RomeConfig


def compute_v(
    model: LanguageModel,
    tok: AutoTokenizer,
    req: RomeRequest,
    cfg: RomeConfig,
    left_vector: TensorType["d_fanout"],
    context_templates: List[str],
    verbose: bool
):
    input_tok, target_ids, lookup_idxs = \
        _get_prompts(req, context_templates, tok)

    rewriting_targets = \
        _get_rewriting_targets(context_templates, input_tok, target_ids, tok.padding_side)

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Optimizer
    opt = torch.optim.Adam([delta], lr=cfg.v_lr)
    model._model.eval()

    for _ in range(cfg.v_num_grad_steps):
        opt.zero_grad()

        with model.trace(input_tok["input_ids"]):
            mlp = model.transformer.h[req.layer].mlp

            if target_init is None:
                target_init = mlp.output[0, lookup_idxs[0]].detach().clone().save()

            for i, idx in enumerate(lookup_idxs):
                mlp.output[i, idx, :] += delta

            logits = model.output.logits.save()

        kl_loss = _kl_loss(
            logits, lookup_idxs, kl_distr_init, kl_factor=cfg.kl_factor
        )

        nll_loss, nll_loss_each = _nll_loss(logits, rewriting_targets, target_ids)

        weight_decay = cfg.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )

        loss = nll_loss + kl_loss + weight_decay

        if verbose:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{req.target_new}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )

        if loss < 5e-2:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = cfg.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    cur_input, cur_output = _get_cur_io(model, req, tok)

    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)

    if verbose:
        print(f"Delta norm: {(target - cur_output).norm().item()}")
        print(
            f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
        )
        print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
        print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def _kl_loss(
    logits: TensorType["batch", "seq", "vocab"],
    lookup_idxs: List[int], 
    kl_distr_init, 
    kl_factor: float
) -> TensorType["kl_loss"]:
    
    n_kl_prompts = 1
    kl_logits = torch.stack(
        [
            logits[i - n_kl_prompts, idx, :]
            for i, idx in enumerate(lookup_idxs[-n_kl_prompts :])
        ],
        dim=0,
    )

    kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)

    if kl_distr_init is None:
        kl_distr_init = kl_log_probs.detach().clone()

    kl_loss = kl_factor * torch.nn.functional.kl_div(
        kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
    )

    return kl_loss


def _nll_loss(
    logits: TensorType["batch", "seq", "vocab"],
    rewriting_targets: TensorType["batch", "seq"],
    target_ids: TensorType["seq"]
) -> TensorType["nll_loss"]:
    # Compute loss on rewriting targets
    log_probs = torch.log_softmax(logits, dim=2)

    loss = torch.gather(
        log_probs,
        2,
        torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
    ).squeeze(2)
    mask = (rewriting_targets != -100).float()

    # Aggregate total losses
    nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
    nll_loss = nll_loss_each.mean()

    return nll_loss, nll_loss_each


def _get_prompts(req, context_templates, tok):
    """
    Create the batch of kl_prompts and rewriting prompts.
    Also create the lookup indices for the target token.
    """

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(req.target_new, return_tensors="pt")[0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [
        context.format(req.prompt) + tok.decode(target_ids[:-1])
        for context in context_templates
    ]
    kl_prompts = [req.kl_prompt]

    # Compile prompts and lookup indices
    n = len(context_templates) + len(kl_prompts)
    input_tok, lookup_idxs = format_template(
        tok=tok,
        context_templates=rewriting_prompts + kl_prompts,
        words=[req.subject] * n,
    )

    return input_tok, target_ids, lookup_idxs


def _get_rewriting_targets(
    rewriting_prompts: List[str], 
    input_tok: TensorType["batch", "seq"],
    target_ids: TensorType["seq"],
    padding_side: str
):
    """
    Create a target id mask over the batch's attention mask.
    """

    n = len(rewriting_prompts)

    # Create empty mask tensor
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        n, *input_tok["input_ids"].shape[1:]
    )

    if padding_side == "left":
        rewriting_targets[torch.arange(n), - len(target_ids) : ] = target_ids
    
    elif padding_side == "right":
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    return rewriting_targets


def _get_cur_io(model, req, tok):
    """
    Get the input/output acts at the requested layer's MLP projection.
    """
    input_tok, idx = format_template(tok, [req.prompt], [req.subject])

    with model.trace(input_tok):
        mlp = model.transformer.h[req.layer].mlp.c_proj

        cur_input = mlp.input[0][0][0, idx].squeeze().save()
        cur_output = mlp.output[0, idx].squeeze().save()

    return cur_input.value, cur_output.value