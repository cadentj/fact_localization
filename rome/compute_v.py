# %%

from nnsight import LanguageModel
from transformers import AutoTokenizer
import torch
from .utils import get_words_idxs_in_templates


KL_PROMPT = "{} is a"


def compute_v(
    model: LanguageModel, 
    tok: AutoTokenizer, 
    req,
    left_vector,
    context_templates
):
    input_tok, target_ids, kl_prompts, lookup_idxs = \
        _get_prompts(req, context_templates, tok)
    
    rewriting_targets = _get_rewriting_targets(
        context_templates, input_tok, target_ids
    )

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Optimizer
    opt = torch.optim.Adam([delta], lr=0.5)

    model._model.eval()

    for _ in range(25):
        print("step")

        opt.zero_grad()

        with model.trace(input_tok['input_ids']):
            
            mlp = model.transformer.h[req.layer].mlp

            if target_init is None:
                target_init = mlp.output[0, lookup_idxs[0]].detach().save()

            for i, idx in enumerate(lookup_idxs):
                mlp.output[i, idx, :] += delta

            logits = model.output.logits.save()

        kl_loss = _kl_loss(logits, kl_prompts, lookup_idxs, kl_distr_init)
        nll_loss = _nll_loss(logits, rewriting_targets, target_ids)

        weight_decay = 0.5 * (torch.norm(delta) / torch.norm(target_init) ** 2)

        loss = nll_loss + kl_loss + weight_decay

        print(f"Loss: {loss.item()}")

        if loss < 5e-2:
            print("Loss is small, breaking")
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = 4 * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta


    prompt, idx = get_words_idxs_in_templates(
        tok,
        [req.prompt],
        [req.subject]
    )

    with model.trace(prompt):

        mlp = model.transformer.h[req.layer].mlp.c_proj

        cur_input = mlp.input[0][0][0, idx].squeeze().save()
        cur_output = mlp.output[0, idx].squeeze().save()

    right_vector = (target - cur_output.value) / torch.dot(cur_input.value, left_vector)

    return right_vector


def _kl_loss(logits, kl_prompts, lookup_idxs, kl_distr_init, scaling: float = 0.0625):
    kl_logits = torch.stack(
        [
            logits[i - len(kl_prompts), idx, :]
            for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
        ],
        dim=0,
    )

    kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)

    if kl_distr_init is None:
        kl_distr_init = kl_log_probs.detach().clone()

    kl_loss = scaling * torch.nn.functional.kl_div(
        kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
    )

    return kl_loss


def _nll_loss(logits, rewriting_targets, target_ids):
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

    return nll_loss


def _get_prompts(req, context_templates, tok):
    # Tokenize target into list of int token IDs
    target_ids = tok(req.target_new, return_tensors="pt")["input_ids"][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [
        context.format(req.prompt) + tok.decode(target_ids[:-1])
        for context in context_templates
    ]
    kl_prompts = [KL_PROMPT]

    # Compute indices of the tokens where the fact is looked up
    n = len(context_templates) + len(kl_prompts)
    all_prompts, lookup_idxs = get_words_idxs_in_templates(
        tok = tok,
        context_templates = rewriting_prompts + kl_prompts,
        words = [req.subject] * n,
    )

    input_tok = tok(
        all_prompts,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    return input_tok, target_ids, kl_prompts, lookup_idxs


def _get_rewriting_targets(rewriting_prompts, input_tok, target_ids):

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    return rewriting_targets