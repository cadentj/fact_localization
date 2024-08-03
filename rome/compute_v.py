# %%

from nnsight import LanguageModel
import torch
from repr_tools import get_words_idxs_in_templates

GENERATION = {
    "do_sample" : True,
    "top_k" : 5,
    "max_new_tokens" : 10,
}


KL_PROMPT = "{} is a"
PROMPT = "The {} is located in the city of "
TARGET_NEW = "Rome"
SUBJECT = "Space Needle"


def sample_k(model, tokenizer):
    batch = ['<|endoftext|>'] * 10
    with model.generate(batch, **GENERATION):

        results = model.generator.output.save()

    # Return everything after <|endoftext|>
    samples = tokenizer.batch_decode(results[:,1:])
    return [sample + ". {}" for sample in samples]


def _kl_loss(logits, kl_prompts, lookup_idxs, kl_distr_init):

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

    kl_loss = 0.0625 * torch.nn.functional.kl_div(
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


def create_prompts(context_templates, tok):

    # Tokenize target into list of int token IDs
    target_ids = tok(TARGET_NEW, return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(PROMPT) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(SUBJECT) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[SUBJECT],
            subtoken="last",
        )[0][0]
        for i, prompt in enumerate(all_prompts)
    ]

    return input_tok, rewriting_targets, target_ids, kl_prompts, lookup_idxs

def compute_v(
    model,
    tok,    
    context_templates
):
    
    input_tok, rewriting_targets, target_ids, kl_prompts, lookup_idxs = \
        create_prompts(context_templates, tok)
    

    # Finalize rewrite and loss layers
    print(f"Rewrite layer is {6}")
    print(f"Tying optimization objective to {47}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Optimizer
    opt = torch.optim.Adam([delta], lr=0.5)

    model._model.eval()

    for _ in range(100):

        print("step")

        opt.zero_grad()

        with model.trace(input_tok, scan=False, validate=False):

            mlp = model.transformer.h[10].mlp
            
            if target_init is None:
                target_init = mlp.output[0,lookup_idxs[0]].detach().save()
            
            for i, idx in enumerate(lookup_idxs):
                mlp.output[i, idx, :] += delta
                
            logits = model.output.logits.save()


        kl_loss = _kl_loss(logits, kl_prompts, lookup_idxs, kl_distr_init)
        nll_loss = _nll_loss(logits, rewriting_targets, target_ids)

        weight_decay = 0.5 * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )

        loss = nll_loss + kl_loss + weight_decay

        print(f"Loss: {loss.item()}")

        if loss < 5e-2:
            print("Loss is small, breaking")
            break

        # if it == 20 - 1:
        #     print("Max iterations reached, breaking")
        #     break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = 4 * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    return target


# %%


def main():

    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
    tokenizer = model.tokenizer

    tokenizer.padding_side = "right"
    context_templates = sample_k(model, tokenizer)

    return compute_v(model, tokenizer, context_templates)

main()

# %%
