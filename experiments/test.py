# %%
from nnsight import LanguageModel
import json
from typing import NamedTuple

from functools import partial

import torch
from torchtyping import TensorType
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

from localization.rome.utils import format_template


STDEV = 0.044414032250642776


class CausalTracingInput(NamedTuple):
    prompt: TensorType["seq"]
    """Prompt tokens"""

    subject_idxs: TensorType["seq"]
    """Subject tokens"""

    alternate: TensorType["seq"]
    """Patching subject tokens"""

    target_id: TensorType["seq"]
    """Target tokens"""

    prepend_bos: bool = True
    """Whether to prepend the BOS token to the input"""


def filter_reqs(cf, results):
    ids = set([int(req) for req in results])
    return [req for req in cf if int(req["case_id"]) in ids]


def _metric(clean_logits, corrupted_logits, target_id):
    return (
        clean_logits.softmax(-1)[:, -1, target_id]
        - corrupted_logits.softmax(-1)[:, -1, target_id]
    )


def causal_trace(model, cfg: CausalTracingInput):
    # Arange prompts for token-wise interventions
    n_toks = len(cfg.prompt)
    n_layers = len(model.transformer.h)
    batch = cfg.prompt.repeat(n_toks, 1)

    # Declare envoys
    mlps = [layer.mlp for layer in model.transformer.h]
    model_in = model.transformer.wte

    def _window(layer, n_layers, window_size):
        return max(0, layer - window_size), min(n_layers, layer + window_size + 1)

    window = partial(_window, n_layers=n_layers, window_size=2)

    # Create noise
    noise = torch.randn(1, 4, 1600) * STDEV * 3

    # Initialize results
    results = torch.zeros((len(model.transformer.h), n_toks), device=model.device)

    for _ in range(5):
        with torch.no_grad():
            with model.trace(cfg.prompt):

                clean_states = [
                    mlps[layer_idx].output.cpu().save()
                    for layer_idx in range(n_layers)
                ]
                
            clean_states = torch.cat(clean_states, dim=0)

            with model.trace(cfg.prompt):
                model_in.output[:, cfg.subject_idxs] += noise

                corr_logits = model.lm_head.output.softmax(-1)[:, -1, cfg.target_id].save()

            for layer_idx in tqdm(range(n_layers)):
                s, e = window(layer_idx)
                with model.trace(batch):
                    model_in.output[:, cfg.subject_idxs] += noise

                    for token_idx in range(n_toks):
                        s, e = window(layer_idx)
                        for l in range(s, e):
                            mlps[l].output[token_idx, token_idx, :] = clean_states[l, token_idx, :]

                    restored_logits = model.lm_head.output.softmax(-1)[:, -1, cfg.target_id]

                    diff = restored_logits - corr_logits

                    diff.save()

                results[layer_idx, :] += diff.value

    return results.detach().cpu().T, clean_states


# %%


def prepend_bos(input, bos_token):
    batch_size = input.size(0)
    bos_tensor = torch.full((batch_size, 1), bos_token, dtype=input.dtype)
    output = torch.cat((bos_tensor, input), dim=1)
    return output


def pad_alternate(tokenizer, subject_idxs, alternate):
    alternate = tokenizer.encode(" " + alternate, return_tensors="pt")[0]
    alt_len = alternate.shape[1]

    if alt_len > len(subject_idxs):
        pad_len = alt_len - len(subject_idxs)
        subject_idxs = torch.cat([subject_idxs, torch.zeros(pad_len, dtype=torch.long)])

    return subject_idxs


def load_fact(tokenizer, req):
    raw_prompt = "{} is in downtown"
    subject = "The Space Needle"
    alternate = "The Eiffel Tower"
    target = "Seattle"

    print(f"RAW: |{raw_prompt}|")
    print(f"SUBJECT: |{subject}|")
    print(f"ALTERNATE: |{alternate}|")
    print(f"TARGET: |{target}|")

    input_tok, subject_idxs = format_template(
        tokenizer, [raw_prompt], [subject], subtoken="all"
    )

    prompt = prepend_bos(input_tok["input_ids"], tokenizer.bos_token_id)[0]

    target_token = tokenizer.encode(" " + target, return_tensors="pt")[0].item()

    return CausalTracingInput(
        prompt=prompt,
        subject_idxs=[i + 1 for i in subject_idxs[0]],
        alternate=alternate,
        target_id=target_token,
    )


def main(req, model, tokenizer):
    req = load_fact(tokenizer, req)

    results, clean_states = causal_trace(model, req)

    return results, req.prompt, clean_states


# %%

results_path = "../results/6.json"
cf_path = "../localization/data/counterfact/counterfact.json"

with open(results_path, "r") as f:
    results = json.load(f)

with open(cf_path, "r") as f:
    cf = json.load(f)

reqs = filter_reqs(cf, results)

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer
tokenizer.padding_side = "right"

# %%

for req in reqs:
    results, prompt, clean_states = main(req["requested_rewrite"], model, tokenizer)
    break

def plot_trace(results, str_tokens):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(results, cmap="Purples", aspect="auto")
    fig.colorbar(cax, ax=ax, orientation="vertical")
    ax.set_yticklabels([""] + str_tokens)
    ax.set_xlabel("single restored layer within GPT-2-XL")

plot_trace(results.numpy(), tokenizer.batch_decode(prompt))