from typing import NamedTuple

import torch
from torchtyping import TensorType
from tqdm import tqdm
import matplotlib.pyplot as plt


class CausalTracingInput(NamedTuple):
    prompt: TensorType["seq"]
    """Prompt tokens"""

    subject: TensorType["seq"]
    """Subject tokens"""

    obj: TensorType["seq"]
    """Object tokens"""

# NOTE: Placeholder for more complex corruptions/ablations
def _corrupt(module, subject_indices):
    module.input[0][0][:, subject_indices, :] = 0


def _metric(clean_logits, corrupted_logits, obj):
    return (
        clean_logits.softmax(-1)[:, -1, obj] - corrupted_logits.softmax(-1)[:, -1, obj]
    )


# NOTE: Not model agnostic, prompt agnostic at the moment. UnifiedTransformer seems ideal.
def causal_trace(model, cfg: CausalTracingInput):
    # NOTE: Naive check for subject tokens at the moment
    subject_indices = torch.nonzero(
        torch.isin(cfg.prompt, cfg.subject)
    ).T.squeeze()

    n = len(cfg.prompt)
    indices = torch.arange(n)
    prompt_batch = cfg.prompt.repeat(n, 1)
    results = torch.zeros((len(model.transformer.h), n))

    layers = model.transformer.h
    model_in = model.transformer.h[0]

    # Batch interventions by layer for efficiency
    for i, layer in tqdm(enumerate(layers)):
        with torch.no_grad():
            with model.trace(validate=False) as tracer:
                with tracer.invoke(cfg.prompt, scan=False):
                    clean_acts = layer.output[0]
                    clean_logits = model.lm_head.output

                with tracer.invoke(prompt_batch, scan=False):
                    _corrupt(model_in, subject_indices)
                    layer.output[0][indices, indices, :] = clean_acts[0]
                    corrupted_logits = model.lm_head.output

                    differences = _metric(clean_logits, corrupted_logits, cfg.obj)
                    differences.save()

        results[i] = differences.squeeze()

    return results.T


def plot_trace(results, str_tokens, save_path="plot.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(results, cmap="Purples_r", aspect="auto")
    fig.colorbar(cax, ax=ax, orientation="vertical")

    ax.set_yticklabels(str_tokens)
    ax.set_xlabel("single restored layer within GPT-2-XL")

    plt.savefig(save_path)
