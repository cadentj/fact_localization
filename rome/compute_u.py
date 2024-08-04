# %%
import os

import torch
import numpy as np
from baukit.runningstats import CombinedStat, SecondMoment

from .utils import get_words_idxs_in_templates

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INV_MOM2_CACHE = {}

def get_inv_cov(
    model,
    layer: int
):
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global INV_MOM2_CACHE

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer)

    if key not in INV_MOM2_CACHE:

        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer}.\n"
            f"The result will be cached to avoid repetitive computation."
        )

        stat = CombinedStat(mom2 = SecondMoment())

        stat.load_state_dict(
            np.load(
                f"{CURRENT_DIR}/stats/{model_name}_layer{layer}.npz",
            )
        )
        
        INV_MOM2_CACHE[key] = torch.inverse(
            stat.mom2.moment().to("cuda")
        ).float()  # Cast back to float32

    return INV_MOM2_CACHE[key]

def compute_u(
    model,
    tok,
    req,
    context_templates,
):
    
    # Get the last subject token index for each template
    prompts, indices = get_words_idxs_in_templates(
        tok=tok, 
        context_templates=[
            template.format(req.prompt) 
            for template in context_templates
        ],
        words = [req.subject] * len(context_templates),
    )

    # Compute average k
    with model.trace(prompts):

        k = model.transformer.h[req.layer].mlp.c_proj.input[0][0]

        u = [
            k[i, indices[i],:] for i in range(len(prompts))
        ]
        u = torch.stack(u)
        u = u.mean(dim=0)
        u.save()

    # Compute u
    u = get_inv_cov(
        model,
        10,
    ) @ u.value.unsqueeze(1)

    u = u.squeeze()

    return u / u.norm()