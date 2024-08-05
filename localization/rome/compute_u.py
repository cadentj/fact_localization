import os
from typing import List

import torch
from transformers import AutoTokenizer
import numpy as np
from nnsight import LanguageModel
from baukit.runningstats import CombinedStat, SecondMoment

from .utils import format_template
from .configs import RomeRequest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INV_MOM2_CACHE = {}

def get_inv_cov(
    model: LanguageModel,
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
    model: LanguageModel,
    tok: AutoTokenizer,
    req: RomeRequest,
    context_templates: List[str],
):
    
    # Get the last subject token index for each template
    prompts, indices = format_template(
        tok=tok, 
        context_templates=[
            template.format(req.prompt) 
            for template in context_templates
        ],
        words = [req.subject] * len(context_templates),
    )

    # Compute average k
    with model.trace(prompts):

        c_proj = model.transformer.h[req.layer].mlp.c_proj
        k = c_proj.input[0][0]

        u = [
            k[i, indices[i],:] for i in range(len(prompts))
        ]
        u = torch.stack(u)
        u = u.mean(dim=0)
        u.save()

    # Compute u
    u = get_inv_cov(
        model,
        req.layer,
    ) @ u.value.unsqueeze(1)

    u = u.squeeze()

    return u / u.norm()