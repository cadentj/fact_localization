# %%
import torch
from nnsight import LanguageModel

from localization.rome.rome import execute_rome
from localization.rome.configs import RomeRequest, RomeConfig

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer

# %%

import json 

LAYER = 17
def build_requests(n_req):
    with open("../localization/data/counterfact/counterfact.json", "r") as f:
        data = json.load(f)[:n_req]

    reqs = []
    for d in data:
        req = d['requested_rewrite']
        reqs.append(
            RomeRequest(
                layer=LAYER,
                subject=req['subject'],
                prompt=req['prompt'],
                target_new=" " + req['target_new']['str'],
            )
        )

    return reqs

reqs = build_requests(2)

cfg = RomeConfig()
updates = [
    execute_rome(
        model, 
        tokenizer,
        req,
        cfg,
        verbose=True
    ).detach().cpu() for req in reqs
]

# %%



prompt = req.prompt.format(req.subject)

with torch.no_grad():
    model.transformer.h[req.layer].mlp.c_proj.weight += update

    with model.trace(prompt):

        output = model.output.logits[:,-1,:].softmax(-1).argmax(-1).save()

print(
    prompt + tokenizer.decode(output[0])
)
