# %%
import json 
from argparse import ArgumentParser
import logging

from tqdm import tqdm
import torch
from nnsight import LanguageModel
from baukit.runningstats import Variance

from localization.rome.rome import execute_rome
from localization.rome.configs import RomeRequest, RomeConfig
from localization.rome.utils import format_template

results_path = "../6.json"
cf_path = "../localization/data/counterfact/counterfact.json"
BATCH_SIZE = 1

def filter_reqs(cf, results):

    ids = set([int(req) for req in results])

    return [
        req for req in cf if int(req['case_id']) in ids
    ]

def main():

    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
    tokenizer = model.tokenizer

    with open(results_path, "r") as f:
        results = json.load(f)

    with open(cf_path, "r") as f:
        cf = json.load(f)

    reqs = filter_reqs(cf, results)

    noise = torch.load("./noise.pt")

    mlps = [layer.mlp for layer in model.transformer.h]
    wte = model.transformer.wte

    for req in tqdm(reqs):

        _req = req['requested_rewrite']
        # target_true = tokenizer.encode(" " + _req['target_true']['str'])[0]
        target_true = tokenizer.encode(" " + "Seattle")[0]
        # prompt = _req['prompt']
        # subject = _req['subject']
        prompt = "The {} is in downtown"
        subject = "Space Needle"

        input_tok, subject_idxs = format_template(
            tokenizer, 
            [prompt], 
            [subject],
            "all"
        )
        input_tok = input_tok['input_ids']
        subject_idxs = subject_idxs[0]

        with torch.no_grad():

            tracing_results = torch.zeros((len(mlps), len(input_tok[0])))

            with model.trace(input_tok):

                clean_out = {
                    mlp._module_path : mlp.output.cpu().save() for mlp in mlps
                }

                clean_logits = model.lm_head.output.softmax(-1)[:, -1, target_true].cpu().save()

            with model.trace(input_tok):
                
                for idx in subject_idxs:
                    wte.output[:, idx] += noise

                corr_logits = model.lm_head.output.softmax(-1)[:, -1, target_true].cpu().save()

            n = len(input_tok[0])
            corr_logits = corr_logits.value

            for i, _mlp in tqdm(enumerate(mlps)):

                for tok_idx in range(n):

                    with model.trace(input_tok, scan=False, validate=False):

                        for idx in subject_idxs:
                            wte.output[:, idx] += noise

                        _mlp.output[:, tok_idx] = clean_out[_mlp._module_path][:,tok_idx]

                        restored = model.lm_head.output.softmax(-1)[:, -1, target_true]

                        ie = clean_logits.value - restored

                        ie.save()

                    tracing_results[i, tok_idx] = ie.value

            break

    return tracing_results
     
# %%
tracing_results = main()
# %%

tracing_results


# %%

import matplotlib.pyplot as plt
import numpy as np


data = np.array(tracing_results.numpy()).T

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(data, cmap="Purples_r", aspect="auto")
fig.colorbar(cax, ax=ax, orientation="vertical")

# ax.set_yticklabels(str_tokens)
ax.set_xlabel("single restored layer within GPT-2-XL")

# plt.savefig(save_path)
plt.show()

# %%
