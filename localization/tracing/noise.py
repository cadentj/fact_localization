# %%

import json

from nnsight import LanguageModel
from tqdm import tqdm
import torch

def filter_reqs(cf, results):
    ids = set([int(req) for req in results])
    return [
        req for req in cf if int(req['case_id']) in ids
    ]

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer
tokenizer.padding_side = "right"

results_path = "../results/6.json"
cf_path = "../localization/data/counterfact/counterfact.json"

with open(results_path, "r") as f:
    results = json.load(f)

with open(cf_path, "r") as f:
    cf = json.load(f)

reqs = filter_reqs(cf, results)

embeddings = []

for r in tqdm(reqs, total=len(reqs)):

    prompt = r['requested_rewrite']['subject']

    with torch.no_grad():
        with model.trace(prompt):
            embed = model.transformer.wte.output
            embed = embed[0].cpu().save()

    embeddings.append(embed)

stdev = torch.cat(embeddings).std().item()
stdev