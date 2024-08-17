# %%
import json
from argparse import ArgumentParser
from collections import defaultdict 
from tqdm import tqdm
import torch
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nnsight import LanguageModel

from localization.data.ds import SubjectDataset, collate_fn

def main(model, tok, layers):
    path = "../localization/data/counterfact/counterfact.json"

    with open(path, "r") as f:
        data = json.load(f)

    prompts = []
    subjects = []
    targets = []

    for d in tqdm(data):
        req = d["requested_rewrite"]
        prompts.append(req["prompt"])
        subjects.append(req["subject"])
        targets.append(req["target_true"]["str"])

    loader = DataLoader(
        SubjectDataset(prompts, subjects, targets),
        batch_size=10,
        shuffle=True,
        collate_fn=partial(collate_fn, tok=tok),
    )

    stats = defaultdict(list)

    with torch.no_grad():

        for batch in tqdm(loader):
            idxs = batch["idx"]

            with model.trace() as tracer:

                with tracer.invoke(batch['input_ids']):
                    clean_logits = model.lm_head \
                        .output[:, idxs].softmax(-1).save()

                corr_logits = []

                for l in layers:

                    with tracer.invoke(batch['input_ids']):

                        # for _l in range(l, len(model.transformer.h)):
                            # model.transformer.h[_l].mlp.output *= 0

                        model.transformer.h[l].mlp.output *= 0

                        corr_logits.append(
                            model.lm_head.output[:, idxs] \
                                .log_softmax(-1).save()
                        )

            for l, corr in zip(layers, corr_logits):
                kl = F.kl_div(corr, clean_logits, reduction="none")

                per_fact = kl.sum(dim=-1).mean(dim=1)

                stats[l].extend(per_fact.tolist())

    # with open(f"kl_{layer}.json", "w") as f:
    #     json.dump(stats, f)

if __name__ == "__main__":

    model = LanguageModel("openai-community/gpt2-xl", device_map="cuda:0", dispatch=True)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"

    n_layers = len(model.transformer.h)
    main(model, tokenizer, list(range(n_layers - 5, n_layers)))


