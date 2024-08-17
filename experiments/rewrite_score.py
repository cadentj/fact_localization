
# %%
import json 
from argparse import ArgumentParser
import logging

from tqdm import tqdm
import torch
from nnsight import LanguageModel

from localization.rome.rome import execute_rome
from localization.rome.configs import RomeRequest, RomeConfig

def filter_requests(model, tokenizer, reqs, batch_size: int = 256):

    prompts = [req.prompt.format(req.subject) for req in reqs]
    targets_new = [tokenizer.encode(req.target_new)[0] for req in reqs]
    targets_true = [tokenizer.encode(req.target_true)[0] for req in reqs]

    batches = [
        {
            "prompt": prompts[i:i+batch_size],
            "target_new": targets_new[i:i+batch_size],
            "target_true": targets_true[i:i+batch_size],
            "reqs": reqs[i:i+batch_size]
        } for i in range(0, len(reqs), batch_size)
    ]

    results = []

    for batch in tqdm(batches):

        idxs = torch.arange(len(batch['prompt']))
        input = tokenizer(batch['prompt'], return_tensors="pt", padding=True)
        lookup_idxs = [
            sum(input['attention_mask'][i]) - 1 for i in idxs
        ]

        with torch.no_grad():

            with model.trace(input['input_ids']):
                logits = model.output.logits[idxs,lookup_idxs,:].softmax(-1)
                p_new = logits[idxs, batch['target_new']].save()
                p_high = logits.argmax(-1).save()

            for i, (true, predicted) in enumerate(zip(batch['target_true'], p_high)):
                if true == predicted:
                    results.append((batch['reqs'][i], p_new[i].item()))

    return results

def main(model, tokenizer, layer, n_reqs, logger):
    def build_requests(n_req):
        with open("../localization/data/counterfact/counterfact.json", "r") as f:
            data = json.load(f)[:n_req]

        reqs = []
        for d in data:
            req = d['requested_rewrite']
            reqs.append(
                RomeRequest(
                    layer=layer,
                    case_id=d['case_id'],
                    subject=req['subject'],
                    prompt=req['prompt'],
                    target_true=" " + req['target_true']['str'],
                    target_new=" " + req['target_new']['str'],
                )
            )

        return reqs

    reqs = build_requests(n_reqs)
    filtered = filter_requests(model, tokenizer, reqs)
    print(len(filtered))

    cfg = RomeConfig()

    deltas = {}

    for req, p_init in tqdm(filtered, desc="Executing ROME on filtered prompts..."):

        update = execute_rome(
            model, 
            tokenizer,
            req,
            cfg,
            verbose=True
        ).detach()

        prompt = req.prompt.format(req.subject)
        target = tokenizer.encode(req.target_new)[0]

        with torch.no_grad():

            model.transformer.h[req.layer].mlp.c_proj.weight += update

            with model.trace(prompt):
                p_new = model.output.logits \
                    .softmax(-1)[0, -1, target].item().save()
            
            deltas[req.case_id] = {
                "init": p_init,
                "new" : p_new.value,
            }

            model.transformer.h[req.layer].mlp.c_proj.weight -= update

            rewrite_score = (p_new - p_init) / (1 - p_init)
            logger.info(f"Case ID: {req.case_id} | Prompt: {req.prompt.format(req.subject)} | Score: {rewrite_score}")
    
    with open(f"{layer}.json", "w") as f:
        json.dump(deltas, f, indent=4)
    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--layer", type=int)
    parser.add_argument("--n_reqs", type=int)
    args = parser.parse_args()

    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logging.basicConfig(filename=f"rewrite_score_{args.layer}.log", filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    main(model, tokenizer, 0, args.n_reqs, logger)