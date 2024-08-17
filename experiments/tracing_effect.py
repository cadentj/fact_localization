# %%

import json

import torch
from tqdm import tqdm
from nnsight import LanguageModel

from localization.tracing.tracing import CausalTracingInput, causal_trace
from localization.rome.utils import format_template 


results_path = "./results/6.json"
cf_path = "./localization/data/counterfact/counterfact.json"


def prepend_bos(input, bos_token):
    batch_size = input.size(0)
    bos_tensor = torch.full((batch_size, 1), bos_token, dtype=input.dtype)
    output = torch.cat((bos_tensor, input), dim=1)
    return output


def load_fact(tokenizer, req):
    raw_prompt = req["prompt"]
    subject = req["subject"]
    target = req["target_true"]["str"]

    print(f"RAW: |{raw_prompt}|")
    print(f"SUBJECT: |{subject}|")
    print(f"TARGET: |{target}|")

    input_tok, subject_idxs = format_template(
        tokenizer, [raw_prompt], [subject], subtoken="all"
    )

    prompt = prepend_bos(input_tok["input_ids"], tokenizer.bos_token_id)[0]
    target_token = tokenizer.encode(" " + target, return_tensors="pt")[0].item()

    return CausalTracingInput(
        prompt=prompt,
        subject_idxs=[i + 1 for i in subject_idxs[0]], # Adj for prepended BOS
        target_id=target_token,
    )


def filter_reqs(cf, results):
    ids = set([int(req) for req in results])
    return [req for req in cf if int(req["case_id"]) in ids]


def main():

    with open(results_path, "r") as f:
        results = json.load(f)

    with open(cf_path, "r") as f:
        cf = json.load(f)
        
    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"

    tracing_results = []
    filtered = filter_reqs(cf, results)

    for req in tqdm(filtered):

        cfg = load_fact(tokenizer, req['requested_rewrite'])
        results = causal_trace(model, cfg, n_iters=5)

        max_values = torch.max(results / 5, dim=1)
        vals = max_values.values
        idxs = max_values.indices
    
        tracing_results.append({
            "case_id": req['case_id'],
            "vals": vals.tolist(),
            "idxs": idxs.tolist(),
            "subject_idxs": cfg.subject_idxs
        })

    with open("tracing_effect.json", "w") as f:
        json.dump(tracing_results, f, indent=2)
    
if __name__ == "__main__":
    main()  