# %%

import os
import json

from nnsight import LanguageModel

from stuff import CausalTracingInput, causal_trace, plot_trace

def load_fact(idx, dataset, tokenizer):

    point = dataset[idx]['requested_rewrite']

    # raw_prompt = point['prompt']
    # subject = point['subject']
    # target = point['target_true']['str']

    raw_prompt = "The {} is in downtown"
    subject = "Space Needle"
    target = "Seattle"

    prompt = raw_prompt.replace("{}", subject)

    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")[0]
    subject_tokens = tokenizer.encode(" " + subject, return_tensors="pt")[0]
    target_token = tokenizer.encode(" " + target, return_tensors="pt")[0]

    return prompt_tokens, subject_tokens, target_token

def main():

    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
    tokenizer = model.tokenizer

    with open("../localization/data/counterfact/counterfact.json") as f:
        counterfact = json.load(f)

    prompt, subject, target = load_fact(0, counterfact, tokenizer)

    cfg = CausalTracingInput(prompt=prompt, subject=subject, obj=target)

    results = causal_trace(model, cfg)

    # plot_trace(results, tokenizer.batch_decode(prompt), save_path="plot.png")

    return results

# if __name__ == "__main__":
#     main()

results = main()

plot_trace(results.numpy(), [" ", " "], save_path="plot.png")