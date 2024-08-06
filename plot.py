# %%

import json
from argparse import ArgumentParser

from tqdm import tqdm
import pandas as pd
import torch
from nnsight import LanguageModel
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer
tokenizer.padding_side = "right"


dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:10%]")
dataset = tokenize_and_concatenate(dataset, tokenizer, max_length=16)

# %%


data = {}

for layer in [6,17,28,38]:

    with open(f"kl_{layer}.json") as f:
        stats = json.load(f)

    top_six = torch.topk(torch.tensor(stats), 100)

    decoded = tokenizer.batch_decode(dataset[top_six.indices]['tokens'])

    decoded = [f"[{s:.2f}] {d} " for d, s in zip(decoded, top_six.values)]

    data[layer] = decoded

df = pd.DataFrame(data)

# %%

df.to_csv("top_100.csv")

# %%

df