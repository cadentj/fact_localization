# %%

from nnsight import LanguageModel
from transformer_lens.utils import tokenize_and_concatenate
from baukit.runningstats import Variance
from tqdm import tqdm
import torch
from datasets import load_dataset

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer
tokenizer.padding_side = "right"


dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:1%]")
dataset = tokenize_and_concatenate(dataset, tokenizer, max_length=64)

stat = Variance()

for p in tqdm(dataset[:1000]['tokens'], total=1000):

    with torch.no_grad():
        with model.trace(p):
            embed = model.transformer.wte.output
            embed = embed.view(-1, embed.size(-1)).save()

    stat.add(embed.value)

stdev = stat.stdev()

noise = torch.randn_like(stdev) * (3 * stdev)

# %%

torch.save(noise.cpu(), "noise.pt")
