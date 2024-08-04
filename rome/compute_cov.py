import torch
from tqdm import tqdm
from datasets import load_dataset
from nnsight import LanguageModel
from transformer_lens.utils import tokenize_and_concatenate

from baukit.runningstats import SecondMoment, tally

def flatten(x):
    return x.view(-1, x.size(-1))

def compute_cov(model, dataset, layer: int, filename: str = "mom2.npz"):

    stat = SecondMoment()

    loader = tally(
        stat,
        dataset,
        cache=filename,
        sample_size=100_000,
        batch_size=100,
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )

    n = len(dataset) / 100

    with torch.no_grad():

        for batch in tqdm(loader, total=n):
        
            with model.trace(batch['input_ids']):

                acts = flatten(model.transformer.h[layer].mlp.act.output)

                acts.save()

                model.transformer.h[layer].mlp.output.stop()

            stat.add(acts.value)

    return stat

def main():

    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
    tokenizer = model.tokenizer

    dataset = load_dataset("kh4dien/fineweb-100m-sample", name="", split="train[:15%]")
    dataset = tokenize_and_concatenate(dataset, tokenizer, max_length=64)
    dataset = dataset.rename_column("tokens", "input_ids")

    compute_cov(model, dataset, 0)