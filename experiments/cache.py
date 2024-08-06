import json
from argparse import ArgumentParser

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nnsight import LanguageModel
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

from localization.rome.rome import execute_rome
from localization.rome.configs import RomeRequest, RomeConfig


def main(model, tokenizer, layer):
    cfg = RomeConfig()

    req = RomeRequest(
        layer=layer,
        case_id="225",
        subject="Visual Studio Code",
        prompt="{} was a product of",
        target_true=" " + "Microsoft",
        target_new=" " + "Airbus",
    )

    update = execute_rome(model, tokenizer, req, cfg, verbose=True).detach()

    dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:10%]")
    dataset = tokenize_and_concatenate(dataset, tokenizer, max_length=16)

    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    stats = []

    with torch.no_grad():
        for batch in tqdm(loader):
            model.transformer.h[req.layer].mlp.c_proj.weight += update

            with model.trace(batch["tokens"]):
                edited = model.lm_head.output.log_softmax(-1).cpu().save()

            model.transformer.h[req.layer].mlp.c_proj.weight -= update

            with model.trace(batch["tokens"]):
                normal = model.lm_head.output.softmax(-1).cpu().save()

            kl = F.kl_div(edited, normal, reduction="none")
            kl = kl.sum(dim=-1).mean(dim=1)
            stats.extend(kl.tolist())

            del edited, normal

    with open(f"kl_{layer}.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()

    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"

    main(model, tokenizer, args.layer)
