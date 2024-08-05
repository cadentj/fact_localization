import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
from nnsight import LanguageModel
from transformer_lens.utils import tokenize_and_concatenate
from baukit.runningstats import SecondMoment, CombinedStat, tally

from configs import StatConfig

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_cov(
    model: LanguageModel, 
    dataset: Dataset, 
    cfg: StatConfig,
):
    stat = CombinedStat(mom2 = SecondMoment())
    
    model_name = model.config._name_or_path.replace("/", "_")
    filename = cfg.filename.format(
        model_name=model_name,
        layer=cfg.layer,
    )
    filename = f"{CURRENT_DIR}/stats/{filename}"    

    loader = tally(
        stat,
        dataset,
        cache=filename,
        sample_size=cfg.sample_size,
        batch_size=cfg.batch_size,
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )

    total = cfg.sample_size // cfg.batch_size

    with torch.no_grad():

        for batch in tqdm(loader, total=total):
        
            with model.trace(batch['input_ids']):

                acts = model.transformer.h[cfg.layer].mlp.c_proj.input[0][0]

                # Flatten the batch dimension
                acts = acts.view(-1, acts.size(-1))
                acts.save()

                model.transformer.h[cfg.layer].mlp.output.stop()

            stat.add(acts.value)

    return stat


def main(model, tok, cfg):

    dataset = load_dataset(cfg.dataset, split=cfg.dataset_split)
    dataset = tokenize_and_concatenate(dataset, tok, max_length=cfg.ctx_len)
    dataset = dataset.rename_column("tokens", "input_ids")

    compute_cov(model, dataset, cfg)

if __name__ == "__main__":

    model = LanguageModel(
        "openai-community/gpt2-xl", 
        device_map="auto", 
        dispatch=True
    )

    tok = model.tokenizer

    cfg = StatConfig()
    
    main(model, tok, cfg)