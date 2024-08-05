from typing import List

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from ..rome.utils import format_template

class SubjectDataset(Dataset):

    def __init__(
        self,
        prompts: List[str],
        subjects: List[str],
        targets: List[str],
        tok: AutoTokenizer
    ):
        assert len(prompts) == len(subjects) == len(targets)
        prompts, idxs = format_template(
            tok,
            context_templates=prompts,
            words=subjects,
            padding_side="left"
        )

        self.prompts = prompts
        self.idxs = idxs
        self.targets = targets

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int):
        
        return {
            "prompt": self.prompts[idx],
            "idx": self.idxs[idx],
            "target": self.targets[idx]
        }
    
def collate_fn(batch, tok):
    
    prompt = tok(
        [
            s['prompt'] for s in batch
        ],
        return_tensors="pt",
        padding=True,
    )

    target = [
        tok.encode(" " + s['target'])[0]
        for s in batch
    ]

    idx = torch.tensor([s['idx'] for s in batch], dtype=torch.long)

    return {
        **prompt,
        "target": target,
        "idx": idx
    }