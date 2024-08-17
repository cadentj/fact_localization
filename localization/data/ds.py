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
    ):
        assert len(prompts) == len(subjects) == len(targets)

        self.prompts = prompts
        self.subjects = subjects
        self.targets = targets

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int):
        
        return {    
            "prompt" : self.prompts[idx],
            "subject": self.subjects[idx],
            "target": self.targets[idx]
        }
    
def collate_fn(batch, tok):
    
    prompts, idxs = format_template(
        tok,
        [s['prompt'] for s in batch],
        [s['subject'] for s in batch],
        subtoken="last"
    )

    target = [
        tok.encode(" " + s['target'])[0]
        for s in batch
    ]

    idx = torch.tensor(idxs, dtype=torch.long)

    return {
        **prompts,
        "target": target,
        "idx": idx
    }