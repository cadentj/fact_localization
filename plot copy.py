# %%

import json
from argparse import ArgumentParser

from tqdm import tqdm
import pandas as pd
import torch
from nnsight import LanguageModel
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate


model = LanguageModel("google/gemma-2b", dispatch=True, device_map='auto')

model