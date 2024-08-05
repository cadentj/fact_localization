# %%
import json
import torch
from tqdm import tqdm
from functools import partial
from localization.data import SubjectDataset, collate_fn
from nnsight import LanguageModel

from torch.utils.data import DataLoader 
from torch.optim import AdamW

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tok = model.tokenizer

# NOTE: Change this from wherever you run this notebook from. 
# Will flush into a module later.
path = "../data/counterfact/counterfact.json"

n_examples = 1_000

with open(path, "r") as f:
    data = json.load(f)[:n_examples]

prompts = []
subjects = []
targets = []

for d in data:
    req = d['requested_rewrite']
    prompts.append(req['prompt'])
    subjects.append(req['subject']) 
    targets.append(req['target_true']['str'])

# %%


# Average token embeddings at last subject position across the dataset
def subject_means():
    mean_embed = None

    wte = model.transformer.wte
    batch_idxs = torch.arange(100, dtype=torch.long)

    for batch in tqdm(loader):
        with model.trace(batch['input_ids']):
            _mean_embed = wte.output[batch_idxs, batch['idx'], :].mean(dim=0).save()

        if mean_embed is None:
            mean_embed = _mean_embed.value
        else:
            mean_embed += _mean_embed.value

    mean_embed /= len(loader)

    return mean_embed.detach().cpu()

# means = subject_means()

# loader = DataLoader(
#     SubjectDataset(prompts, subjects, targets, tok), 
#     batch_size=100, 
#     shuffle=True, 
#     collate_fn=partial(collate_fn, tok=tok)
# )

# with open("./subject_means.pth", "wb") as f:
#     torch.save(means, f)

# %%

loader = DataLoader(
    SubjectDataset(prompts, subjects, targets, tok), 
    batch_size=1, 
    shuffle=True, 
    collate_fn=partial(collate_fn, tok=tok)
)

n_layers = len(model.transformer.h)
init_token = torch.load("./subject_means.pth")
init_token = init_token.unsqueeze(0).repeat(n_layers, 1)
null_token = torch.nn.Parameter(init_token)

opt = AdamW([null_token], lr=1e-5, weight_decay=0)
wte = model.transformer.wte
submodules = [model.transformer.h[i].mlp for i in range(n_layers)]
grad_acc = 8

loss_stats= []

for it in range(10):

    for n, prompt in tqdm(enumerate(loader)):

        corr = prompt['input_ids'].repeat(n_layers, 1)
        clean = prompt['input_ids']

        with model.trace() as tracer:

            with tracer.invoke(clean):

                clean_out = {
                    mlp._module_path : mlp.output
                    for mlp in submodules
                }

                clean_logit = model.lm_head.output[:, prompt['idx']] \
                    .log_softmax(dim=-1)[:, 0, prompt['target']].save()

            with tracer.invoke(corr):

                for i in range(n_layers):
                    wte.output[i, prompt['idx'], :] = null_token[i, :]  

                for i, (idx, mlp) in enumerate(zip(prompt['idx'], submodules)):
                    mlp.output[i, idx, :] = clean_out[mlp._module_path][i, idx, :]

                corr_logits = model.lm_head.output[:, prompt['idx']] \
                    .log_softmax(dim=-1)[:, 0, prompt['target']].save()
                        
        loss = (clean_logit.repeat(n_layers,1) - corr_logits).relu().sum()
        loss.backward()

        if (n + 1) % grad_acc == 0:
            opt.step()
            opt.zero_grad()

            print(f"Iter: {n + 1} | Loss: {loss.item()}")

            loss_stats.append(loss.item())

    