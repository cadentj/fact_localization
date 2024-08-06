# %%
import torch
from nnsight import LanguageModel

from localization.rome.rome import execute_rome
from localization.rome.configs import RomeRequest, RomeConfig

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer
tokenizer.padding_side = "right"

# %%

req = RomeRequest(
    case_id="0",
    layer=17,
    subject="Steve Jobs",
    prompt="{} was the founder of",
    # Should probably make this whitespace agnostic
    target_new="Microsoft",
    target_true="Apple"
)

cfg = RomeConfig()

update = execute_rome(
    model, 
    tokenizer,
    req,
    cfg,
    verbose=True
)

# %%

from localization.rome.utils import format_template
import torch

input_tok, idxs = format_template(
    tokenizer,
    context_templates=[
        "The founder of {} is",
        "The creator of the company known as Apple {} is",
        "The inventor of {} is wholly responsible for the destruction of millions of young lives.",
    ],
    words=[req.subject] * 3
)

tokenizer.decode(input_tok['input_ids'][torch.arange(3), idxs])

# %%

prompt = req.prompt.format(req.subject)

with torch.no_grad():
    model.transformer.h[req.layer].mlp.c_proj.weight += update

    with model.trace(prompt):

        output = model.output.logits[:,-1,:].softmax(-1).argmax(-1).save()

print(
    prompt + tokenizer.decode(output[0])
)
