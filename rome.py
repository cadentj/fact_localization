# %%
import torch
from nnsight import LanguageModel

from rome.rome import execute_rome
from rome.configs import RomeRequest, RomeConfig

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer

# %%

req = RomeRequest(
    layer=17,
    subject="Steve Jobs",
    prompt="{} was the founder of",
    # Should probably make this whitespace agnostic
    target_new=" Microsoft"
)

cfg = RomeConfig()

update = execute_rome(
    model, 
    tokenizer,
    req,
    cfg
)

# %%

prompt = req.prompt.format(req.subject)

with torch.no_grad():
    model.transformer.h[req.layer].mlp.c_proj.weight += update

    with model.trace(prompt):

        output = model.output.logits[:,-1,:].softmax(-1).argmax(-1).save()

print(
    prompt + tokenizer.decode(output[0])
)
