import torch
import einops
from nnsight import LanguageModel

from .wrapper import FnEnvoy


def set_attn_patterns(model):

    def fn(base):
        return base.input[0][0] 
    
    def inverse(base, output):
        base.input[0][0][:] = output

    layers = model.transformer.h

    for layer in layers:
        fn_envoy = FnEnvoy(layer.attn.attn_dropout, fn, inverse)
        layer.attn.patterns = fn_envoy


def set_attn_result(model):

    def fn(base):
        z = einops.rearrange(
            base.input[0][0],
            "batch seq (head_idx d_head) -> batch seq head_idx d_head",
            head_idx = 25,
            d_head = 64
        )

        W_O = einops.rearrange(
            base.weight,
            "(head_idx d_head) d_model -> head_idx d_head d_model",
            head_idx=25,
            d_head=64
        )

        attn_result = einops.einsum(
            z, W_O,
            "batch seq head_idx d_head, head_idx d_head d_model -> batch seq head_idx d_model"
        )
        
        return attn_result

    def inverse(base, output):
        base = torch.sum(output, dim = 2) + base.bias

    layers = model.transformer.h

    for layer in layers:
        fn_envoy = FnEnvoy(layer.attn.c_proj, fn, inverse)
        layer.attn.result = fn_envoy


def gpt2_xl():

    model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)

    set_attn_result(model)
    set_attn_patterns(model)

    return model

