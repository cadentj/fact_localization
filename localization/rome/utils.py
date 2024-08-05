from typing import List, Literal
from copy import deepcopy

from transformers import AutoTokenizer
from nnsight import LanguageModel


def sample_k(
    model: LanguageModel, 
    tokenizer: AutoTokenizer,
    n_prompts: int,
    **generation_kwargs
) -> List[str]:
    
    batch = ['<|endoftext|>'] * n_prompts
    with model.generate(batch, **generation_kwargs):

        results = model.generator.output.save()

    # Return everything after <|endoftext|>
    samples = tokenizer.batch_decode(results[:,1:])
    return [sample + ". {}" for sample in samples]


def format_template(
    tok: AutoTokenizer, 
    context_templates: List[str], 
    words: str, 
    subtoken: str = "last", 
    padding_side: Literal["right", "left"] = "left"
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "Multiple fill-ins not supported."

    assert subtoken == "last", "Only last token retrieval supported."

    # Compute prefixes and suffixes of the tokenized context
    prefixes, suffixes = _split_templates(context_templates)
    _words = deepcopy(words)

    # Compute lengths of prefixes, words, and suffixes
    prefixes_len, words_len, suffixes_len = \
        _get_split_lengths(tok, prefixes, _words, suffixes)
    
    # Format the prompts bc why not
    prompts = [
        template.format(word)
        for template, word in zip(context_templates, words)
    ]

    # Compute indices of last tokens
    if padding_side == "right":

        word_idxs = [
            prefixes_len[i] + words_len[i] - 1
            for i in range(len(prefixes))
        ]

        return prompts, word_idxs
    
    elif padding_side == "left":

        word_idxs = [
            -(suffixes_len[i] + 1)
            for i in range(len(prefixes))
        ]

        return prompts, word_idxs
    

def _get_split_lengths(tok, prefixes, words, suffixes):
    # Pre-process tokens to account for different 
    # tokenization strategies
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes])
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    return prefixes_len, words_len, suffixes_len


def _split_templates(context_templates):
    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes = [tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)]
    suffixes = [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]

    return prefixes, suffixes