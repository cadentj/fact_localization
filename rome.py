# %%


from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2-xl", device_map="auto", dispatch=True)
tokenizer = model.tokenizer

# %%

from rome.rome import execute_rome

execute_rome(
    model, 
    tokenizer
)