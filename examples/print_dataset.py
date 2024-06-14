# %%
import sys

sys.path.append("/root/circuit-finder")
import pandas as pd
import transformer_lens as tl
from circuit_finder.constants import ProjectDir
from circuit_finder.data_loader import load_datasets_from_json

dataset_path = ProjectDir / "datasets/greaterthan_gpt2-small_prompts.json"
model = tl.HookedTransformer.from_pretrained(
    "gpt2",
    device="cuda",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
)

train_loader, _ = load_datasets_from_json(
    model=model,
    path=dataset_path,
    device="cuda",
)

batch = next(iter(train_loader))
clean_tokens = batch.clean
answer_tokens = batch.answers
wrong_answer_tokens = batch.wrong_answers
corrupt_tokens = batch.corrupt

# %%
print(clean_tokens.shape)
print(answer_tokens.shape)
print(wrong_answer_tokens.shape)
print(corrupt_tokens.shape)

# %%


pd.set_option("display.max_colwidth", None)

df = pd.DataFrame(
    {
        "clean": model.to_string(clean_tokens),
        "answer": model.to_string(answer_tokens),
        "wrong_answer": model.to_string(wrong_answer_tokens),
        "corrupt": model.to_string(corrupt_tokens),
    }
)
df.head()

# print(model.to_string(clean_tokens))
# print(model.to_string(answer_tokens))
# print(model.to_string(wrong_answer_tokens))
# print(model.to_string(corrupt_tokens))

# %%
