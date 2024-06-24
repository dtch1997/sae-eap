# type: ignore
"""Handler for the Greater-Than dataset, Hanna (2023)"""

from __future__ import annotations

import torch
import pandas as pd
import torch.nn.functional as F

from functools import partial
from contextlib import contextmanager
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader

from sae_eap.data.handler import BatchHandler
from sae_eap.core.types import Model


def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(
        inputs, padding="longest", return_tensors="pt", add_special_tokens=True
    )
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths


class GreaterThanHandler(BatchHandler):
    def __init__(self, model: Model):
        self.model = model
        self.prob_diff_fn = partial(get_prob_diff(self.tokenizer), loss=True, mean=True)

        # Initialize dummy batch
        self.clean = "hello"
        self.corrupted = "hello"
        self.label = None

    def set_batch(self, clean: str, corrupted: str, label: int):
        self.clean = clean
        self.corrupted = corrupted
        self.label = label
        n_pos, input_lengths = get_npos_input_lengths(self.model, self.clean)
        self.n_pos = n_pos
        self.input_lengths = input_lengths

    def get_batch(self):
        return self.clean, self.corrupted, self.label

    @contextmanager
    def handle(self, clean: str, corrupted: str, label: int):
        old_batch = self.get_batch()
        self.set_batch(clean, corrupted, label)
        yield
        self.set_batch(*old_batch)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def get_logits(self, model, input="clean"):
        if input == "clean":
            return model(self.clean)
        else:
            return model(self.corrupted)

    def get_metric(self, logits):
        return self.prob_diff_fn(logits, self.input_lengths, self.label)

    def get_batch_size(self):
        return len(self.clean)

    def get_n_pos(self):
        return self.n_pos


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels


class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row["clean"], row["corrupted"], row["label"]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits


def get_prob_diff(tokenizer: PreTrainedTokenizer):
    year_indices = torch.tensor(
        [tokenizer(f"{year:02d}").input_ids[0] for year in range(100)]
    )

    def prob_diff(
        logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,
        mean=True,
        loss=False,
    ):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)[:, year_indices]

        results = []
        for prob, year in zip(probs, labels):
            results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())

        results = torch.stack(results)
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    return prob_diff


def kl_div(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean=True,
    loss=True,
):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = F.kl_div(
        probs.log(), clean_probs.log(), log_target=True, reduction="none"
    ).mean(-1)
    return results.mean() if mean else results
