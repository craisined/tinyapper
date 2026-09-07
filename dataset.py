import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()


def tokenize_text_dataset(raw_dataset, tokenizer, save_path="datasets/wikitext.npy"):

    all_tokens = []
    for idx, item in enumerate(raw_dataset):
        if idx % 200_000 == 0:
            logger.info(f"Tokenizing sample number {idx}")
        text = item["text"]
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        all_tokens.extend(tokens)

    np.save(save_path, np.array(all_tokens, dtype=np.int32))


def tokenize_response_dataset(
    raw_dataset, tokenizer, seq_len=1024, save_path="datasets/ultrachat.npy"
):

    examples = []
    msg_text = ""
    for idx, item in enumerate(raw_dataset):
        if idx % 10_000 == 0:
            print(f"Tokenizing sample number {idx}")
        index = 0
        x, y = [tokenizer.pad_token_id] * (seq_len + 1), [tokenizer.pad_token_id] * (seq_len + 1)
        for message in item["messages"]:
            msg_text = "\n<|user|>:\n" if message["role"] == "user" else ""
            msg_text += message["content"]
            msg_text += (
                "\n<|assistant|>:\n" if message["role"] == "user" else tokenizer.eos_token
            )
            encoded = tokenizer.encode(msg_text, truncation=False)
            if index + len(encoded) > seq_len + 1:
                break
            x[index:index + len(encoded)] = encoded
            if message["role"] == "assistant":
                y[index:index + len(encoded)] = encoded
            index += len(encoded)
        if len(set(y)) != 1:
            examples.append([x, y])

    np.save(save_path, np.array(examples))


class TextDataset(Dataset):

    def __init__(self, array_path, seq_len=1024):
        self.samples = np.load(array_path)
        cutoff = -(len(self.samples) % (seq_len + 1))
        if cutoff != 0:
            self.samples = self.samples[:cutoff]
        self.samples = self.samples.reshape(-1, seq_len + 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.samples[idx]).to(torch.long)
        return chunk[:-1], chunk[1:]


class ResponseDataset(Dataset):

    def __init__(self, array_path, seq_len=1024):
        self.samples = np.load(array_path)
        assert seq_len + 1 == self.samples.shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.samples[idx]).to(torch.long)
        return chunk[0][:-1], chunk[1][1:]
