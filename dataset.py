import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):

    def __init__(self, raw_dataset, tokenizer, seq_len=1024):

        self.seq_len = seq_len
        self.samples = []

        all_tokens = []
        for item in raw_dataset:
            text = item["text"]
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

            chunk_size = seq_len + 1
            while len(all_tokens) >= chunk_size:
                self.samples.append(all_tokens[:chunk_size])
                all_tokens = all_tokens[seq_len:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.samples[idx], dtype=torch.long)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
