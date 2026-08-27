import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()

class TextDataset(Dataset):

    def __init__(self, raw_dataset, tokenizer, seq_len=1024):

        self.seq_len = seq_len
        self.samples = []

        all_tokens = []
        for idx, item in enumerate(raw_dataset):
            if idx % 200_000 == 0:
                logger.info(f"Loading sample number {idx}")
            text = item["text"]
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
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
