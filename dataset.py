import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()


# TODO: add eos
def tokenize_dataset(raw_dataset, tokenizer, save_path="wikitext.npy"):

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
