from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from types import SimpleNamespace

from model import Model

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_avaliable() else "cpu"


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = SimpleNamespace(checkpoint["config"])
    model = Model(vocab_size=config.vocab_, max_context=config.context)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
