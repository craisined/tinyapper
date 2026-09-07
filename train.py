from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from types import SimpleNamespace

from model import Model
from dataset import TextDataset

logger = logging.getLogger(__name__)


def train(config=None, **kwargs):

    assert torch.cuda.is_available(), "CUDA GPU required for training."
    device = "cuda"

    config_defaults = {
        "checkpoint_dir": "checkpoints",
        "context": 1024,
        "dataset_path": "wikitext.npy",
        "epochs": 1,
        "grad_norm": 1,
        "grad_steps": 8,
        "microbatch_size": 4,
        "logging_rate": 20,
        "lr": 6e-4,
        "vocab_size": 50257,  # Default GPT 2 tokenization
        "warmup_batches": 200,
        "weight_decay": 0.1,
    }
    if config is None:
        config = kwargs
    config = SimpleNamespace(**(config_defaults | config))

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    dataset = TextDataset(config.dataset_path, seq_len=config.context)
    dataloader = DataLoader(
        dataset,
        batch_size=config.microbatch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    logger.info("Loaded dataset!")

    raw_model = Model(
        vocab_size=config.vocab_size,
        max_context=config.context,
    ).to(device)
    model = torch.compile(raw_model)
    logger.info("Loaded model!")

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.lr, betas=(0.9, 0.95), eps=1e-8
    )
    criterion = nn.CrossEntropyLoss()

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_batches,
        num_training_steps=config.epochs * (len(dataloader) // config.grad_steps),
    )

    model.train()

    starting_time = time()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.tensor(0.0, device=device)
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / config.grad_steps
            accum_loss += loss.detach()
            loss.backward()

            if step % config.grad_steps == config.grad_steps - 1:
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if (
                step % (config.logging_rate * config.grad_steps)
                == config.logging_rate * config.grad_steps - 1
            ):
                logger.info(
                    f"Step: {step + 1} | Loss per token: {accum_loss.item() / config.logging_rate} | Avg time / step: {((time() - starting_time) / (step + 1)):2f}"
                )
                accum_loss.zero_()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": raw_model.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(config),
        }
        torch.save(checkpoint, Path(config.checkpoint_dir) / f"{epoch}.pt")
        logger.info(f"Saved checkpoint for epoch {epoch + 1}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="train.log", filemode="w")
    train()
