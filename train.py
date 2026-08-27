import logging
import torch
import torch.nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from types import SimpleNamespace

from model import Model
from data import TextDataset

logger = logging.getLogger(__name__)


def train(config=None, **kwargs):

    assert torch.cuda.is_available(), "CUDA GPU required for training."
    device = "cuda"

    config_defaults = {
        "context": 1024,
        "epochs": 1,
        "grad_steps": 4,
        "microbatch_size": 8,
        "logging_rate": 20,
    }
    if config is None:
        config = kwargs
    config = SimpleNamespace(**(config_defaults | config))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    dataset = TextDataset(raw_dataset, tokenizer, seq_len=config.context)
    dataloader = DataLoader(
        dataset,
        batch_size=config.microbatch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = Model(
        vocab_size=tokenizer.vocab_size,
        max_context=config.context,
    ).to(device)
    model = torch.compile(model)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / config.grad_steps
            accum_loss += loss.item()
            loss.backward()

            if step % config.grad_steps == config.grad_steps - 1:
                # TODO: gradient clipping
                # TODO: variable learning
                optimizer.step()

            if (
                step % (config.logging_rate * config.grad_steps)
                == config.grad_steps - 1
            ):
                logger.info(f"Step {step} | Loss {accum_loss} ")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="train.log", filemode="a")
    train()
