import logging
import torch
import torch.nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import Model
from data import TextDataset

logger = logging.getLogger(__name__)

def train():

    assert torch.cuda.is_available(), "CUDA GPU required for training."
    device = "cuda"

    context_len = 1024
    batch_size = 8
    grad_accum_steps = 4
    epochs = 1

    logging_rate = 200

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    dataset = TextDataset(raw_dataset, tokenizer, seq_len=context_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    model = Model(
        vocab_size=tokenizer.vocab_size,
        max_context=context_len,
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

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps
            accum_loss += loss.item()
            loss.backward()

            if step % grad_accum_steps == grad_accum_steps - 1:
                # TODO: gradient clipping
                # TODO: variable learning
                optimizer.step()
            
            if step % (logging_rate * grad_accum_steps) == grad_accum_steps - 1:
                logger.info(f"Step {step} | Loss {accum_loss} ")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename="train.log",
        filemode="a"
    )
    train()
