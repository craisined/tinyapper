from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from types import SimpleNamespace

from model import Model

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.truncation_side = "left"


def load_model(checkpoint_path):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = SimpleNamespace(**checkpoint["config"])
    model = Model(vocab_size=tokenizer.vocab_size, max_context=config.context).to(
        device
    )
    unwrapped_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state"].items()
    }
    model.load_state_dict(unwrapped_state_dict)
    model.eval()
    return model, config


# TODO: migrate from torch.ao.quantization
def load_quantized_model(checkpoint_path):

    model, config = load_model(checkpoint_path)
    model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return model, config


@torch.no_grad()
def run_model(input_text, loaded_model, config=None, **kwargs):

    config_defaults = {
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_k": 40,
    }
    if config is None:
        config = kwargs
    config = SimpleNamespace(**(config_defaults | config))

    model, model_config = loaded_model
    cache_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    caches = model.create_kv_caches(batches=1, device=device, dtype=cache_dtype)

    input_tokens = tokenizer.encode(
        input_text, truncation=True, return_tensors="pt"
    ).to(device)
    output_tokens = []

    for _ in range(config.max_tokens):

        is_cuda = device == "cuda"
        with torch.amp.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=is_cuda
        ):
            logits = model(input_tokens, caches=caches)[:, -1, :]
        logits = logits / max(config.temperature, 1e-5)

        v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        output_tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_tokens = next_token.view(1, 1)

    return input_text + tokenizer.decode(output_tokens)

@torch.no_grad()
def stream_model(input_text, loaded_model, config=None, **kwargs):

    config_defaults = {
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_k": 40,
    }
    if config is None:
        config = kwargs
    config = SimpleNamespace(**(config_defaults | config))
    is_cuda = device == "cuda"

    model, model_config = loaded_model
    cache_dtype = torch.bfloat16 if is_cuda else torch.float32
    caches = model.create_kv_caches(batches=1, device=device, dtype=cache_dtype)

    input_tokens = tokenizer.encode(
        input_text, truncation=True, return_tensors="pt"
    ).to(device)
    output_tokens = []

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=is_cuda):
        logits = model(input_tokens, caches=caches)[:, -1, :]
    logits = logits / max(config.temperature, 1e-5)

    v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float("Inf")
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    yield tokenizer.decode([next_token.item()])
    if next_token.item() == tokenizer.eos_token_id:
        return

    input_tokens = next_token.view(1, 1)

    for _ in range(config.max_tokens):

        with torch.amp.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=is_cuda
        ):
            logits = model(input_tokens, caches=caches)[:, -1, :]
        logits = logits / max(config.temperature, 1e-5)

        v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        output_tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

        yield tokenizer.decode([next_token.item()])
        input_tokens = next_token.view(1, 1)


if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints")
    checkpoint_file = "0.pt"
    model = load_model(checkpoint_dir / checkpoint_file)
    # model = load_quantized_model(checkpoint_dir / checkpoint_file)
    prompt = input("Enter prompt: ")
    if prompt:
        print(run_model(prompt, model, max_tokens=100))
        for output in stream_model(prompt, model, max_tokens=100):
            print(output, end="", flush=True)
