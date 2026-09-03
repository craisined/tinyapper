import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class Model(nn.Module):

    def __init__(self, vocab_size=32768, embed_dim=512, max_context=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(
            max_context, embed_dim
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Tie weights
        self.transformer = Transformer(embed_dim=embed_dim)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.size()
        input_tokens = self.token_embedding(x)
        pos = self.pos_embedding(torch.arange(T, device=x.device))
        x = input_tokens + pos
        x = self.transformer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class Transformer(nn.Module):

    def __init__(self, embed_dim, num_heads=8, layers=12):
        super().__init__()
        layers = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(layers)
        ]
        self.model = nn.Sequential(*layers)  # TODO: switch off sequential

    def forward(self, x):
        return self.model(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(embed_dim=embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.ffn(self.ln2(x))
        return x


class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, max_context=1024):
        super().__init__()

        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.head_dim = embed_dim // num_heads
        self.max_context = max_context

    def forward(self, x, cache=None):

        B, T, C = x.size()

        # Equivalent to running x through 3 linear layers
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        prior_tokens = 0 if cache is None else cache.total_tokens
        if prior_tokens + T > self.max_context:
            raise ValueError(
                f"{prior_tokens + T} tokens exceed the current conext window"
            )

        if cache is not None:
            k, v = cache.push(k, v)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(T > 1)
        )  # TODO: custom mask for multiple fills

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class FeedForwardNetwork(nn.Module):

    def __init__(self, embed_dim, expansion_factor=4):
        super().__init__()

        hidden_dim = embed_dim * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class KVCache(nn.Module):

    def __init__(self, batches, max_context, num_heads, head_dim):
        super().__init__()
        shape = (batches, num_heads, max_context, head_dim)
        self.max_context = max_context
        self.total_tokens = 0
        self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    def push(self, k, v):

        k = k.to(self.k_cache.dtype)
        v = v.to(self.v_cache.dtype)
        tokens = k.shape[2]

        end_idx = self.total_tokens + tokens
        self.k_cache[:, :, self.total_tokens : end_idx] = k
        self.v_cache[:, :, self.total_tokens : end_idx] = v
        self.total_tokens += tokens
        return (
            self.k_cache[:, :, : self.total_tokens],
            self.v_cache[:, :, : self.total_tokens],
        )

    def reset(self):
        self.total_tokens = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
