import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class Model(nn.Module):

    def __init__(self, vocab_size=32768, embed_dim=512, max_context=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
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
        x = self.token_embedding(x)
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

        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )  # TODO: replace with torch
        self.ffn = FeedForwardNetwork(embed_dim=embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, max_context=1024):
        super().__init__()

        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.head_dim = embed_dim // num_heads
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim, max_seq_len=max_context, base=10000
        )

    def forward(self, x, start_pos=0, cache=None):

        B, T, C = x.size()

        # Equivalent to running x through 3 linear layers
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        input_pos = torch.arange(start_pos, start_pos + T, device=x.device)

        q = self.rope(q, input_pos=input_pos)
        k = self.rope(k, input_pos=input_pos)

        if cache is not None:
            k, v = cache.push(k, v)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=bool(cache))

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)

    def reset(self):
        self.kv_cache.reset()


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
        self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    def push(self, k, v, start_pos):

        k = k.to(self.k_cache.dtype)
        v = v.to(self.v_cache.dtype)
        tokens = k.shape[2]

        input_pos = torch.arange(start_pos, start_pos + tokens, device=k.device)
        ptr = input_pos % self.max_context
        self.k_cache[:, :, ptr] = k
        self.v_cache[:, :, ptr] = v
        return self.unroll(start_pos + tokens)

    def unroll(self, total_tokens):
        k_unrolled = self.k_cache
        v_unrolled = self.v_cache
        if total_tokens > self.max_context:
            shift = -(total_tokens % self.max_context)
            k_unrolled = torch.roll(self.k_cache, shifts=shift, dims=2)
            v_unrolled = torch.roll(self.v_cache, shifts=shift, dims=2)
        cache_len = min(total_tokens, self.max_context)
        return k_unrolled[:, :, :cache_len], v_unrolled[:, :, :cache_len]

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
