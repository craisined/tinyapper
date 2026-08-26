import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, vocab_size=32768, embed_dim=512, max_context=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(
            max_context, embed_dim
        )  # TODO: better positional markers (RoPE)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Tie weights
        self.transformer = Transformer(embed_dim=embed_dim)
        self.ln_f = nn.LayerNorm(embed_dim)

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
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(embed_dim=embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()

        # Equivalent to running x through 3 linear layers
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = C // self.num_heads
        q = q.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
