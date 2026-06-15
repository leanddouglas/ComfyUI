"""
Native port of Roblox/cube's shape GPT (DualStreamRoformer).

Reference: https://github.com/Roblox/cube  (cube3d/model/gpt/dual_stream_roformer.py
and cube3d/model/transformers/*).

This is an autoregressive transformer over discrete VQ shape tokens, conditioned on
CLIP text embeddings. It is NOT a diffusion model; it is driven by the dedicated
`sample_cube` sampler (see comfy/k_diffusion/sampling.py), not KSampler.

The forward pass is kept faithful to upstream so token IDs match bit-for-bit:
  * rope_theta = 10000
  * per-head RMSNorm on Q and K
  * dual-stream (MM-DiT style) joint attention; last dual block is cond_pre_only
  * two separate RoPE frequency tensors (dual blocks offset cond tokens by S)
  * SwiGLU MLP, non-affine LayerNorm upcast to fp32
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norms (faithful to cube3d/model/transformers/norm.py)
# ---------------------------------------------------------------------------

class CubeLayerNorm(nn.Module):
    """Non-affine LayerNorm that upcasts to fp32 then back (matches cube)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = (dim,)
        self.eps = eps

    def forward(self, x):
        y = F.layer_norm(x.float(), self.dim, None, None, self.eps)
        return y.type_as(x)


class CubeRMSNorm(nn.Module):
    """Per-head RMSNorm with learnable weight, computed in fp32 (matches cube)."""

    def __init__(self, dim, eps=1e-5, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight).type_as(x)


# ---------------------------------------------------------------------------
# RoPE (faithful to cube3d/model/transformers/rope.py)
# ---------------------------------------------------------------------------

def apply_rotary_emb(x, freqs_cis, curr_pos_id=None):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if curr_pos_id is None:
        freqs_cis = freqs_cis[:, -x.shape[2]:].unsqueeze(1)
    else:
        freqs_cis = freqs_cis[:, curr_pos_id, :].unsqueeze(1)
    y = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return y.type_as(x)


def precompute_freqs_cis(dim, t, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=t.device) / dim))
    freqs = torch.outer(t.contiguous().view(-1), freqs).reshape(*t.shape, -1)
    return torch.polar(torch.ones_like(freqs), freqs)


def sdpa_with_rope(q, k, v, freqs_cis, attn_mask=None, curr_pos_id=None, is_causal=False):
    q = apply_rotary_emb(q, freqs_cis, curr_pos_id=curr_pos_id)
    k = apply_rotary_emb(k, freqs_cis, curr_pos_id=None)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0,
        is_causal=is_causal and attn_mask is None,
    )


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class Cache:
    def __init__(self, key_states, value_states):
        self.key_states = key_states
        self.value_states = value_states

    def update(self, curr_pos_id, k, v):
        self.key_states.index_copy_(2, curr_pos_id, k)
        self.value_states.index_copy_(2, curr_pos_id, v)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.gate_proj = operations.Linear(embed_dim, hidden_dim, bias=bias, dtype=dtype, device=device)
        self.up_proj = operations.Linear(embed_dim, hidden_dim, bias=bias, dtype=dtype, device=device)
        self.down_proj = operations.Linear(hidden_dim, embed_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SelfAttentionWithRotaryEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.c_qk = operations.Linear(embed_dim, 2 * embed_dim, bias=False, dtype=dtype, device=device)
        self.c_v = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.c_proj = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.q_norm = CubeRMSNorm(head_dim, dtype=dtype, device=device)
        self.k_norm = CubeRMSNorm(head_dim, dtype=dtype, device=device)

    def forward(self, x, freqs_cis, attn_mask=None, is_causal=False, kv_cache=None, curr_pos_id=None, decode=False):
        b, l, d = x.shape
        q, k = self.c_qk(x).chunk(2, dim=-1)
        v = self.c_v(x)
        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)
        k = k.view(b, l, self.num_heads, -1).transpose(1, 2)
        v = v.view(b, l, self.num_heads, -1).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if kv_cache is not None:
            if not decode:
                kv_cache.key_states[:, :, :k.shape[2], :].copy_(k)
                kv_cache.value_states[:, :, :k.shape[2], :].copy_(v)
            else:
                kv_cache.update(curr_pos_id, k, v)
            k = kv_cache.key_states
            v = kv_cache.value_states
        y = sdpa_with_rope(q, k, v, freqs_cis=freqs_cis, attn_mask=attn_mask,
                           curr_pos_id=curr_pos_id if decode else None, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(b, l, d)
        return self.c_proj(y)


class DecoderLayerWithRotaryEmbedding(nn.Module):
    """Single-stream decoder layer (shape tokens only)."""

    def __init__(self, embed_dim, num_heads, bias=True, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.ln_1 = CubeLayerNorm(embed_dim, eps=eps)
        self.attn = SelfAttentionWithRotaryEmbedding(embed_dim, num_heads, bias=bias, eps=eps,
                                                     dtype=dtype, device=device, operations=operations)
        self.ln_2 = CubeLayerNorm(embed_dim, eps=eps)
        self.mlp = SwiGLUMLP(embed_dim, embed_dim * 4, bias=bias, dtype=dtype, device=device, operations=operations)

    def forward(self, x, freqs_cis, attn_mask=None, is_causal=True, kv_cache=None, curr_pos_id=None, decode=False):
        x = x + self.attn(self.ln_1(x), freqs_cis=freqs_cis, attn_mask=attn_mask, is_causal=is_causal,
                          kv_cache=kv_cache, curr_pos_id=curr_pos_id, decode=decode)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Dual-stream blocks (faithful to dual_stream_attention.py)
# ---------------------------------------------------------------------------

class DismantledPreAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, query=True, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.query = query
        head_dim = embed_dim // num_heads
        if query:
            self.c_qk = operations.Linear(embed_dim, 2 * embed_dim, bias=False, dtype=dtype, device=device)
            self.q_norm = CubeRMSNorm(head_dim, dtype=dtype, device=device)
        else:
            self.c_k = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.k_norm = CubeRMSNorm(head_dim, dtype=dtype, device=device)
        self.c_v = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.num_heads = num_heads

    def _to_mha(self, x):
        return x.view(*x.shape[:2], self.num_heads, -1).transpose(1, 2)

    def forward(self, x):
        if self.query:
            q, k = self.c_qk(x).chunk(2, dim=-1)
            q = self.q_norm(self._to_mha(q))
        else:
            q = None
            k = self.c_k(x)
        k = self.k_norm(self._to_mha(k))
        v = self._to_mha(self.c_v(x))
        return (q, k, v)


class DismantledPostAttention(nn.Module):
    def __init__(self, embed_dim, bias=True, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.c_proj = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.ln_3 = CubeLayerNorm(embed_dim, eps=eps)
        self.mlp = SwiGLUMLP(embed_dim, embed_dim * 4, bias=bias, dtype=dtype, device=device, operations=operations)

    def forward(self, x, a):
        x = x + self.c_proj(a)
        x = x + self.mlp(self.ln_3(x))
        return x


class DualStreamAttentionWithRotaryEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, cond_pre_only=False, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.cond_pre_only = cond_pre_only
        self.pre_x = DismantledPreAttention(embed_dim, num_heads, query=True, bias=bias,
                                            dtype=dtype, device=device, operations=operations)
        self.pre_c = DismantledPreAttention(embed_dim, num_heads, query=not cond_pre_only, bias=bias,
                                            dtype=dtype, device=device, operations=operations)

    def forward(self, x, c, freqs_cis, attn_mask=None, is_causal=False, kv_cache=None, curr_pos_id=None, decode=False):
        if kv_cache is None or not decode:
            qkv_c = self.pre_c(c)
            qkv_x = self.pre_x(x)
            if self.cond_pre_only:
                q = qkv_x[0]
            else:
                q = torch.cat([qkv_c[0], qkv_x[0]], dim=2)
            k = torch.cat([qkv_c[1], qkv_x[1]], dim=2)
            v = torch.cat([qkv_c[2], qkv_x[2]], dim=2)
        else:
            is_causal = False
            q, k, v = self.pre_x(x)

        if kv_cache is not None:
            if not decode:
                kv_cache.key_states[:, :, :k.shape[2], :].copy_(k)
                kv_cache.value_states[:, :, :k.shape[2], :].copy_(v)
            else:
                kv_cache.update(curr_pos_id, k, v)
            k = kv_cache.key_states
            v = kv_cache.value_states

        if attn_mask is not None:
            if decode:
                attn_mask = attn_mask[..., curr_pos_id, :]
            else:
                attn_mask = attn_mask[..., -q.shape[2]:, :]

        y = sdpa_with_rope(q, k, v, freqs_cis=freqs_cis, attn_mask=attn_mask,
                           curr_pos_id=curr_pos_id if decode else None, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(x.shape[0], -1, x.shape[2])

        if y.shape[1] == x.shape[1]:
            return y, None
        y_c, y_x = torch.split(y, [c.shape[1], x.shape[1]], dim=1)
        return y_x, y_c


class DualStreamDecoderLayerWithRotaryEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, cond_pre_only=False, bias=True, eps=1e-6,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.ln_1 = CubeLayerNorm(embed_dim, eps=eps)
        self.ln_2 = CubeLayerNorm(embed_dim, eps=eps)
        self.attn = DualStreamAttentionWithRotaryEmbedding(embed_dim, num_heads, cond_pre_only=cond_pre_only,
                                                           bias=bias, dtype=dtype, device=device, operations=operations)
        self.post_1 = DismantledPostAttention(embed_dim, bias=bias, eps=eps, dtype=dtype, device=device, operations=operations)
        if not cond_pre_only:
            self.post_2 = DismantledPostAttention(embed_dim, bias=bias, eps=eps, dtype=dtype, device=device, operations=operations)

    def forward(self, x, c, freqs_cis, attn_mask=None, is_causal=True, kv_cache=None, curr_pos_id=None, decode=False):
        a_x, a_c = self.attn(
            self.ln_1(x),
            self.ln_2(c) if c is not None else None,
            freqs_cis=freqs_cis, attn_mask=attn_mask, is_causal=is_causal,
            kv_cache=kv_cache, curr_pos_id=curr_pos_id, decode=decode,
        )
        x = self.post_1(x, a_x)
        if a_c is not None:
            c = self.post_2(c, a_c)
        else:
            c = None
        return x, c


# ---------------------------------------------------------------------------
# DualStreamRoformer
# ---------------------------------------------------------------------------

class DualStreamRoformer(nn.Module):
    def __init__(
        self,
        n_layer=23,
        n_single_layer=1,
        rope_theta=10000,
        n_head=12,
        n_embd=1536,
        bias=True,
        eps=1e-6,
        shape_model_vocab_size=16384,
        shape_model_embed_dim=32,
        text_model_embed_dim=768,
        use_bbox=True,
        image_model=None,  # detection key; unused
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.n_layer = n_layer
        self.n_single_layer = n_single_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.rope_theta = rope_theta
        self.head_dim = n_embd // n_head

        self.text_proj = operations.Linear(text_model_embed_dim, n_embd, bias=bias, dtype=dtype, device=device)
        self.shape_proj = operations.Linear(shape_model_embed_dim, n_embd, bias=True, dtype=dtype, device=device)

        self.vocab_size = shape_model_vocab_size
        self.shape_bos_id = self.vocab_size
        self.shape_eos_id = self.vocab_size + 1
        self.padding_id = self.vocab_size + 2
        self.vocab_size += 3

        self.transformer = nn.ModuleDict(dict(
            wte=operations.Embedding(self.vocab_size, n_embd, padding_idx=self.padding_id, dtype=dtype, device=device),
            dual_blocks=nn.ModuleList([
                DualStreamDecoderLayerWithRotaryEmbedding(
                    n_embd, n_head, cond_pre_only=(i == n_layer - 1), bias=bias, eps=eps,
                    dtype=dtype, device=device, operations=operations,
                )
                for i in range(n_layer)
            ]),
            single_blocks=nn.ModuleList([
                DecoderLayerWithRotaryEmbedding(n_embd, n_head, bias=bias, eps=eps,
                                                dtype=dtype, device=device, operations=operations)
                for _ in range(n_single_layer)
            ]),
            ln_f=CubeLayerNorm(n_embd, eps=eps),
        ))

        self.lm_head = operations.Linear(n_embd, self.vocab_size, bias=False, dtype=dtype, device=device)

        self.use_bbox = use_bbox
        if use_bbox:
            self.bbox_proj = operations.Linear(3, n_embd, bias=True, dtype=dtype, device=device)

    def encode_text(self, text_embed):
        return self.text_proj(text_embed)

    def encode_token(self, tokens):
        return self.transformer.wte(tokens)

    def init_kv_cache(self, batch_size, cond_len, max_shape_tokens, dtype, device):
        max_all = cond_len + max_shape_tokens
        kv = [
            Cache(
                torch.zeros((batch_size, self.n_head, max_all, self.head_dim), dtype=dtype, device=device),
                torch.zeros((batch_size, self.n_head, max_all, self.head_dim), dtype=dtype, device=device),
            )
            for _ in range(len(self.transformer.dual_blocks))
        ]
        kv += [
            Cache(
                torch.zeros((batch_size, self.n_head, max_shape_tokens, self.head_dim), dtype=dtype, device=device),
                torch.zeros((batch_size, self.n_head, max_shape_tokens, self.head_dim), dtype=dtype, device=device),
            )
            for _ in range(len(self.transformer.single_blocks))
        ]
        return kv

    def forward(self, embed, cond, kv_cache=None, curr_pos_id=None, decode=False):
        b, l = embed.shape[:2]
        s = cond.shape[1]
        device = embed.device

        attn_mask = torch.tril(torch.ones(s + l, s + l, dtype=torch.bool, device=device))

        position_ids = torch.arange(l, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        s_freqs_cis = precompute_freqs_cis(self.head_dim, position_ids, theta=self.rope_theta)

        position_ids = torch.cat([
            torch.zeros([b, s], dtype=torch.long, device=device),
            position_ids,
        ], dim=1)
        d_freqs_cis = precompute_freqs_cis(self.head_dim, position_ids, theta=self.rope_theta)

        if kv_cache is not None and decode:
            embed = embed[:, curr_pos_id, :]

        h = embed
        c = cond
        layer_idx = 0
        for block in self.transformer.dual_blocks:
            h, c = block(
                h, c=c, freqs_cis=d_freqs_cis, attn_mask=attn_mask, is_causal=True,
                kv_cache=kv_cache[layer_idx] if kv_cache is not None else None,
                curr_pos_id=curr_pos_id + s if curr_pos_id is not None else None,
                decode=decode,
            )
            layer_idx += 1
        for block in self.transformer.single_blocks:
            h = block(
                h, freqs_cis=s_freqs_cis, attn_mask=None, is_causal=True,
                kv_cache=kv_cache[layer_idx] if kv_cache is not None else None,
                curr_pos_id=curr_pos_id, decode=decode,
            )
            layer_idx += 1

        h = self.transformer.ln_f(h)
        return self.lm_head(h)
