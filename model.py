# Modified from https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import register_lib

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.token_embd = nn.Embedding(config.vocab_size, config.dim)
        self.blk = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.output_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Pre-dequantize the embedding layer because nn.Embedding doesn't support quantized weight
        if prefix + 'token_embd.weight' in state_dict:
             dtype = int(state_dict[prefix + 'token_embd.weight_type'])
             if dtype >= 2:
                 weight = torch.ops.llama_cpp.ggml_dequantize(state_dict[prefix + 'token_embd.weight'].cuda(), dtype, self.config.vocab_size, self.config.dim)
                 state_dict[prefix + 'token_embd.weight'] = weight.cpu()
             del state_dict[prefix + 'token_embd.weight_type']
        # ignore weight type in layernorm
        for key in tuple(state_dict.keys()):
            if 'norm.weight_type' in key:
                del state_dict[key]

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blk:
            b.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.token_embd(idx)

        for i, layer in enumerate(self.blk):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.output_norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_json(cls, name: str):
        config = json.load(open(name, 'r'))
        return cls(ModelArgs(**config))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        # Attention norm
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)

        # Attention layer
        self.attn_q = Linear(config.dim, config.n_head * config.head_dim, bias=False)
        self.attn_k = Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)
        self.attn_v = Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)
        self.attn_output = Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # ffn norm
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

        # ffn layer
        self.ffn_gate = Linear(config.dim, config.intermediate_size, bias=False)
        self.ffn_up = Linear(config.dim, config.intermediate_size, bias=False)
        self.ffn_down = Linear(config.intermediate_size, config.dim, bias=False)

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        attn_x = self.attn_norm(x)
        # attention
        q = self.attn_q(attn_x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.attn_k(attn_x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = self.attn_v(attn_x).view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = x + self.attn_output(y)
        mlp_y = self.ffn_norm(y)

        # mlp
        z = self.ffn_down(F.silu(self.ffn_gate(mlp_y)) * self.ffn_up(mlp_y))
        z = z + y
        return z


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Linear(nn.Module):
    def __init__(self, infeatures, outfeatures, bias, **kwargs):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        # Fake weight
        self.register_buffer('weight', torch.zeros(()))
        self.register_buffer('weight_type', torch.zeros((), dtype=torch.int))

        if bias:
            self.register_buffer(
                'bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + 'weight' in state_dict:
             setattr(self, 'weight', state_dict[prefix + 'weight'].clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        xshape = x.view(-1, x.shape[-1])
        if self.weight_type_int < 2:
            output = x @ self.weight.T
        elif xshape.shape[0] == 1:
            output = torch.ops.llama_cpp.ggml_mul_mat_vec(self.weight, xshape, self.weight_type_int, self.outfeatures)
        else:
            weight = torch.ops.llama_cpp.ggml_dequantize(self.weight, self.weight_type_int, self.outfeatures, self.infeatures)
            output = x @ weight.T
        if self.bias is not None:
            output = output + self.bias
        output = output.view(*x.shape[:-1], self.outfeatures)
        return output


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
