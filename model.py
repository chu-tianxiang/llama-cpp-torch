# Modified from https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.distributed import _functional_collectives as funcol

import register_lib

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    architecture: str = "llama"
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = -1
    rope_base: float = 10000
    rope_type: str = "none"
    rope_dim: int = -1
    norm_eps: float = 1e-5
    moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    hidden_act: str = "silu"
    mlp_gate: bool = True
    layernorm: bool = False
    logit_scale: float = 1.0

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.head_dim == -1:
            self.head_dim = self.dim // self.n_head
        if self.rope_dim == -1:
            self.rope_dim = self.head_dim


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

        self.token_embd = Embedding(config.vocab_size, config.dim)
        self.blk = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        if config.layernorm:
            self.output_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        else:
            self.output_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.head_dim
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blk:
            b.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.rope_dim, self.config.rope_base)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.token_embd(idx)
        if self.config.architecture == "gemma":
            x = x * (self.config.dim ** 0.5)
        if self.config.architecture == "minicpm":
            x = x * 12.0

        for i, layer in enumerate(self.blk):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.output_norm(x)
        logits = self.output(x)
        if self.config.architecture == "minicpm":
            logits = logits / (self.config.dim // 256)
        logits *= self.config.logit_scale
        return logits

    @classmethod
    def from_json(cls, name: str):
        config = json.load(open(name, 'r'))
        return cls(ModelArgs(**config))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        assert config.dim % config.n_head == 0

        # Attention norm
        if config.layernorm:
            self.attn_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        else:
            self.attn_norm = RMSNorm(config.dim, config.norm_eps)

        # Attention layer
        # https://github.com/pacman100/llama.cpp/blob/ee5b171250f707b08334aa8dcda259888bc2ccc6/gguf-py/gguf/tensor_mapping.py#L97
        if config.architecture in ["qwen", "phi2"]:
            self.concat_qkv = True
            self.attn_qkv = Linear(config.dim, config.head_dim * (config.n_head + config.n_local_heads * 2), bias=True)
        else:
            self.concat_qkv = False
            self.attn_q = Linear(config.dim, config.n_head * config.head_dim, bias=True)
            self.attn_k = Linear(config.dim, config.n_local_heads * config.head_dim, bias=True)
            self.attn_v = Linear(config.dim, config.n_local_heads * config.head_dim, bias=True)
        self.attn_output = Linear(config.n_head * config.head_dim, config.dim, bias=True)
        self.kv_cache = None

        # ffn norm
        if config.layernorm:
            self.ffn_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        else:
            self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        if config.hidden_act == "gelu_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        elif config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            self.act_fn = nn.SiLU()

        # ffn layer
        if config.moe:
            self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=True)
            self.ffn_gate = nn.ModuleList(Linear(config.dim, config.intermediate_size, bias=True) for _ in range(config.num_experts))
            self.ffn_up = nn.ModuleList(Linear(config.dim, config.intermediate_size, bias=True) for _ in range(config.num_experts))
            self.ffn_down = nn.ModuleList(Linear(config.intermediate_size, config.dim, bias=True) for _ in range(config.num_experts))
        else:
            if config.mlp_gate:
                self.ffn_gate = Linear(config.dim, config.intermediate_size, bias=True)
            self.ffn_up = Linear(config.dim, config.intermediate_size, bias=True)
            self.ffn_down = Linear(config.intermediate_size, config.dim, bias=True)

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.moe = config.moe
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.rope_type = config.rope_type
        self.rope_dim = config.rope_dim
        self.world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        attn_x = self.attn_norm(x)
        # attention
        if self.concat_qkv:
            qkv = self.attn_qkv(attn_x).view(bsz, seqlen, -1, self.head_dim)
            q, k, v = qkv.split([self.n_head, self.n_local_heads, self.n_local_heads],
                                dim=2)
        else:
            q = self.attn_q(attn_x).view(bsz, seqlen, self.n_head, self.head_dim)
            k = self.attn_k(attn_x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = self.attn_v(attn_x).view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.rope_dim != self.head_dim:
            q_rot, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
            k_rot, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]
            q_rot = apply_rotary_emb(q_rot, freqs_cis, self.rope_type)
            k_rot = apply_rotary_emb(k_rot, freqs_cis, self.rope_type)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        else:
            q = apply_rotary_emb(q, freqs_cis, self.rope_type)
            k = apply_rotary_emb(k, freqs_cis, self.rope_type)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        y = self.attn_output(y)

        if self.world_size > 1:
            y = funcol.all_reduce(y, "sum", list(range(self.world_size)))

        if self.config.architecture == "minicpm":
            y = y * 1.4 / math.sqrt(self.config.n_layer)

        y = x + y

        if self.config.architecture in ["phi2", "command-r"]:
            mlp_y = attn_x
        else:
            mlp_y = self.ffn_norm(y)

        # mlp
        if self.moe:
            # reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py
            # This is inefficient for small batch size since it calculates all experts
            mlp_y = mlp_y.view(-1, mlp_y.shape[-1])
            routing_weights = F.softmax(self.ffn_gate_inp(mlp_y), dim=1)
            routing_weights, selected_experts = torch.topk(routing_weights,
                                                           self.num_experts_per_tok,
                                                           dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            z = None
            for idx in range(self.num_experts):
                if self.ffn_gate[idx] is None: continue
                z_idx = self.ffn_down[idx](self.act_fn(self.ffn_gate[idx](mlp_y)) * self.ffn_up[idx](mlp_y))
                expert_mask = (selected_experts == idx)
                expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                     keepdim=True)
                z_idx = z_idx * expert_weights
                z = z_idx if z is None else z + z_idx
        elif self.config.mlp_gate:
            z = self.ffn_down(self.act_fn(self.ffn_gate(mlp_y)) * self.ffn_up(mlp_y))
        else:
            z = self.ffn_down(self.act_fn(self.ffn_up(mlp_y)))

        if self.world_size > 1:
            z = funcol.all_reduce(z, "sum", list(range(self.world_size)))

        if self.config.architecture == "minicpm":
            z = z * 1.4 / math.sqrt(self.config.n_layer)
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
    """Quantized linear layer"""
    def __init__(self, infeatures, outfeatures, bias, **kwargs):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        # Fake weight
        self.register_buffer('weight', torch.nn.parameter.UninitializedBuffer())
        self.register_buffer('weight_type', torch.zeros((), dtype=torch.int))

        if bias:
            self.register_buffer(
                'bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + 'weight' in state_dict:
             weight = state_dict[prefix + 'weight']
             self.weight.materialize(weight.shape, dtype=weight.dtype)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        xshape = x.view(-1, x.shape[-1])
        if self.weight_type_int < 2:
            output = xshape @ self.weight.view(self.outfeatures, self.infeatures).T
        # Force to use dequant for 2-bit model for now
        elif xshape.shape[0] == 1:
            output = torch.ops.llama_cpp.ggml_mul_mat_vec_a8(self.weight, xshape, self.weight_type_int, self.outfeatures)
        elif xshape.shape[0] < 8 and self.weight_type_int < 16:
            output = torch.ops.llama_cpp.ggml_mul_mat_a8(self.weight, xshape, self.weight_type_int, self.outfeatures)
        else:
            weight = torch.ops.llama_cpp.ggml_dequantize(self.weight, self.weight_type_int, self.outfeatures, self.infeatures)
            output = xshape @ weight.T
        if self.bias is not None:
            output = output + self.bias
        output = output.view(*x.shape[:-1], self.outfeatures)
        return output

class Embedding(nn.Module):
    """Quantized embedding layer"""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # Fake weight
        self.register_buffer('weight', torch.nn.parameter.UninitializedBuffer())
        self.register_buffer('weight_type', torch.zeros((), dtype=torch.int))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + 'weight' in state_dict:
             weight = state_dict[prefix + 'weight']
             self.weight.materialize(weight.shape, dtype=weight.dtype)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, ind):
        if self.weight_type_int < 2:
            return torch.embedding(self.weight.view(self.vocab_size, self.dim), ind)
        ind_flat = ind.flatten()
        quant = torch.index_select(self.weight.view(self.vocab_size, -1), dim=0, index=ind_flat)
        dequant = torch.ops.llama_cpp.ggml_dequantize(quant, self.weight_type_int,
                                                      self.dim, ind_flat.shape[0])
        return dequant.view(*ind.shape, self.dim)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, rope_type: str) -> Tensor:
    if rope_type == "norm":
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
                xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        )
    else:
        xshaped = x.float().reshape(*x.shape[:-1], 2, -1)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(-1), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0, :] * freqs_cis[..., 0] - xshaped[..., 1, :] * freqs_cis[..., 1],
                xshaped[..., 1, :] * freqs_cis[..., 0] + xshaped[..., 0, :] * freqs_cis[..., 1],
            ],
            -1,
        )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
