# Modified from https://github.com/pytorch-labs/gpt-fast/blob/main/tp.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from model import Transformer, Linear
from constants import GGML_QUANT_SIZES

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_local():
    return _get_rank() == 0

def local_break():
    if is_local():
        breakpoint()
    dist.barrier()

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank


def _apply_tp_linear(linear: Linear, style: str) -> None:
    rank = _get_rank()
    world_size = _get_world_size()

    block_size = GGML_QUANT_SIZES[linear.weight_type_int][0]
    assert linear.infeatures % block_size == 0

    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    weight = linear.weight.view(linear.outfeatures, linear.infeatures // block_size, -1)
    if style == "colwise":
        sharded_weight = shard(weight, 0)
        linear.outfeatures = linear.outfeatures // world_size
        if linear.bias is not None:
            linear.bias = nn.Parameter(shard(linear.bias, 0), requires_grad=False)
    else:
        sharded_weight = shard(weight, 1)
        linear.infeatures = linear.infeatures // world_size
    linear.weight = nn.Parameter(sharded_weight.contiguous().view(-1), requires_grad=False)


def _apply_tp_Transformer(Transformer: Transformer) -> None:
    # overwrite config before Transformer.setup_cache is called
    world_size = _get_world_size()
    Transformer.config.n_head = Transformer.config.n_head // world_size
    Transformer.config.dim = Transformer.config.dim // world_size
    Transformer.config.n_local_heads = Transformer.config.n_local_heads // world_size


def apply_tp(model: Transformer) -> None:
    rank = _get_rank()
    world_size = _get_world_size()
    _apply_tp_Transformer(model)
    for block in model.blk:
        if isinstance(block.ffn_gate, nn.ModuleList):
            # Expert parallel for MOE
            expert_indicies = np.array_split(range(
                block.num_experts), world_size)[rank].tolist()
            block.ffn_gate = nn.ModuleList(block.ffn_gate[i] if i in expert_indicies else None for i in range(block.num_experts))
            block.ffn_up = nn.ModuleList(block.ffn_up[i] if i in expert_indicies else None for i in range(block.num_experts))
            block.ffn_down = nn.ModuleList(block.ffn_down[i] if i in expert_indicies else None for i in range(block.num_experts))
        else:
            _apply_tp_linear(block.ffn_gate, "colwise")
            _apply_tp_linear(block.ffn_up, "colwise")
            _apply_tp_linear(block.ffn_down, "rowwise")
        _apply_tp_linear(block.attn_q, "colwise")
        _apply_tp_linear(block.attn_k, "colwise")
        _apply_tp_linear(block.attn_v, "colwise")
        _apply_tp_linear(block.attn_output, "rowwise")

        # overwrite
        block.n_head = block.n_head // world_size
        block.dim = block.dim // world_size
        block.head_dim = block.dim // block.n_head
        block.n_local_heads = block.n_local_heads // world_size
