import torch
import llamacpp_cuda

import torch._custom_ops
from torch import Tensor

my_lib = torch.library.Library("llama_cpp", "DEF")

@torch._custom_ops.custom_op("llama_cpp::ggml_dequantize")
def ggml_dequantize(x: Tensor, type: int, m: int, n: int) -> Tensor:
    raise NotImplementedError()

@torch._custom_ops.impl_abstract("llama_cpp::ggml_dequantize")
def ggml_dequantize_abs(x: Tensor, type: int, m: int, n: int) -> Tensor:
    return x.new_empty((m, n), dtype=torch.half)

@torch._custom_ops.impl("llama_cpp::ggml_dequantize", device_types="cuda")
def ggml_dequantize_cuda(x: Tensor, type: int, m: int, n: int) -> Tensor:
    return llamacpp_cuda.ggml_dequantize(x, type, m, n)

@torch._custom_ops.custom_op("llama_cpp::ggml_mul_mat_vec")
def ggml_mul_mat_vec(x: Tensor, y: Tensor, type: int, m: int) -> Tensor:
    raise NotImplementedError()

@torch._custom_ops.impl_abstract("llama_cpp::ggml_mul_mat_vec")
def ggml_mul_mat_vec_abs(x: Tensor, y: Tensor, type: int, m: int) -> Tensor:
    assert x.device == y.device
    result = x.new_empty((1, m), dtype=torch.half)
    return result

@torch._custom_ops.impl("llama_cpp::ggml_mul_mat_vec", device_types="cuda")
def ggml_mul_mat_vec_cuda(x: Tensor, y: Tensor, type: int, m: int) -> Tensor:
    return llamacpp_cuda.ggml_mul_mat_vec(x, y, type, m)
