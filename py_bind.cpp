#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor ggml_dequantize(
    torch::Tensor X,
    int8_t type,
    int64_t m,
    int64_t n
);

torch::Tensor ggml_mul_mat_vec(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int8_t type,
    int64_t m
);

torch::Tensor ggml_mul_mat_vec_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int8_t type,
    int64_t row
);

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int8_t type,
    int64_t row
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ggml_dequantize", &ggml_dequantize, "ggml_dequantize");
    m.def("ggml_mul_mat_vec", &ggml_mul_mat_vec, "ggml_mul_mat_vec");
    m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8, "ggml_mul_mat_vec_a8");
    m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "ggml_mul_mat_a8");
}
