# Convert [`llama.cpp`](https://github.com/ggerganov/llama.cpp) to Pytorch

Note: this is a work in progress.

The `llama.cpp` library is a cornerstone in language modeling with a variety of quantization techniques, but it's largely used within its own ecosystem. This repo's aim is to make these methods more accessible to the PyTorch community.

This repo provides an example for converting GGUF files back into PyTorch state dict, allowing you to run inference purely in PyTorch. For now only LLaMA and Mixtral is supported.

The code is largely inspired by the original [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and [`GPT-Fast`](https://github.com/pytorch-labs/gpt-fast).

## Getting Started

* Install the CUDA extension

```bash
python setup.py install
```

* Convert GGUF file to torch state dict

```bash
python convert.py --input tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --output TinyLlama-Q4_K_M
```

* Running inference

```bash
python generate.py --checkpoint_path TinyLlama-Q4_K_M --interactive --compile
```

`torch.compile` will take minutes, you can also run in eager mode without `--compile` flag.


## Todo
* Add support to more model
* Optimize Mixtral performace