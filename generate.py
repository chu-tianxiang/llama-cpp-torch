# Modified from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.distributed as dist
from sentencepiece import SentencePieceProcessor
from gpt2_tokenizer import GPT2Tokenizer

from model import Transformer
from tp import maybe_init_dist, _get_world_size

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token
    callback(next_token)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
    seq[T + 1:] = torch.cat(generated_tokens)

    return seq


def encode_tokens(tokenizer, special_tokens, string, bos=True, device='cuda'):
    text_list = [string]
    for token in special_tokens:
        new_text_list = []
        for text in text_list:
            text = text.split(token)
            new_text_list.extend([e for pair in zip(text, [token] * (len(text) - 1)) for e in pair] + [text[-1]])
        text_list = new_text_list

    tokens = []
    for text in text_list:
        if not text:
            continue
        if text in special_tokens:
            tokens.append(int(special_tokens[text]))
        else:
            tokens.extend(tokenizer.encode(text))
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision, use_tp):
    with torch.device('meta'):
        model = Transformer.from_json(str(checkpoint_path / "config.json"))

    checkpoint = torch.load(str(checkpoint_path / "pytorch_model.bin"), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, strict=False, assign=True)
    # Fixed tied embedding
    if model.output.weight.device == torch.device("meta"):
        model.output.weight = model.token_embd.weight
        model.output.weight_type = model.token_embd.weight_type

    model = model._apply(lambda t: torch.zeros_like(t, device="cpu")
                             if t.device == torch.device("meta") else t)
    for name, module in model.named_modules():
        if hasattr(module, "weight_type"):
            module.weight_type_int = int(module.weight_type)

    if use_tp:
        from tp import apply_tp
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 20,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("."),
    compile: bool = True,
    compile_prefill: bool = False,
    device='cuda',
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    tokenizer_path = checkpoint_path / "tokenizer.model"
    use_spm = True
    if not tokenizer_path.is_file():
        use_spm = False
        tokenizer_path = (checkpoint_path / "vocab.json",
                          checkpoint_path / "merges.txt",
                          checkpoint_path / "tokenizer_config.json")

    B_INST, E_INST = "[INST]", "[/INST]"
    if "qwen" in str(checkpoint_path).lower():
        B_INST, E_INST = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    precision = torch.float16
    is_chat = "chat" in str(checkpoint_path).lower()

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    if use_spm:
        tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    else:
        tokenizer = GPT2Tokenizer(*tokenizer_path)
    tokenizer_conf = json.load(open(checkpoint_path / "tokenizer_config.json"))
    special_tokens = {v["content"]: k for k, v in tokenizer_conf.get("added_tokens_decoder", {}).items()}
    encoded = encode_tokens(tokenizer, special_tokens, prompt,
                            bos=tokenizer_conf.get("add_bos_token", False), device=device)
    prompt_length = encoded.size(0)

    #torch.manual_seed(1234)
    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {
        'tokens_per_sec': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device) # MKG
        if i >= 0 and interactive:
            if not use_tp or rank == 0:
                prompt = input("What is your prompt? ")
            else:
                prompt = ""
            if use_tp:
                prompt_list = [None for _ in range(_get_world_size())]
                dist.all_gather_object(prompt_list, prompt)
                prompt = prompt_list[0]

            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, special_tokens, prompt.strip(), bos=tokenizer_conf.get("add_bos_token", False), device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        y = generate(
            model,
            encoded,
            max_new_tokens,
            interactive=interactive,
            callback=callback,
            temperature=temperature,
            top_k=top_k,
        )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        if not interactive:
            print(tokenizer.decode(y.tolist()))
        else:
            print()
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--device', type=str, default="cuda", help='device to use')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.device
    )
