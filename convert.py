import os
import json

import torch
from gguf import GGUFReader
from sentencepiece import sentencepiece_model_pb2


def convert_to_state_dict(checkpoint, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = {}
    result = GGUFReader(checkpoint)
    architecture = result.fields['general.architecture']
    architecture = str(bytes(architecture.parts[architecture.data[0]]), encoding = 'utf-8')
    if architecture != "llama":
        print(f"Unsupported architecture {architecture}")
        return
    # write tensor
    for ts in result.tensors:
        if hasattr(ts.data.dtype, 'names') and ts.data.dtype.names is not None:
            for name in ts.data.dtype.names:
                state_dict[ts.name + "_" + name] = torch.tensor(ts.data[name])
        else:
            state_dict[ts.name] = torch.tensor(ts.data)
        state_dict[ts.name.replace("weight", "weight_type")] = torch.tensor(int(ts.tensor_type), dtype=torch.int)
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    # write vocab
    vocab = sentencepiece_model_pb2.ModelProto()
    vocab_size = len(result.fields['tokenizer.ggml.token_type'].data)
    # BPE
    vocab.trainer_spec.model_type = 2
    vocab.trainer_spec.vocab_size = vocab_size
    vocab.trainer_spec.byte_fallback = True
    vocab.normalizer_spec.remove_extra_whitespaces = False
    tokens = result.fields['tokenizer.ggml.tokens']
    scores = result.fields['tokenizer.ggml.scores']
    types = result.fields['tokenizer.ggml.token_type']
    for i in range(vocab_size):
        new_token = vocab.SentencePiece()
        new_token.piece = str(bytes(tokens.parts[tokens.data[i]]), encoding = 'utf-8')
        new_token.score = scores.parts[scores.data[i]]
        # llama.cpp tokentype is the same with sentencepiece token type
        new_token.type = int(types.parts[types.data[i]])
        vocab.pieces.append(new_token)
    with open(os.path.join(save_dir, "tokenizer.model"), 'wb') as f:
        f.write(vocab.SerializeToString())
    # write config
    context_length = int(result.fields['llama.context_length'].parts[-1])
    n_layer = int(result.fields['llama.block_count'].parts[-1])
    n_head = int(result.fields['llama.attention.head_count'].parts[-1])
    n_local_heads = int(result.fields['llama.attention.head_count_kv'].parts[-1])
    intermediate_size = int(result.fields['llama.feed_forward_length'].parts[-1])
    norm_eps = float(result.fields['llama.attention.layer_norm_rms_epsilon'].parts[-1])
    dim = int(result.fields['llama.embedding_length'].parts[-1])
    model_config= {
        "block_size": context_length,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "dim": dim,
        "intermediate_size": intermediate_size,
        "n_local_heads": n_local_heads,
        "norm_eps": norm_eps
    }
    if 'llama.rope.freq_base' in result.fields:
        model_config['rope_base'] = float(result.fields['llama.rope.freq_base'].parts[-1])
    if 'llama.expert_count' in result.fields:
        model_config['num_experts'] = int(result.fields['llama.expert_count'].parts[-1])
        model_config['num_experts_per_tok'] = int(result.fields['llama.expert_used_count'].parts[-1])
        model_config['moe'] = (model_config['num_experts'] > 1)

    json.dump(model_config, open(os.path.join(save_dir, "config.json"), 'w'), indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert GGUF checkpoints to torch')

    parser.add_argument('--input', type=str, help='The path to GGUF file')
    parser.add_argument('--output', type=str, help='The path to output directory')
    args = parser.parse_args()
    convert_to_state_dict(args.input, args.output)
