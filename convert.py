import os
import json

import torch
from gguf_reader import GGUFReader
from sentencepiece import sentencepiece_model_pb2


def convert_to_state_dict(checkpoint, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = {}
    result = GGUFReader(checkpoint)
    architecture = result.fields['general.architecture']
    architecture = str(bytes(architecture.parts[architecture.data[0]]), encoding = 'utf-8')
    if architecture not in ["llama", "qwen2", "internlm2", "starcoder2", "qwen",
                            "stablelm", "orion", "minicpm", "gemma", "xverse", "command-r"]:
        print(f"Unsupported architecture {architecture}")
        return
    # write tensor
    for ts in result.tensors:
        if hasattr(ts.data.dtype, 'names') and ts.data.dtype.names is not None:
            for name in ts.data.dtype.names:
                state_dict[ts.name + "_" + name] = torch.tensor(ts.data[name])
        else:
            state_dict[ts.name] = torch.tensor(ts.data)
        if "weight" in ts.name:
            state_dict[ts.name.replace("weight", "weight_type")] = torch.tensor(int(ts.tensor_type), dtype=torch.int)
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    # write vocab
    # note we ignore added tokens for simplicity
    vocab_type = result.fields["tokenizer.ggml.model"]
    vocab_type = str(bytes(vocab_type.parts[vocab_type.data[0]]), encoding = 'utf-8')
    if vocab_type == "gpt2":
        # bpe vocab
        merges = result.fields["tokenizer.ggml.merges"]
        with open(os.path.join(save_dir, "merges.txt"), 'w') as f:
            for idx in merges.data:
                data = str(bytes(merges.parts[idx]), encoding = 'utf-8')
                f.write(f"{data}\n")
        tokens = result.fields['tokenizer.ggml.tokens']
        types = result.fields['tokenizer.ggml.token_type']
        vocab_size = len(tokens.data)
        vocab = {}
        special_vocab = {}
        vocab_list = []
        for i, idx in enumerate(tokens.data):
            token = str(bytes(tokens.parts[idx]), encoding='utf-8')
            token_type = int(types.parts[types.data[i]])
            #if (token.startswith("[PAD") or token.startswith("<dummy")) and token_type == 4:
            #    break
            vocab_list.append(token)
            vocab[token] = i
            if token_type == 3:
                special_vocab[i] = {"content": token, "special": True}
        json.dump(vocab, open(os.path.join(save_dir, "vocab.json"), 'w'),
                  ensure_ascii=False, indent=2)
    else:
        # sentencepiece
        vocab = sentencepiece_model_pb2.ModelProto()
        vocab_list = []
        vocab_size = len(result.fields['tokenizer.ggml.token_type'].data)
        # model_type = BPE
        vocab.trainer_spec.model_type = 2
        vocab.trainer_spec.vocab_size = vocab_size
        if architecture not in ['orion']:
            vocab.trainer_spec.byte_fallback = True
        vocab.normalizer_spec.remove_extra_whitespaces = False
        tokens = result.fields['tokenizer.ggml.tokens']
        if 'tokenizer.ggml.scores' in result.fields:
            scores = result.fields['tokenizer.ggml.scores']
        else:
            scores = None
        types = result.fields['tokenizer.ggml.token_type']
        special_vocab = {}
        has_unk = False
        for i in range(vocab_size):
            new_token = vocab.SentencePiece()
            new_token.piece = str(bytes(tokens.parts[tokens.data[i]]), encoding = 'utf-8')
            if scores:
                new_token.score = scores.parts[scores.data[i]]
            # llama.cpp tokentype is the same with sentencepiece token type
            new_token.type = int(types.parts[types.data[i]])
            if new_token.type == 2:
                has_unk = True
            # fix for xverse, is it correct?
            if new_token.piece == r"<b'\x00'>":
                new_token.piece = rb'\x00'
                new_token.type = 1
            vocab_list.append(new_token.piece)
            vocab.pieces.append(new_token)
            if new_token.type == 3:
                special_vocab[i] = {"content": new_token.piece, "special": True}
        # hf_vocab doesn't correctly set unk token type, so we force one
        if not has_unk:
            vocab.pieces[0].type = 2

        with open(os.path.join(save_dir, "tokenizer.model"), 'wb') as f:
            f.write(vocab.SerializeToString())

    tokenizer_conf = {}
    if 'tokenizer.ggml.bos_token_id' in result.fields:
        tokenizer_conf["bos_token"] = vocab_list[int(result.fields['tokenizer.ggml.bos_token_id'].parts[-1])]
    if 'tokenizer.ggml.eos_token_id' in result.fields:
        tokenizer_conf["eos_token"] = vocab_list[int(result.fields['tokenizer.ggml.eos_token_id'].parts[-1])]
    if 'tokenizer.ggml.padding_token_id' in result.fields:
        tokenizer_conf["pad_token"] = vocab_list[int(result.fields['tokenizer.ggml.padding_token_id'].parts[-1])]
    if 'tokenizer.ggml.unknown_token_id' in result.fields:
        tokenizer_conf["unk_token"] = vocab_list[int(result.fields['tokenizer.ggml.unknown_token_id'].parts[-1])]
    if 'tokenizer.ggml.add_bos_token' in result.fields:
        tokenizer_conf["add_bos_token"] = bool(result.fields['tokenizer.ggml.add_bos_token'].parts[-1])
    if 'tokenizer.ggml.add_eos_token' in result.fields:
        tokenizer_conf["add_eos_token"] = bool(result.fields['tokenizer.ggml.add_eos_token'].parts[-1])
    if 'tokenizer.chat_template' in result.fields:
        tokenizer_conf['chat_template'] = str(bytes(result.fields['tokenizer.chat_template'].parts[-1]), encoding = 'utf-8')
    if special_vocab:
        tokenizer_conf["added_tokens_decoder"] = special_vocab
    json.dump(tokenizer_conf, open(os.path.join(save_dir, "tokenizer_config.json"), 'w'), indent=2)


    # write config
    context_length = int(result.fields[f'{architecture}.context_length'].parts[-1])
    n_layer = int(result.fields[f'{architecture}.block_count'].parts[-1])
    n_head = int(result.fields[f'{architecture}.attention.head_count'].parts[-1])
    intermediate_size = int(result.fields[f'{architecture}.feed_forward_length'].parts[-1])
    # qwen use ffn_size / 2 for ffn layers
    if architecture == "qwen":
        intermediate_size = intermediate_size / 2
    dim = int(result.fields[f'{architecture}.embedding_length'].parts[-1])
    if f'{architecture}.logit_scale' in result.fields:
        logit_scale = float(result.fields[f'{architecture}.logit_scale'].parts[-1])
    else:
        logit_scale = 1.0
    # https://github.com/ggerganov/llama.cpp/blob/9731134296af3a6839cd682e51d9c2109a871de5/llama.cpp#L12301
    if architecture in ["qwen2", "gemma", "qwen", "stablelm", "starcoder2", "phi2"]:
        rope_type = "neox"
    elif architecture in ["llama", "internlm2", "baichuan", "startcoder", "orion", "minicpm",
                          "xverse", "command-r"]:
        rope_type = "norm"
    else:
        rope_type = "none"

    if architecture in ["starcoder2", "phi2", "gemma"]:
        hidden_act = "gelu_tanh"
    else:
        hidden_act = "silu"

    if architecture in ["starcoder2", "phi2"]:
        mlp_gate = False
    else:
        mlp_gate = True

    if architecture in ["starcoder2", "phi2", "stablelm", "orion", "command-r"]:
        layernorm = True
    else:
        layernorm = False
    model_config= {
        "architecture": architecture,
        "block_size": context_length,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "dim": dim,
        "intermediate_size": intermediate_size,
        "hidden_act": hidden_act,
        "rope_type": rope_type,
        "mlp_gate": mlp_gate,
        "layernorm": layernorm,
        "logit_scale": logit_scale
    }
    if f'{architecture}.attention.head_count_kv' in result.fields:
        model_config['n_local_heads'] = int(result.fields[f'{architecture}.attention.head_count_kv'].parts[-1])
    if f'{architecture}.attention.layer_norm_rms_epsilon' in result.fields:
        model_config['norm_eps'] = float(result.fields[f'{architecture}.attention.layer_norm_rms_epsilon'].parts[-1])
    if f'{architecture}.attention.key_length' in result.fields:
        model_config['head_dim'] = int(result.fields[f'{architecture}.attention.key_length'].parts[-1])
    if f'{architecture}.rope.freq_base' in result.fields:
        model_config['rope_base'] = float(result.fields[f'{architecture}.rope.freq_base'].parts[-1])
    if f'{architecture}.rope_dimension_count' in result.fields:
        model_config['rope_dim'] = int(result.fields[f'{architecture}.rope_dimension_count'].parts[-1])
    if f'{architecture}.expert_count' in result.fields:
        model_config['num_experts'] = int(result.fields[f'{architecture}.expert_count'].parts[-1])
        model_config['num_experts_per_tok'] = int(result.fields[f'{architecture}.expert_used_count'].parts[-1])
        model_config['moe'] = (model_config['num_experts'] > 1)

    json.dump(model_config, open(os.path.join(save_dir, "config.json"), 'w'), indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert GGUF checkpoints to torch')

    parser.add_argument('--input', type=str, help='The path to GGUF file')
    parser.add_argument('--output', type=str, help='The path to output directory')
    args = parser.parse_args()
    convert_to_state_dict(args.input, args.output)
