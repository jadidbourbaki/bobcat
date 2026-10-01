"""Dump reference activations from transformers for bobcat's tests.

Usage:

    uv run python ref_dump.py --model LiquidAI/LFM2.5-350M \
        --out ../models/ref/LFM2.5-350M

    uv run python ref_dump.py --model LiquidAI/LFM2.5-350M \
        --gguf ../models/LFM2.5-350M-Q8_0.gguf \
        --out ../models/ref/LFM2.5-350M-Q8_0

    uv run python ref_dump.py --model LiquidAI/LFM2.5-8B-A1B \
        --gguf ../models/LFM2.5-8B-A1B-Q4_K_M.gguf \
        --out ../models/ref/LFM2.5-8B-A1B-Q4_K_M

With `--gguf`, the weights come from the GGUF file, dequantized to
float32, and the tokenizer still comes from `--model`. transformers
reads LFM2 GGUF files itself. For the mixture-of-experts `lfm2moe`
files, which transformers cannot read, the script builds the model from
the configuration of `--model` and copies in each tensor that the gguf
package dequantizes. The float32 8B model needs about 33 GB of memory.
The script runs the model in float32 on the CPU and writes raw
little-endian files to the output directory. `tokens.i32` holds the
prompt token ids. `embedding.f32`, `layer_NN.f32`, and `final_norm.f32`
hold one row of hidden size per prompt token. `logits.f32` holds one
row of vocabulary size per prompt token. `generated.i32` holds the
greedy continuation. `manifest.json` records the shapes and versions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections.abc import Callable

import gguf
import numpy as np
import torch
import transformers
from torch import nn

DEFAULT_PROMPT = "The capital of France is"
# The names transformers gives the tensors of one LFM2 layer, by their GGUF names.
LAYER_TENSOR_NAMES = {
    "attn_norm.weight": "operator_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.out_proj.weight",
    "attn_q_norm.weight": "self_attn.q_layernorm.weight",
    "attn_k_norm.weight": "self_attn.k_layernorm.weight",
    "shortconv.conv.weight": "conv.conv.weight",
    "shortconv.in_proj.weight": "conv.in_proj.weight",
    "shortconv.out_proj.weight": "conv.out_proj.weight",
    "ffn_gate.weight": "feed_forward.w1.weight",
    "ffn_up.weight": "feed_forward.w3.weight",
    "ffn_down.weight": "feed_forward.w2.weight",
    "ffn_gate_inp.weight": "feed_forward.gate.weight",
    "exp_probs_b.bias": "feed_forward.expert_bias",
    "ffn_down_exps.weight": "feed_forward.experts.down_proj",
}
LAYER_TENSOR_PATTERN = re.compile(r"blk\.(?P<layer>\d+)\.(?P<name>.+)")
# The names transformers gives the tensors of one Qwen3.5 layer, by their GGUF names.
QWEN35_TENSOR_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}
# llama.cpp's converter adds 1 to these zero-centered norm weights.
QWEN35_CENTERED_NORMS = {
    "attn_norm.weight",
    "post_attention_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump reference activations for bobcat's tests.")
    parser.add_argument("--model", required=True, help="Hugging Face repo or path")
    parser.add_argument("--gguf", type=pathlib.Path, help="load the weights from this GGUF file")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--generate", type=int, default=16, help="greedy tokens to generate")
    return parser.parse_args()


def write_raw(path: pathlib.Path, array: np.ndarray) -> None:
    array.astype(array.dtype.newbyteorder("<")).tofile(path)


def gguf_architecture(path: pathlib.Path) -> str:
    reader = gguf.GGUFReader(path)
    architecture = reader.fields["general.architecture"].contents()
    if not isinstance(architecture, str):
        raise TypeError(f"{path} names no architecture")
    return architecture


def dequantized(tensor: gguf.ReaderTensor) -> torch.Tensor:
    """Return the float32 values of `tensor` in the shape transformers uses, rows first."""
    values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def load_moe_from_gguf(model_name: str, path: pathlib.Path) -> transformers.Lfm2MoeForCausalLM:
    """Return the LFM2 mixture-of-experts model of `model_name` with the weights in `path`."""
    config = transformers.Lfm2MoeConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = transformers.Lfm2MoeForCausalLM(config)

    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": dequantized(tensors.pop("token_embd.weight")),
        "model.embedding_norm.weight": dequantized(tensors.pop("token_embd_norm.weight")),
    }
    # transformers stacks the experts' gate and up matrices into one tensor, gate first.
    expert_halves: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        match = LAYER_TENSOR_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected tensor {name}")
        layer = int(match["layer"])
        suffix = match["name"]
        values = dequantized(tensor)
        if suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
            halves = expert_halves.setdefault(layer, {})
            halves[suffix] = values
            if len(halves) == 2:
                # Joining each pair as it completes keeps one copy of the expert weights.
                gate = halves.pop("ffn_gate_exps.weight")
                up = halves.pop("ffn_up_exps.weight")
                gate_up = torch.cat([gate, up], dim=1)
                state[f"model.layers.{layer}.feed_forward.experts.gate_up_proj"] = gate_up
            continue
        if suffix == "shortconv.conv.weight":
            # GGUF holds [hidden, taps], and transformers holds [hidden, 1, taps].
            values = values.unsqueeze(1)
        state[f"model.layers.{layer}.{LAYER_TENSOR_NAMES[suffix]}"] = values

    # The output matrix is tied to the embeddings, so it takes no entry of its own.
    model.load_state_dict(state, strict=False, assign=True)
    model.tie_weights()
    # The rotary frequencies are buffers that a model built on the meta device never computes.
    rotary = model.model.pos_emb
    model.model.pos_emb = type(rotary)(config=config)
    missing = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    missing += [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if missing:
        raise ValueError(f"the GGUF file lacks {missing}")
    return model


def load_qwen35_from_gguf(model_name: str, path: pathlib.Path) -> transformers.Qwen3_5ForCausalLM:
    """Return the Qwen3.5 text model of `model_name` with the weights in `path`.

    The script undoes the converter's changes: the +1 baked into the zero-centered norms and the
    -exp(A_log) stored as `ssm_a`. Models with as many value heads as key heads have no reordered
    heads, and the script rejects the others.
    """
    config = transformers.AutoConfig.from_pretrained(model_name)
    config = getattr(config, "text_config", config)
    if config.linear_num_value_heads != config.linear_num_key_heads:
        raise ValueError("the converter reorders value heads, which this loader does not undo")
    with torch.device("meta"):
        model = transformers.Qwen3_5ForCausalLM(config)

    reader = gguf.GGUFReader(path)
    n_layers = config.num_hidden_layers
    state: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        values = dequantized(tensor)
        if tensor.name == "token_embd.weight":
            state["model.embed_tokens.weight"] = values
            continue
        if tensor.name == "output_norm.weight":
            state["model.norm.weight"] = values - 1
            continue
        if tensor.name == "output.weight":
            state["lm_head.weight"] = values
            continue
        match = LAYER_TENSOR_PATTERN.fullmatch(tensor.name)
        if match is None:
            raise ValueError(f"unexpected tensor {tensor.name}")
        layer = int(match["layer"])
        suffix = match["name"]
        # The multi-token prediction block follows the model's layers.
        if layer >= n_layers:
            continue
        if suffix in QWEN35_CENTERED_NORMS:
            values = values - 1
        elif suffix == "ssm_a":
            values = torch.log(-values)
        elif suffix == "ssm_conv1d.weight":
            # GGUF holds [channels, taps], and transformers holds [channels, 1, taps].
            values = values.unsqueeze(1)
        state[f"model.layers.{layer}.{QWEN35_TENSOR_NAMES[suffix]}"] = values

    model.load_state_dict(state, strict=False, assign=True)
    if "lm_head.weight" not in state:
        model.tie_weights()
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=config)
    missing = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    missing += [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if missing:
        raise ValueError(f"the GGUF file lacks {missing}")
    return model


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer is None:
        raise RuntimeError(f"no tokenizer found for {args.model}")
    architecture = gguf_architecture(args.gguf) if args.gguf is not None else None
    if args.gguf is not None and architecture == "lfm2moe":
        model = load_moe_from_gguf(args.model, args.gguf)
    elif args.gguf is not None and architecture == "qwen35":
        model = load_qwen35_from_gguf(args.model, args.gguf)
    elif args.gguf is not None:
        gguf_path = args.gguf.resolve()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            gguf_path.parent,
            gguf_file=gguf_path.name,
            dtype=torch.float32,
            device_map="cpu",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, device_map="cpu"
        )
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    # The LFM2.5-8B-A1B tokenizer adds no BOS token of its own, and the model repeats itself
    # without one, so the prompt starts with BOS as the chat template's prompts do.
    # Qwen3.5 has no BOS token at all.
    if tokenizer.bos_token_id is not None and input_ids[0, 0] != tokenizer.bos_token_id:
        bos = torch.tensor([[tokenizer.bos_token_id]])
        input_ids = torch.cat([bos, input_ids], dim=1)
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str) -> Callable[[nn.Module, tuple, torch.Tensor | tuple], None]:
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            # Some decoder layers return a tuple whose first item is the hidden state.
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden.detach()

        return hook

    decoder = model.model
    # LFM2 names its final norm embedding_norm, and Qwen3.5 names it norm.
    final_norm = getattr(decoder, "embedding_norm", None) or decoder.norm
    if not isinstance(final_norm, nn.Module):
        raise TypeError("the model's final norm is not a module")
    handles = [
        decoder.embed_tokens.register_forward_hook(capture("embedding")),
        final_norm.register_forward_hook(capture("final_norm")),
    ]
    for index, layer in enumerate(decoder.layers):
        handles.append(layer.register_forward_hook(capture(f"layer_{index:02d}")))

    with torch.no_grad():
        logits = model(input_ids).logits
    for handle in handles:
        handle.remove()

    # transformers 5.17 declares a GenerativePreTrainedModel protocol that
    # its own model classes fail to satisfy, so ty rejects the call and its
    # first argument.
    with torch.no_grad():
        output_ids = model.generate(  # ty: ignore[invalid-argument-type]
            input_ids,  # ty: ignore[invalid-argument-type]
            do_sample=False,
            max_new_tokens=args.generate,
            min_new_tokens=args.generate,
        )
    generated = output_ids[0, input_ids.shape[1] :]

    write_raw(args.out / "tokens.i32", input_ids[0].numpy().astype(np.int32))
    write_raw(args.out / "generated.i32", generated.numpy().astype(np.int32))
    write_raw(args.out / "logits.f32", logits[0].numpy().astype(np.float32))
    for name, tensor in captured.items():
        write_raw(args.out / f"{name}.f32", tensor[0].numpy().astype(np.float32))

    manifest = {
        "model": args.model,
        "gguf": str(args.gguf) if args.gguf is not None else None,
        "prompt": args.prompt,
        "n_tokens": int(input_ids.shape[1]),
        "n_generated": int(generated.shape[0]),
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "n_layers": len(decoder.layers),
        "generated_text": tokenizer.decode(generated),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
