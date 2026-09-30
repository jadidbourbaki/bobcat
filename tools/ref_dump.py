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
The script runs
the model in float32 on the CPU and writes raw
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
    field = reader.fields["general.architecture"]
    return field.contents()


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


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer is None:
        raise RuntimeError(f"no tokenizer found for {args.model}")
    if args.gguf is not None and gguf_architecture(args.gguf) == "lfm2moe":
        model = load_moe_from_gguf(args.model, args.gguf)
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
    bos = torch.tensor([[tokenizer.bos_token_id]])
    if input_ids[0, 0] != tokenizer.bos_token_id:
        input_ids = torch.cat([bos, input_ids], dim=1)
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            captured[name] = output.detach()

        return hook

    decoder = model.model
    handles = [
        decoder.embed_tokens.register_forward_hook(capture("embedding")),
        decoder.embedding_norm.register_forward_hook(capture("final_norm")),
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
