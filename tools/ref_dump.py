"""Dump reference activations from transformers for gip's tests.

Usage:

    uv run python ref_dump.py --model LiquidAI/LFM2.5-350M \
        --out ../models/ref/LFM2.5-350M

    uv run python ref_dump.py --model LiquidAI/LFM2.5-350M \
        --gguf ../models/LFM2.5-350M-Q8_0.gguf \
        --out ../models/ref/LFM2.5-350M-Q8_0

With `--gguf`, the weights come from the GGUF file, dequantized to
float32, and the tokenizer still comes from `--model`. The script runs
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

import numpy as np
import torch
import transformers
from torch import nn

DEFAULT_PROMPT = "The capital of France is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump reference activations for gip's tests."
    )
    parser.add_argument("--model", required=True, help="Hugging Face repo or path")
    parser.add_argument(
        "--gguf", type=pathlib.Path, help="load the weights from this GGUF file"
    )
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--generate", type=int, default=16, help="greedy tokens to generate"
    )
    return parser.parse_args()


def write_raw(path: pathlib.Path, array: np.ndarray) -> None:
    array.astype(array.dtype.newbyteorder("<")).tofile(path)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer is None:
        raise RuntimeError(f"no tokenizer found for {args.model}")
    if args.gguf is not None:
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
    # its own auto model classes fail to satisfy, so ty rejects the call.
    with torch.no_grad():
        output_ids = model.generate(  # ty: ignore[invalid-argument-type]
            input_ids,
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
