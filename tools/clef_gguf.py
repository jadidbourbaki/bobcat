"""Join Cloudflare's Clef joint schema head to its converted backbone in one GGUF file.

Usage:

    uv run python clef_gguf.py --backbone ../models/clef-flash-backbone-Q8_0.gguf \
        --release ../models/hf/clef-flash --out ../models/clef-flash-Q8_0.gguf

llama.cpp's converter writes the Qwen3.5 backbone of a Clef release. This script copies every key
and tensor of that file, then adds the head's configuration as `clef.*` keys and its tensors as
`clef.` followed by their names in `joint_head.safetensors`. The head's matrices stay bf16, and
its vectors and scalars become F32, which bobcat reads as vectors. A backbone that already holds a
head, such as an earlier output of this script, loses that head first.

The converter names Qwen3.5's pre-tokenizer for every Qwen3.5 backbone, while Clef releases
tokenize with Qwen2's. The script sets `tokenizer.ggml.pre` from the release's `tokenizer.json`.
"""

from __future__ import annotations

import argparse
import pathlib

import gguf
import numpy as np
import torch
from pydantic import BaseModel, PositiveInt
from safetensors.torch import load_file

# The pre-tokenizer regexes of the Qwen tokenizers, by their GGUF names.
PRE_TOKENIZERS = {
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+": "qwen2",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "qwen35",
}


class HeadConfig(BaseModel):
    """The shape of a Clef joint schema head, from `joint_head_config.json`."""

    hidden_size: PositiveInt
    width: PositiveInt
    routing_layers: PositiveInt
    layers: PositiveInt
    heads: PositiveInt
    feedforward: PositiveInt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join a Clef head to its GGUF backbone.")
    parser.add_argument("--backbone", required=True, type=pathlib.Path)
    parser.add_argument("--release", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    return parser.parse_args()


def head_arrays(release: pathlib.Path) -> dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType]]:
    """Return each head tensor's data and GGUF type, by its GGUF name."""
    arrays: dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType]] = {}
    for name, tensor in sorted(load_file(release / "joint_head.safetensors").items()):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} is {tensor.dtype}, and the script expects bfloat16")
        if tensor.dim() == 2:
            # The bits of bfloat16 values go into the file unchanged.
            bits = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
            arrays[f"clef.{name}"] = (bits, gguf.GGMLQuantizationType.BF16)
        else:
            values = tensor.float().reshape(-1).numpy()
            arrays[f"clef.{name}"] = (values, gguf.GGMLQuantizationType.F32)
    return arrays


class SplitPattern(BaseModel):
    """The regex of a `Split` pre-tokenizer step."""

    Regex: str


class PreTokenizerStep(BaseModel):
    """One step of a `Sequence` pre-tokenizer. Only `Split` steps carry a pattern."""

    type: str
    pattern: SplitPattern | None = None


class PreTokenizer(BaseModel):
    """The `Sequence` pre-tokenizer of a Hugging Face `tokenizer.json`."""

    pretokenizers: list[PreTokenizerStep]


class TokenizerJson(BaseModel):
    """The part of a Hugging Face `tokenizer.json` that names its pre-tokenizer."""

    pre_tokenizer: PreTokenizer


def pre_tokenizer(release: pathlib.Path) -> str:
    """Return the GGUF name of the pre-tokenizer in the release's `tokenizer.json`."""
    tokenizer = TokenizerJson.model_validate_json((release / "tokenizer.json").read_text())
    for step in tokenizer.pre_tokenizer.pretokenizers:
        if step.type == "Split" and step.pattern is not None:
            pattern = step.pattern.Regex
            if pattern not in PRE_TOKENIZERS:
                raise ValueError(f"the release splits text with an unknown regex {pattern!r}")
            return PRE_TOKENIZERS[pattern]
    raise ValueError("the release's tokenizer.json has no Split pre-tokenizer")


def main() -> None:
    args = parse_args()
    config = HeadConfig.model_validate_json((args.release / "joint_head_config.json").read_text())
    reader = gguf.GGUFReader(args.backbone)
    architecture = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    if not isinstance(architecture, str):
        raise TypeError("the backbone names no architecture")
    writer = gguf.GGUFWriter(args.out, architecture)

    for field in reader.fields.values():
        # The writer adds the architecture and the GGUF header fields itself, and the script
        # writes the pre-tokenizer and the head's keys below.
        if (
            field.name == gguf.Keys.General.ARCHITECTURE
            or field.name.startswith("GGUF.")
            or field.name.startswith("clef.")
            or field.name == gguf.Keys.Tokenizer.PRE
        ):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)
    writer.add_string(gguf.Keys.Tokenizer.PRE, pre_tokenizer(args.release))
    for key, value in config.model_dump().items():
        writer.add_uint32(f"clef.{key}", value)

    head = head_arrays(args.release)
    backbone = [tensor for tensor in reader.tensors if not tensor.name.startswith("clef.")]
    for tensor in backbone:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
    for name, (data, tensor_type) in head.items():
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in backbone:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for data, _ in head.values():
        writer.write_tensor_data(data)
    writer.close()
    print(f"wrote {args.out} with {len(backbone)} backbone and {len(head)} head tensors")


if __name__ == "__main__":
    main()
