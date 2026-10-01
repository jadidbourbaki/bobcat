"""Join Cloudflare's Clef joint schema head to its converted backbone in one GGUF file.

Usage:

    uv run python clef_gguf.py --backbone ../models/clef-flash-backbone-Q8_0.gguf \
        --release ../models/hf/clef-flash --out ../models/clef-flash-Q8_0.gguf

llama.cpp's converter writes the Qwen3.5 backbone of a Clef release. This script copies every key
and tensor of that file, then adds the head's configuration as `clef.*` keys and its tensors as
`clef.` followed by their names in `joint_head.safetensors`. The head's matrices stay bf16, and
its vectors and scalars become F32, which bobcat reads as vectors.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import gguf
import numpy as np
import torch
from pydantic import BaseModel, PositiveInt
from safetensors.torch import load_file


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


def main() -> None:
    args = parse_args()
    config = HeadConfig.model_validate_json((args.release / "joint_head_config.json").read_text())
    reader = gguf.GGUFReader(args.backbone)
    architecture = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    if not isinstance(architecture, str):
        raise TypeError("the backbone names no architecture")
    writer = gguf.GGUFWriter(args.out, architecture)

    for field in reader.fields.values():
        # The writer adds the architecture and the GGUF header fields itself.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)
    for key, value in json.loads(config.model_dump_json()).items():
        writer.add_uint32(f"clef.{key}", value)

    head = head_arrays(args.release)
    for tensor in reader.tensors:
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
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for data, _ in head.values():
        writer.write_tensor_data(data)
    writer.close()
    print(f"wrote {args.out} with {len(reader.tensors)} backbone and {len(head)} head tensors")


if __name__ == "__main__":
    main()
