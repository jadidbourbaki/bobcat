"""Dump a reference decision from Cloudflare's Clef code for bobcat's tests.

Usage:

    uv run python clef_dump.py --release ../models/hf/clef-flash \
        --out ../models/ref/clef-flash

The script loads a Clef release with the release's own `joint_schema_model.py`, encodes one
SystemOne request, and runs the backbone in bfloat16 on the CPU. It then runs the joint schema
head in float32 on the backbone's hidden states, so a float32 head given the same states matches
it to rounding. It writes raw little-endian files to the output directory:

- `tokens.i32` holds the request's token ids.
- `hidden.f32` holds the backbone's normalized last hidden states, one row per token.
- `request.json` holds the request.
- `encoded.json` holds each question's kind, its instructions span, and its options' ids and spans.
- `logits.json` holds the head's float32 logits per question.
- `answers.json` holds the SystemOne answers of the release's own `systemone`, from its bfloat16
  head.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import types

import numpy as np
import torch
import transformers
from safetensors.torch import load_file

from clef_gguf import HeadConfig

REQUEST = {
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a reference Clef decision.")
    parser.add_argument("--release", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    return parser.parse_args()


def write_raw(path: pathlib.Path, array: np.ndarray) -> None:
    array.astype(array.dtype.newbyteorder("<")).tofile(path)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # The release ships its model code next to its weights.
    sys.path.insert(0, str(args.release.resolve()))
    clef = importlib.import_module("joint_schema_model")
    # The release's loader also builds the image and video processor, which needs torchvision.
    # Text requests need only the tokenizer, so the script loads the pieces itself.
    backbone = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        args.release, dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    head_config = HeadConfig.model_validate_json(
        (args.release / "joint_head_config.json").read_text()
    )
    joint_head = clef.JointSchemaHead(**head_config.model_dump())
    joint_head.load_state_dict(load_file(args.release / "joint_head.safetensors"), strict=True)
    model = clef.ClefModel(backbone, joint_head.to(torch.bfloat16)).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.release)
    if tokenizer is None:
        raise RuntimeError(f"no tokenizer found in {args.release}")
    processor = types.SimpleNamespace(tokenizer=tokenizer)

    encoded = clef.encode_record(tokenizer, REQUEST)
    batch = clef.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
    text_model = model.language_model.model.language_model
    with torch.inference_mode():
        hidden = text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.float()
        answers = clef.systemone(model, processor, REQUEST)["answers"]
        head = model.head.float()
        output_embedding = model.language_model.get_output_embeddings().weight.float()
        logits = head(
            hidden, batch["input_ids"], batch["attention_mask"], batch["records"], output_embedding
        )[0]

    write_raw(args.out / "tokens.i32", np.asarray(encoded.input_ids, dtype=np.int32))
    write_raw(args.out / "hidden.f32", hidden[0].numpy().astype(np.float32))
    (args.out / "request.json").write_text(json.dumps(REQUEST, indent=2) + "\n")
    questions = [
        {
            "id": question.question_id,
            "type": question.question_type,
            "span": list(question.question_span),
            "option_ids": list(question.option_ids),
            "option_spans": [list(span) for span in question.option_spans],
        }
        for question in encoded.questions
    ]
    (args.out / "encoded.json").write_text(json.dumps(questions, indent=2) + "\n")
    (args.out / "logits.json").write_text(
        json.dumps([question_logits.tolist() for question_logits in logits], indent=2) + "\n"
    )
    (args.out / "answers.json").write_text(json.dumps(answers, indent=2) + "\n")
    print(json.dumps({"tokens": len(encoded.input_ids), "answers": answers}, indent=2))


if __name__ == "__main__":
    main()
