#!/usr/bin/env python3
"""Base 模型的纯文本续写；不用聊天模板，也不将续写成功当作聊天验收。"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dummym.utils.files import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("Choose between 1 and 256 new tokens")
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    prompts = [
        "The sun rises in the",
        "Once upon a time, there was a little",
        "Water is important because",
        "To make a cup of tea, first",
    ]
    samples = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, encoded.input_ids.shape[1] :]
        samples.append(
            {
                "prompt": prompt,
                "continuation": tokenizer.decode(continuation, skip_special_tokens=True),
                "new_tokens": len(continuation),
            }
        )
    write_json(
        args.output,
        {
            "model": str(args.model),
            "sampling": "greedy",
            "samples": samples,
            "note": "Completion smoke test; does not establish assistant capability",
        },
    )


if __name__ == "__main__":
    main()
