#!/usr/bin/env python3
"""隔离推理环境的离线冒烟测试；不将 Base 模型冒充聊天模型。"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
    )
    results = model.chat(
        [[{"role": "user", "content": "Hello. Introduce yourself briefly."}]],
        SamplingParams(
            temperature=0,
            max_tokens=64,
            stop_token_ids=[model.get_tokenizer().convert_tokens_to_ids("<|im_end|>")],
        ),
    )
    answer = results[0].outputs[0].text
    if not answer.strip():
        raise RuntimeError("vLLM returned an empty answer")
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "answer": answer,
                "note": "Inference smoke only, not semantic acceptance",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
