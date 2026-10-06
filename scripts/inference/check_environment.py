#!/usr/bin/env python3
"""在分配 GPU 前核对推理依赖，并验证 vLLM 实际使用的 Tokenizer API。"""

import argparse
from importlib import metadata
import json
from pathlib import Path


def expected_versions(requirements: Path) -> dict[str, str]:
    return {
        line.split("==", 1)[0]: line.split("==", 1)[1]
        for raw in requirements.read_text().splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    }


def validate_versions(expected: dict[str, str]) -> dict[str, str]:
    actual = {name: metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Inference dependencies differ from the pinned recipe: {actual}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requirements = Path(__file__).resolve().parents[2] / "requirements/inference.txt"
    actual = validate_versions(expected_versions(requirements))
    from transformers import AutoTokenizer
    from vllm.transformers_utils.tokenizer import get_cached_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        raise RuntimeError("Tokenizer lacks the API required by vLLM 0.11.0")
    cached = get_cached_tokenizer(tokenizer)
    text = "Hello, world! café 🙂"
    if cached.encode(text) != tokenizer.encode(text):
        raise RuntimeError("vLLM tokenizer cache changes text encoding")
    if cached.all_special_ids != tokenizer.all_special_ids:
        raise RuntimeError("vLLM tokenizer cache changes special-token IDs")
    if cached.convert_tokens_to_ids("<|im_end|>") != 5 or tokenizer.eos_token_id != 2:
        raise RuntimeError("Tokenizer differs from this project's EOS/ChatML contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "versions": actual,
                "tokenizer_class": type(tokenizer).__name__,
                "cached_tokenizer_class": type(cached).__name__,
                "scope": "Tokenizer compatibility only; GPU inference and semantics require separate checks",
            },
            indent=2,
        )
        + "\n"
    )
    print("Inference dependency and tokenizer preflight passed", flush=True)


if __name__ == "__main__":
    main()
