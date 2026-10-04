"""JSONL 文档到定长 uint16 序列；按来源 token 配额组成训练混合。"""

import argparse
from collections import deque
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from dummym.utils.files import sha256_file, write_json


def pack_file(
    path: Path, destination: Path, tokenizer: Tokenizer, tokens: int, sequence_length: int = 2048
) -> dict:
    target = tokens // sequence_length * sequence_length
    if target < sequence_length:
        raise ValueError("Each source/split needs at least one sequence")
    eos = tokenizer.token_to_id("</s>")
    if eos is None or tokenizer.get_vocab_size() > 65_536:
        raise ValueError("Packing requires EOS and a vocabulary fitting uint16")
    pending = deque()
    written, documents = 0, 0
    with path.open() as source, destination.open("wb") as stream:
        batch = []

        def consume(texts):
            nonlocal written, documents
            for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
                documents += 1
                pending.extend(encoding.ids)
                pending.append(eos)
                while len(pending) >= sequence_length and written < target:
                    row = [pending.popleft() for _ in range(sequence_length)]
                    stream.write(np.asarray(row, dtype="<u2").tobytes())
                    written += sequence_length
                if written == target:
                    return

        for line in source:
            batch.append(json.loads(line)["text"])
            if len(batch) == 256:
                consume(batch)
                batch.clear()
            if written == target:
                break
        if batch and written < target:
            consume(batch)
    if written != target:
        raise ValueError(f"Insufficient text in {path}: {written:,}/{target:,} tokens")
    return {
        "written_tokens": written,
        "sequences": written // sequence_length,
        "bytes": written * 2,
        "documents": documents,
    }


def prepare_mixture(
    clean: Path,
    tokenizer_path: Path,
    output: Path,
    train_tokens: int,
    validation_tokens: int,
    weights: dict[str, float],
) -> None:
    if json.loads((clean / "clean.json").read_text())["status"] != "complete":
        raise ValueError("Cleaning must complete before packing")
    if (
        not weights
        or any(w <= 0 for w in weights.values())
        or abs(sum(weights.values()) - 1) > 1e-6
    ):
        raise ValueError("Positive source weights must sum to one")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    shutil.copyfile(tokenizer_path, output / "tokenizer.json")
    sources, splits = {}, {}
    for split, budget in (
        ("train", train_tokens),
        ("validation", validation_tokens),
        ("test", validation_tokens),
    ):
        combined = output / f"{split}.bin"
        with combined.open("wb") as stream:
            for source, weight in weights.items():
                temporary = output / f"{source}.{split}.bin"
                stats = pack_file(
                    clean / f"{source}.{split}.jsonl", temporary, tokenizer, int(budget * weight)
                )
                sources[f"{source}.{split}"] = stats
                with temporary.open("rb") as part:
                    shutil.copyfileobj(part, stream, length=8 * 1024 * 1024)
                temporary.unlink()
        count = combined.stat().st_size // 2
        splits[split] = {
            "written_tokens": count,
            "sequences": count // 2048,
            "bytes": count * 2,
            "sha256": sha256_file(combined),
        }
    write_json(
        output / "summary.json",
        {
            "status": "complete",
            "format": {"dtype": "<u2", "sequence_length": 2048},
            "tokenizer": {
                "file": "tokenizer.json",
                "sha256": sha256_file(output / "tokenizer.json"),
                "vocab_size": tokenizer.get_vocab_size(),
                "eos_token_id": tokenizer.token_to_id("</s>"),
            },
            "splits": splits,
            "sources": sources,
            "weights": weights,
            "clean_manifest_sha256": sha256_file(clean / "clean.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--validation-tokens", type=int, default=10_000_000)
    parser.add_argument("--weights", default="fineweb=0.7,cosmopedia=0.25,tinystories=0.05")
    args = parser.parse_args()
    weights = {
        name: float(weight)
        for name, weight in (value.split("=", 1) for value in args.weights.split(","))
    }
    prepare_mixture(
        args.clean, args.tokenizer, args.output, args.train_tokens, args.validation_tokens, weights
    )
