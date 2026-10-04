#!/usr/bin/env python3
"""检查真实 pilot 的文档隔离、Tokenizer 往返与 assistant 监督边界。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from dummym.data.corpus import document_split
from dummym.posttrain.sft import assert_assistant_mask
from dummym.utils.files import sha256_file, write_json


def audit(clean: Path, tokenizer_path: Path, output: Path) -> None:
    if json.loads((clean / "clean.json").read_text())["status"] != "complete":
        raise ValueError("Only audit complete clean corpora")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    assert_assistant_mask(tokenizer)
    samples = [
        "Hello, world!\n\nA second paragraph.",
        "中文、café、naïve、🙂 and mathematical symbols: ∑ ≤ π.",
        " spaces\tand\nnewlines  stay intact ",
    ]
    for sample in samples:
        ids = tokenizer.encode(sample, add_special_tokens=False)
        if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != sample:
            raise ValueError("Tokenizer does not preserve the original text")
    expected = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]
    if [tokenizer.convert_tokens_to_ids(token) for token in expected] != list(range(6)):
        raise ValueError("Special-token IDs differ from the model contract")
    seen, counts = set(), Counter()
    for path in sorted(clean.glob("*.jsonl")):
        source, split, _ = path.name.rsplit(".", 2)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                digest = hashlib.sha256(row["text"].encode()).hexdigest()
                if row["sha256"] != digest or document_split(digest) != split:
                    raise ValueError(f"Incorrect document fingerprint/split: {path}")
                if digest in seen:
                    raise ValueError(f"Duplicate document across source/split: {path}")
                seen.add(digest)
                counts[f"{source}.{split}"] += 1
    write_json(
        output,
        {
            "status": "passed",
            "documents": len(seen),
            "counts": dict(counts),
            "roundtrip_samples": len(samples),
            "assistant_mask": "passed",
            "special_token_ids": list(range(6)),
            "clean_manifest_sha256": sha256_file(clean / "clean.json"),
            "tokenizer_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
            "scope": "Exact-document isolation; does not establish semantic deduplication or quality",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.clean, args.tokenizer, args.output)


if __name__ == "__main__":
    main()
