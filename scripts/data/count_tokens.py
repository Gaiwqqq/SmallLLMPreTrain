#!/usr/bin/env python3
"""精确测量清洗语料的 token 容量，正式预算不得超过各来源实际容量。"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from dummym.utils.files import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--only-validation", action="store_true", help="只测量验证和测试容量，适用于 pilot"
    )
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    counts = {}
    splits = ("validation", "test") if args.only_validation else ("train", "validation", "test")
    for split in splits:
        count, documents, batch = 0, 0, []
        with (args.clean / f"{args.source}.{split}.jsonl").open() as stream:
            for line in stream:
                batch.append(json.loads(line)["text"])
                if len(batch) == 256:
                    count += sum(
                        len(item.ids) + 1
                        for item in tokenizer.encode_batch(batch, add_special_tokens=False)
                    )
                    documents += len(batch)
                    batch.clear()
            if batch:
                count += sum(
                    len(item.ids) + 1
                    for item in tokenizer.encode_batch(batch, add_special_tokens=False)
                )
                documents += len(batch)
        counts[split] = {"tokens": count, "documents": documents}
        print(f"source={args.source} split={split} tokens={count:,}", flush=True)
    write_json(args.output, counts)


if __name__ == "__main__":
    main()
