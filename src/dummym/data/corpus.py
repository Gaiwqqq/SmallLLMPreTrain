"""清洗后先去重再划分文档；不同来源共享同一去重范围。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from datasketch import MinHash, MinHashLSH
import pyarrow.parquet as pq

from dummym.utils.files import write_json


def normalize(text: str) -> str | None:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = re.sub(r"[^\S\n]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not 200 <= len(text) <= 100_000:
        return None
    words = text.lower().split()
    if len(words) < 40 or len(set(words)) / len(words) < 0.1:
        return None
    return text


def document_split(digest: str, seed: int = 2026) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:8], 16) % 1000
    return "test" if bucket < 5 else "validation" if bucket < 10 else "train"


def iter_texts(directory: Path):
    manifest = json.loads((directory / "download.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete download: {directory}")
    for record in manifest["files"]:
        path = directory / record["path"]
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Changed raw file: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if isinstance(text, str):
                    yield text


def clean_corpus(sources: dict[str, Path], output: Path, max_documents: int = 0) -> None:
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    near = MinHashLSH(threshold=0.85, num_perm=32)
    counts = Counter()
    metadata = {}
    try:
        for source, directory in sources.items():
            metadata[source] = json.loads((directory / "download.json").read_text())
            handles = {
                split: (output / f"{source}.{split}.jsonl").open("w")
                for split in ("train", "validation", "test")
            }
            try:
                for raw in iter_texts(directory):
                    counts[f"{source}.scanned"] += 1
                    text = normalize(raw)
                    if text is None:
                        counts[f"{source}.filtered"] += 1
                        continue
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest in seen:
                        counts[f"{source}.exact_duplicate"] += 1
                        continue
                    seen.add(digest)
                    words = text.lower().split()
                    # 均匀选取至多 256 个五词 shingle，控制 Python 去重的计算量。
                    stride = max(1, (len(words) - 4) // 256)
                    signature = MinHash(num_perm=32)
                    signature.update_batch(
                        [
                            " ".join(words[i : i + 5]).encode()
                            for i in range(0, len(words) - 4, stride)
                        ]
                    )
                    if near.query(signature):
                        counts[f"{source}.near_duplicate"] += 1
                        continue
                    near.insert(digest, signature)
                    split = document_split(digest)
                    handles[split].write(json.dumps({"sha256": digest, "text": text}) + "\n")
                    counts[f"{source}.{split}"] += 1
                    if counts[f"{source}.scanned"] % 10_000 == 0:
                        print(json.dumps(dict(counts)), flush=True)
                    accepted = sum(counts[f"{source}.{s}"] for s in handles)
                    if max_documents and accepted >= max_documents:
                        break
            finally:
                for handle in handles.values():
                    handle.close()
        write_json(
            output / "clean.json",
            {
                "status": "complete",
                "sources": metadata,
                "counts": dict(counts),
                "split": "sha256(seed:normalized_text_digest), train/val/test=99/0.5/0.5%",
                "near_dedup": "MinHashLSH 32 permutations, threshold .85, sampled 5-word shingles",
                "limitations": "Approximate near-dedup may miss duplicates or reject distinct documents",
            },
        )
    except BaseException:
        write_json(output / "clean.json", {"status": "failed", "counts": dict(counts)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME=/path/to/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=0, help="每个来源的接收上限，0 不限")
    args = parser.parse_args()
    if args.max_documents < 0:
        parser.error("--max-documents must be nonnegative")
    sources = {name: Path(path) for name, path in (value.split("=", 1) for value in args.source)}
    clean_corpus(sources, args.output, args.max_documents)
