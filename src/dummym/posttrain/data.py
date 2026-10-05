"""SFT 数据准备在单进程 CPU 阶段完成，训练 rank 只读取已就绪的 Arrow 数据。"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from datasets import load_dataset, load_from_disk

from dummym.utils.files import sha256_file, write_json

MAX_LENGTH = 2048


def data_contract(data_dir: Path, model_dir: Path, tokenizer) -> dict:
    return {
        "format_version": 1,
        "source_manifest_sha256": sha256_file(data_dir / "download.json"),
        "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
        "chat_template": tokenizer.chat_template,
        "max_length": MAX_LENGTH,
        "split_seed": 2026,
        "validation_fraction": 0.01,
        "deduplication": "whole-conversation SHA256, first occurrence retained",
    }


def load_conversations(data_dir: Path):
    manifest = json.loads((data_dir / "download.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete SFT download")
    files = [
        str(data_dir / item["path"]) for item in manifest["files"] if "/train-" in item["path"]
    ]
    if not files:
        raise ValueError("No SFT training files in the manifest")
    seen = set()

    def unique(example):
        digest = hashlib.sha256(
            json.dumps(example["messages"], sort_keys=True).encode()
        ).hexdigest()
        if digest in seen:
            return False
        seen.add(digest)
        return True

    dataset = load_dataset("parquet", data_files=files, split="train")
    return dataset.filter(unique, load_from_cache_file=True), manifest


def encode_conversation(example, tokenizer, max_length: int = MAX_LENGTH) -> dict:
    # 与 TRL 0.24 的 conversational 路径使用相同模板；显式截断避免晚于同步屏障。
    encoded = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        tools=example.get("tools"),
        **(example.get("chat_template_kwargs") or {}),
    )
    ids = encoded["input_ids"][:max_length]
    mask = encoded["assistant_masks"][:max_length]
    if len(ids) != len(mask):
        raise ValueError("Assistant mask and input length differ")
    return {"input_ids": ids, "assistant_masks": mask}


def has_targets(example) -> bool:
    # 第一位置没有前一个 token 的预测，不能作为有效的下一 token 监督。
    return any(example["assistant_masks"][1:])


def prepare_data(data_dir: Path, model_dir: Path, output: Path, tokenizer, workers: int) -> None:
    contract = data_contract(data_dir, model_dir, tokenizer)
    if output.exists():
        load_prepared(output, contract)
        print(f"Prepared SFT data already verified: {output}", flush=True)
        return
    dataset, manifest = load_conversations(data_dir)
    unique_count = len(dataset)
    # 保留 messages 供 TRL 判定 conversational；collator 仅读取 token 和 mask。
    dataset = dataset.map(
        encode_conversation,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=workers,
        desc="Encode and truncate SFT conversations on CPU",
    )
    dataset = dataset.filter(has_targets, num_proc=workers)
    split = dataset.train_test_split(test_size=0.01, seed=2026)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        split.save_to_disk(str(temporary))
        write_json(
            temporary / "ready.json",
            {
                "contract": contract,
                "source_revision": manifest["revision"],
                "unique_conversations": unique_count,
                "filtered_no_targets": unique_count - len(dataset),
                "train_conversations": len(split["train"]),
                "validation_conversations": len(split["test"]),
                "files": {
                    path.relative_to(temporary).as_posix(): sha256_file(path)
                    for path in sorted(temporary.rglob("*"))
                    if path.is_file()
                },
            },
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(
        f"Prepared SFT data: {len(split['train'])} train / {len(split['test'])} validation",
        flush=True,
    )


def load_prepared(output: Path, contract: dict):
    ready = json.loads((output / "ready.json").read_text())
    if ready["contract"] != contract:
        raise ValueError("Prepared SFT data/tokenizer/recipe mismatch")
    for name, digest in ready["files"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"Prepared SFT data checksum mismatch: {name}")
    split = load_from_disk(str(output))
    if any(
        len(split[key]) != ready[count]
        for key, count in [("train", "train_conversations"), ("test", "validation_conversations")]
    ):
        raise ValueError("Prepared SFT split counts differ")
    return split
