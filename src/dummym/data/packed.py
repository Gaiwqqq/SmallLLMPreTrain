"""已校验的 uint16 数据映射；batch 复制后转换为模型所需的 int64。"""

import json
from pathlib import Path

import numpy as np
import torch

from dummym.models.llama_like import MiniLlamaConfig
from dummym.utils.files import sha256_file


class PackedRows:
    def __init__(self, path: Path, rows: int, length: int, vocab_size: int, in_memory: bool):
        if rows <= 0 or length < 2 or path.stat().st_size != rows * length * 2:
            raise ValueError(f"Invalid packed data shape/size: {path}")
        self.rows = np.memmap(path, dtype="<u2", mode="r").reshape(rows, length)
        if int(self.rows.max()) >= vocab_size:
            raise ValueError(f"Token ID exceeds vocabulary: {path}")
        if in_memory:
            self.rows = np.array(self.rows)
        self.sequence_length = length

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, indices, device: torch.device) -> torch.Tensor:
        array = self.rows[indices].astype(np.int64)
        batch = torch.from_numpy(array)
        if device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(device, non_blocking=device.type == "cuda")


def load_packed(directory: Path, config: MiniLlamaConfig, in_memory: bool = False):
    from tokenizers import Tokenizer

    manifest = json.loads((directory / "summary.json").read_text())
    if manifest["status"] != "complete" or manifest["format"]["dtype"] != "<u2":
        raise ValueError("Data preparation has not completed")
    length = manifest["format"]["sequence_length"]
    if length > config.max_position_embeddings:
        raise ValueError("Packed sequence exceeds model context")
    tokenizer_path = directory / manifest["tokenizer"]["file"]
    tokenizer_digest = sha256_file(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer_digest != manifest["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer fingerprint mismatch")
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    if tokenizer.token_to_id("</s>") != config.eos_token_id:
        raise ValueError("Tokenizer/model EOS mismatch")
    if tokenizer.token_to_id("<s>") != config.bos_token_id:
        raise ValueError("Tokenizer/model BOS mismatch")
    datasets, fingerprint = {}, {"tokenizer": tokenizer_digest, "sequence_length": length}
    for split in ("train", "validation"):
        stats = manifest["splits"][split]
        if stats["written_tokens"] != stats["sequences"] * length:
            raise ValueError("Inconsistent token/sequence count")
        path = directory / f"{split}.bin"
        fingerprint[split] = sha256_file(path)
        if "sha256" in stats and stats["sha256"] != fingerprint[split]:
            raise ValueError(f"Data fingerprint mismatch: {split}")
        datasets[split] = PackedRows(path, stats["sequences"], length, config.vocab_size, in_memory)
    return datasets, fingerprint, tokenizer_path
