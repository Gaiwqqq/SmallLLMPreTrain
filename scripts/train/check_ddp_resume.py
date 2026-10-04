#!/usr/bin/env python3
"""四卡 BF16、dropout 与不均匀尾部的完整训练/中断恢复对照。"""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from tokenizers import Tokenizer, models
import yaml

from dummym.models.llama_like import MiniLlamaConfig
from dummym.utils.files import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        return
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.output / "data"
    data.mkdir(exist_ok=True)
    vocabulary = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"word{i}": i for i in range(3, 64)}}
    Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>")).save(str(data / "tokenizer.json"))
    stats = {}
    for split, rows in (("train", 13), ("validation", 7)):
        (np.arange(rows * 16, dtype=np.uint16) % 61 + 3).astype("<u2").tofile(data / f"{split}.bin")
        stats[split] = {"written_tokens": rows * 16, "sequences": rows, "bytes": rows * 32}
    write_json(
        data / "summary.json",
        {
            "status": "complete",
            "format": {"dtype": "<u2", "sequence_length": 16},
            "tokenizer": {"file": "tokenizer.json", "sha256": sha256_file(data / "tokenizer.json")},
            "splits": stats,
        },
    )
    config = MiniLlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.1,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    model_config = args.output / "model.yaml"
    model_config.write_text(yaml.safe_dump({"model_config": config.to_dict()}))
    executable = Path(sys.executable).parent / "dummym-pretrain"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "--no-python",
        str(executable),
        "--model-config",
        str(model_config),
        "--data-dir",
        str(data),
        "--total-tokens",
        "208",
        "--global-batch-size",
        "8",
        "--micro-batch-size",
        "1",
        "--warmup-ratio",
        "0.5",
    ]
    full, resumed = args.output / "full", args.output / "resumed"
    subprocess.run(command + ["--output-dir", str(full)], check=True)
    subprocess.run(command + ["--output-dir", str(resumed), "--stop-after-steps", "1"], check=True)
    subprocess.run(
        command + ["--output-dir", str(resumed), "--resume", str(resumed / "checkpoint.pt")],
        check=True,
    )
    left = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
    right = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
    difference = 0.0
    for name, value in left["model_state_dict"].items():
        torch.testing.assert_close(value, right["model_state_dict"][name], atol=5e-6, rtol=5e-5)
        difference = max(difference, (value - right["model_state_dict"][name]).abs().max().item())
    for a, b in zip(left["rng_states"], right["rng_states"]):
        torch.testing.assert_close(a["cpu"], b["cpu"], atol=0, rtol=0)
        torch.testing.assert_close(a["cuda"], b["cuda"], atol=0, rtol=0)
    if left["scheduler_state_dict"] != right["scheduler_state_dict"]:
        raise AssertionError("Scheduler mismatch after resume")
    write_json(
        args.output / "summary.json",
        {
            "status": "passed",
            "world_size": 4,
            "precision": "bf16",
            "dropout": 0.1,
            "train_rows": 13,
            "validation_rows": 7,
            "max_parameter_difference": difference,
            "rng_states_match": True,
        },
    )


if __name__ == "__main__":
    main()
