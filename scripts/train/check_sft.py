#!/usr/bin/env python3
"""在极小合成会话上执行真实 TRL SFT，验证模板、保存和四卡训练接口。"""

import argparse
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from dummym.checkpoint.huggingface import to_huggingface
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.tokenizer.train import CHAT_TEMPLATE, SPECIAL_TOKENS
from dummym.utils.files import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        return
    args.output.mkdir(parents=True, exist_ok=False)
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in ["Hello", "Good", "morning", "user", "assistant", "friend"]:
        vocabulary[token] = len(vocabulary)
    for index in range(len(vocabulary), 128):
        vocabulary[f"word{index}"] = index
    core = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
        chat_template=CHAT_TEMPLATE,
    )
    config = MiniLlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    model_dir = args.output / "model"
    to_huggingface(MiniLlamaForCausalLM(config)).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    data = args.output / "data"
    data.mkdir()
    conversations = [
        {
            "messages": [
                {"role": "user", "content": f"Hello word{index}"},
                {"role": "assistant", "content": "Good morning friend"},
            ]
        }
        for index in range(16, 48)
    ]
    pq.write_table(pa.Table.from_pylist(conversations), data / "train-0.parquet")
    write_json(
        data / "download.json",
        {
            "status": "complete",
            "revision": "synthetic",
            "files": [{"path": "data/train-0.parquet"}],
        },
    )
    # CLI 使用标准来源的 data/train-* 形状。
    (data / "data").mkdir()
    (data / "train-0.parquet").rename(data / "data/train-0.parquet")
    executable = Path(sys.executable).parent / "dummym-sft"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "--no-python",
            str(executable),
            "--model",
            str(model_dir),
            "--data-dir",
            str(data),
            "--output",
            str(args.output / "trained"),
            "--epochs",
            "1",
        ],
        check=True,
    )
    if not (args.output / "trained/final/model.safetensors").exists():
        raise RuntimeError("SFT export is missing")
    snapshots = list((args.output / "trained/trainer-milestones").glob("*/ready.json"))
    if len(snapshots) != 1:
        raise RuntimeError("SFT complete recovery snapshot is missing")
    required = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "model.safetensors"]
    if any(not (snapshots[0].parent / name).exists() for name in required):
        raise RuntimeError("SFT snapshot does not contain complete Trainer state")
    if len(list(snapshots[0].parent.glob("rng_state_*.pth"))) != 4:
        raise RuntimeError("SFT snapshot does not contain all four rank RNG states")
    write_json(
        args.output / "summary.json",
        {"status": "passed", "world_size": 4, "conversations": 32, "synthetic_data": True},
    )


if __name__ == "__main__":
    main()
