"""TRL 全参数 SFT：只监督 assistant 内容，原模型仍从随机权重预训练。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from datasets import load_dataset
from accelerate import PartialState
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from dummym.utils.files import write_json


def assert_assistant_mask(tokenizer) -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Good morning"},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    mask = encoded["assistant_masks"]
    if not any(mask) or all(mask):
        raise ValueError("Chat template does not identify assistant-only training targets")
    supervised = tokenizer.decode(
        [token for token, keep in zip(encoded["input_ids"], mask) if keep]
    )
    if "Good morning" not in supervised or "Hello" in supervised:
        raise ValueError("Assistant mask incorrectly includes user text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-conversations", type=int, default=0, help="SFT 对照短跑规模，0 使用全部"
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.epochs)
        or args.epochs <= 0
    ):
        parser.error("Learning rate and epochs must be positive and finite")
    if args.batch_size <= 0 or args.max_conversations < 0:
        parser.error("Invalid batch size or conversation limit")
    if args.output.exists() and args.resume is None:
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert_assistant_mask(tokenizer)
    manifest = json.loads((args.data_dir / "download.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete SFT download")
    files = [
        str(args.data_dir / item["path"]) for item in manifest["files"] if "/train-" in item["path"]
    ]
    # 按整个会话去重，不将同一会话的不同轮次拆到两个 split。
    seen = set()

    def unique(example):
        digest = hashlib.sha256(
            json.dumps(example["messages"], sort_keys=True).encode()
        ).hexdigest()
        if digest in seen:
            return False
        seen.add(digest)
        return True

    with PartialState().main_process_first():
        dataset = load_dataset("parquet", data_files=files, split="train")
        dataset = dataset.filter(unique, load_from_cache_file=True)
    split = dataset.train_test_split(test_size=0.01, seed=2026)
    if args.max_conversations:
        split["train"] = split["train"].select(
            range(min(args.max_conversations, len(split["train"])))
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    config = SFTConfig(
        output_dir=str(args.output),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        max_length=2048,
        packing=False,
        assistant_only_loss=True,
        bf16=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        logging_steps=10,
        report_to="tensorboard",
        seed=2026,
        optim="adamw_torch_fused",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output / "final"))
    validation_metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output / "final")
        write_json(
            args.output / "summary.json",
            {
                "status": "complete",
                "metrics": result.metrics,
                "train_conversations": len(split["train"]),
                "validation_conversations": len(split["test"]),
                "source_revision": manifest["revision"],
                "learning_rate": args.learning_rate,
                "validation_loss": validation_metrics["eval_loss"],
            },
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
