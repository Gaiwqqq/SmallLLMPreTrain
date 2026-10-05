"""TRL 全参数 SFT：只监督 assistant 内容，原模型仍从随机权重预训练。"""

import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile

from accelerate import PartialState
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from dummym.posttrain.data import (
    data_contract,
    encode_conversation,
    has_targets,
    load_conversations,
    load_prepared,
    prepare_data,
)
from dummym.utils.files import write_json


class ArchiveTrainerCheckpoint(TrainerCallback):
    """在所有 rank 完成保存后，保留不受 Trainer 轮转删除影响的恢复快照。"""

    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if not state.is_world_process_zero:
            return control
        milestones = Path(args.output_dir) / "trainer-milestones"
        completed = [
            json.loads(path.read_text())["step"] for path in milestones.glob("*/ready.json")
        ]
        previous = max(completed, default=0)
        thresholds = [max(1, math.ceil(state.max_steps * ratio)) for ratio in (0.1, 0.25, 0.5, 1)]
        if not any(previous < threshold <= state.global_step for threshold in thresholds):
            return control
        source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        milestones.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".saving-", dir=milestones))
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        destination = milestones / f"step-{state.global_step:06d}"
        temporary.rename(destination)
        write_json(
            destination / "ready.json",
            {
                "format": "transformers-trainer",
                "step": state.global_step,
                "epoch": state.epoch,
                "max_steps": state.max_steps,
                "world_size": args.world_size,
                "resume": "Trainer resume_from_checkpoint; preserve model, optimizer, scheduler and all rank RNG files",
            },
        )
        return control


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
    parser.add_argument("--prepared-data", type=Path, help="独立 CPU 阶段生成的完整 SFT 数据")
    parser.add_argument("--prepare-only", action="store_true", help="仅在单进程中准备数据")
    parser.add_argument("--data-workers", type=int, default=8)
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
    if args.data_workers <= 0:
        parser.error("Data workers must be positive")
    if args.prepare_only and (args.prepared_data is None or args.resume is not None):
        parser.error("Preparation requires --prepared-data and forbids --resume")
    if not args.prepare_only and args.output.exists() and args.resume is None:
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert_assistant_mask(tokenizer)
    manifest = json.loads((args.data_dir / "download.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete SFT download")
    if args.prepare_only:
        # 在创建 PartialState/SFTConfig 前返回，不初始化 NCCL，也不加载模型权重。
        prepare_data(args.data_dir, args.model, args.prepared_data, tokenizer, args.data_workers)
        return
    if args.prepared_data:
        split = load_prepared(
            args.prepared_data, data_contract(args.data_dir, args.model, tokenizer)
        )
    else:
        # 小规模教学入口仍可在线准备；正式四卡任务必须使用预处理产物。
        with PartialState().main_process_first():
            dataset, _ = load_conversations(args.data_dir)
            dataset = dataset.map(encode_conversation, fn_kwargs={"tokenizer": tokenizer})
            dataset = dataset.filter(has_targets)
        split = dataset.train_test_split(test_size=0.01, seed=2026)
    if args.max_conversations:
        split["train"] = split["train"].select(
            range(min(args.max_conversations, len(split["train"])))
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        # 全参数 SFT 保留 FP32 权重与 AdamW 状态，BF16 只用于 Trainer 的计算。
        dtype=torch.float32,
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
        dataset_kwargs={"skip_prepare_dataset": True},
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
        callbacks=[ArchiveTrainerCheckpoint()] if not args.max_conversations else [],
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
