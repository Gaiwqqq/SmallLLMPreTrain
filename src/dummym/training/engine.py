"""显式 DDP 训练循环：全局取样、累积梯度、更新、验证、保存。

先阅读 train_update，再阅读 run。尾部 batch 不重复或丢弃样本；DDP 平均梯度，
所以局部 loss 乘 world_size × 局部样本数 / 全局样本数。
"""

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from dummym.checkpoint.distributed import restore_rng, save_checkpoint
from dummym.data.packed import PackedRows, load_packed
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.training.distributed import ProcessContext
from dummym.training.recipe import TrainRecipe, learning_rate_factor, make_optimizer
from dummym.utils.files import write_json


def autocast(context: ProcessContext, recipe: TrainRecipe):
    if recipe.precision == "bf16":
        return torch.autocast(context.device.type, dtype=torch.bfloat16)
    return nullcontext()


def train_update(
    model,
    optimizer,
    dataset: PackedRows,
    indices: np.ndarray,
    recipe: TrainRecipe,
    context: ProcessContext,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    local = indices[context.rank :: context.world_size]
    windows = math.ceil(len(indices) / (context.world_size * recipe.micro_batch_size))
    loss_sum = torch.zeros((), dtype=torch.float64, device=context.device)
    for window in range(windows):
        part = local[window * recipe.micro_batch_size : (window + 1) * recipe.micro_batch_size]
        # 所有 rank 必须完成相同次数的 forward/backward，包括不足整除的尾部。
        batch = dataset.batch(part if len(part) else indices[:1], context.device)
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and window < windows - 1
            else nullcontext()
        )
        with sync:
            with autocast(context, recipe):
                loss = model(input_ids=batch, labels=batch).loss
                invalid = (~torch.isfinite(loss)).to(torch.int32)
                if context.sum(invalid).item():
                    raise FloatingPointError(
                        "Nonfinite loss on at least one rank; no optimizer update"
                    )
                weight = context.world_size * len(part) / len(indices)
                weighted = loss * weight
            # no_sync 同时包围 forward 和 backward；只在最后一个窗口同步梯度。
            weighted.backward()
        loss_sum += loss.detach().double() * len(part)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True
    )
    optimizer.step()
    return float(context.sum(loss_sum).item() / len(indices)), float(norm)


@torch.inference_mode()
def evaluate(
    model, dataset: PackedRows, recipe: TrainRecipe, context: ProcessContext, max_rows: int = 0
) -> float:
    was_training = model.training
    model.eval()
    count = min(len(dataset), max_rows) if max_rows else len(dataset)
    indices = np.arange(context.rank, count, context.world_size)
    totals = torch.zeros(2, dtype=torch.float64, device=context.device)
    try:
        for start in range(0, len(indices), recipe.micro_batch_size):
            part = indices[start : start + recipe.micro_batch_size]
            batch = dataset.batch(part, context.device)
            with autocast(context, recipe):
                loss = model(batch, labels=batch).loss
            totals[0] += loss.double() * len(part)
            totals[1] += len(part)
        context.sum(totals)
        if not torch.isfinite(totals).all().item() or totals[1].item() == 0:
            raise FloatingPointError("Invalid validation loss/count")
        return float((totals[0] / totals[1]).item())
    finally:
        model.train(was_training)


def run(
    config: MiniLlamaConfig,
    recipe: TrainRecipe,
    data_dir: Path,
    output: Path,
    context: ProcessContext,
    *,
    resume: Path | None = None,
    stop_after_steps: int = 0,
    eval_every: int = 500,
    save_every: int = 500,
    eval_rows: int = 0,
    in_memory: bool = False,
    compile_model: bool = False,
    max_hours: float = 96,
) -> dict:
    recipe.validate(context.world_size)
    if recipe.precision == "bf16" and (
        context.device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("BF16 requires a supported CUDA device")
    torch.set_num_threads(4)
    datasets, fingerprint, tokenizer_path = load_packed(data_dir, config, in_memory)
    length = datasets["train"].sequence_length
    total_rows = recipe.total_tokens // length
    if not 0 < total_rows <= len(datasets["train"]):
        raise ValueError("Token budget exceeds unique packed training data; prepare more data")
    total_steps = math.ceil(total_rows / recipe.global_batch_size)
    contract = {
        "model": config.to_dict(),
        "recipe": asdict(recipe),
        "data": fingerprint,
        "world_size": context.world_size,
        "total_steps": total_steps,
        "device_type": context.device.type,
        "compile": compile_model,
        "eval_rows": eval_rows,
    }
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2 or checkpoint["contract"] != contract:
            raise ValueError("Checkpoint/data/recipe/world-size mismatch")
    if context.primary:
        output.mkdir(parents=True, exist_ok=resume is not None)
        shutil.copyfile(tokenizer_path, output / "tokenizer.json")
        write_json(output / "config.json", contract)
    context.barrier()
    torch.manual_seed(recipe.seed)
    raw_model = MiniLlamaForCausalLM(config).to(context.device)
    optimizer = make_optimizer(raw_model, recipe, context.device.type == "cuda")
    warmup = min(total_steps - 1, int(total_steps * recipe.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda index: learning_rate_factor(index, total_steps, warmup, recipe.min_lr_ratio),
    )
    progress = {
        "step": 0,
        "next_sequence": 0,
        "tokens_seen": 0,
        "elapsed_seconds": 0.0,
        "validation_loss": None,
    }
    if checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        progress = checkpoint["progress"]
        cursor = progress["next_sequence"]
        if not 0 <= cursor <= total_rows or progress["tokens_seen"] != cursor * length:
            raise ValueError("Invalid checkpoint cursor/token counter")
        if (
            progress["step"] != math.ceil(cursor / recipe.global_batch_size)
            or scheduler.last_epoch != progress["step"]
        ):
            raise ValueError("Checkpoint cursor/scheduler is not at an update boundary")
        if cursor < total_rows and cursor % recipe.global_batch_size:
            raise ValueError("Checkpoint cursor is inside a partial update")
    model = torch.compile(raw_model) if compile_model else raw_model
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.device.index] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    if checkpoint:
        restore_rng(checkpoint["rng_states"], context)
        del checkpoint
    order = torch.randperm(
        len(datasets["train"]), generator=torch.Generator().manual_seed(recipe.seed)
    ).numpy()
    writer = (
        SummaryWriter(str(output / "tensorboard"), purge_step=progress["step"] + 1)
        if context.primary
        else None
    )
    metrics_path = output / "metrics.jsonl"
    if context.primary and resume and metrics_path.exists():
        retained = [
            line
            for line in metrics_path.read_text().splitlines()
            if json.loads(line)["step"] <= progress["step"]
        ]
        metrics_path.write_text("\n".join(retained) + ("\n" if retained else ""))
    log = metrics_path.open("a") if context.primary else None
    started = time.perf_counter()
    previous_elapsed = progress["elapsed_seconds"]
    stop_step = min(total_steps, stop_after_steps or total_steps)
    milestone_steps = {
        max(1, math.ceil(total_steps * fraction)) for fraction in (0.1, 0.25, 0.5, 1.0)
    }
    try:
        if progress["step"] == 0:
            progress["validation_loss"] = evaluate(
                raw_model, datasets["validation"], recipe, context, eval_rows
            )
            save_checkpoint(
                output / "checkpoint.pt",
                raw_model,
                optimizer,
                scheduler,
                progress,
                contract,
                context,
            )
        raw_model.train()
        while progress["step"] < stop_step:
            cursor = progress["next_sequence"]
            end = min(total_rows, cursor + recipe.global_batch_size)
            indices = order[cursor:end]
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            update_started = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            loss, norm = train_update(model, optimizer, datasets["train"], indices, recipe, context)
            scheduler.step()
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            seconds = time.perf_counter() - update_started
            elapsed = previous_elapsed + time.perf_counter() - started
            progress.update(
                step=progress["step"] + 1,
                next_sequence=end,
                tokens_seen=end * length,
                elapsed_seconds=elapsed,
            )
            step = progress["step"]
            metrics = {
                "step": step,
                "loss": loss,
                "learning_rate": lr,
                "gradient_norm": norm,
                "tokens_seen": end * length,
                "tokens_per_second": len(indices) * length / seconds,
                "targets_per_second": len(indices) * (length - 1) / seconds,
                "update_seconds": seconds,
            }
            if context.device.type == "cuda":
                memory = torch.tensor(
                    torch.cuda.max_memory_allocated(context.device) / 1024**3, device=context.device
                )
                if context.world_size > 1:
                    torch.distributed.all_reduce(memory, op=torch.distributed.ReduceOp.MAX)
                metrics["max_rank_peak_memory_gib"] = memory.item()
            timeout = (
                context.sum(
                    torch.tensor(int(elapsed >= max_hours * 3600), device=context.device)
                ).item()
                > 0
            )
            final = step == stop_step or timeout
            if step % eval_every == 0 or final or step in milestone_steps:
                progress["validation_loss"] = evaluate(
                    raw_model, datasets["validation"], recipe, context, eval_rows
                )
                metrics["validation_loss"] = progress["validation_loss"]
            if context.primary:
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                for key, value in metrics.items():
                    if key != "step":
                        writer.add_scalar(key, value, step)
                if step == 1 or step % 10 == 0 or final:
                    print(json.dumps(metrics), flush=True)
            if step % save_every == 0 or final or step in milestone_steps:
                progress["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
                save_checkpoint(
                    output / "checkpoint.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    progress,
                    contract,
                    context,
                )
                if context.primary and (step in milestone_steps or final):
                    milestone = output / "milestones" / f"step-{step:06d}"
                    milestone.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(output / "checkpoint.pt", milestone / "checkpoint.pt")
                    shutil.copyfile(output / "tokenizer.json", milestone / "tokenizer.json")
                    write_json(
                        milestone / "ready.json", {"progress": progress, "contract": contract}
                    )
                context.barrier()
            if timeout:
                break
        summary = {
            "status": "complete" if progress["step"] == total_steps else "paused",
            "progress": progress,
            "contract": contract,
            "parameters": raw_model.num_parameters(),
        }
        if context.primary:
            write_json(output / "summary.json", summary)
        return summary
    finally:
        if writer:
            writer.close()
        if log:
            log.close()
