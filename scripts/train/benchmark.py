#!/usr/bin/env python3
"""固定全局 batch 的稳态 GPU 更新基准；合成数据不用于能力结论。"""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import yaml

from dummym.data.packed import PackedRows
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.training.distributed import initialize
from dummym.training.engine import train_update
from dummym.training.recipe import TrainRecipe, make_optimizer
from dummym.utils.files import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    context = initialize()
    try:
        config = MiniLlamaConfig.from_dict(
            yaml.safe_load(args.model_config.read_text())["model_config"]
        )
        length, global_batch = 2048, 256
        if context.primary:
            args.output.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(2026)
            rng.integers(6, config.vocab_size, size=(global_batch, length), dtype=np.uint16).astype(
                "<u2"
            ).tofile(args.output / "synthetic.bin")
        context.barrier()
        dataset = PackedRows(
            args.output / "synthetic.bin", global_batch, length, config.vocab_size, True
        )
        results = []
        for micro in args.micro_batches:
            recipe = TrainRecipe(
                total_tokens=global_batch * length,
                global_batch_size=global_batch,
                micro_batch_size=micro,
            )
            recipe.validate(context.world_size)
            torch.manual_seed(2026)
            raw = MiniLlamaForCausalLM(config).to(context.device)
            optimizer = make_optimizer(raw, recipe, True)
            model = torch.compile(raw) if args.compile else raw
            if context.world_size > 1:
                model = DistributedDataParallel(
                    model, device_ids=[context.device.index], broadcast_buffers=False
                )
            torch.cuda.reset_peak_memory_stats(context.device)
            timings = []
            try:
                for step in range(args.steps + 2):
                    torch.cuda.synchronize(context.device)
                    started = time.perf_counter()
                    train_update(
                        model, optimizer, dataset, np.arange(global_batch), recipe, context
                    )
                    torch.cuda.synchronize(context.device)
                    if step >= 2:
                        timings.append(time.perf_counter() - started)
                memory = torch.cuda.max_memory_allocated(context.device)
                result = {
                    "micro_batch": micro,
                    "world_size": context.world_size,
                    "compiled": args.compile,
                    "median_update_seconds": statistics.median(timings),
                    "tokens_per_second": global_batch * length / statistics.median(timings),
                    "peak_memory_gib": memory / 1024**3,
                    "has_memory_margin": memory
                    < torch.cuda.get_device_properties(context.device).total_memory * 0.85,
                }
                results.append(result)
            except torch.OutOfMemoryError:
                if context.world_size > 1:
                    raise  # 多卡 OOM 不尝试继续 collective，交由启动器终止全部 rank。
                results.append({"micro_batch": micro, "status": "out_of_memory"})
            finally:
                del model, raw, optimizer
                torch.cuda.empty_cache()
            if context.primary:
                write_json(
                    args.output / "summary.json", {"results": results, "synthetic_data": True}
                )
                print(json.dumps(results[-1]), flush=True)
    finally:
        context.close()


if __name__ == "__main__":
    main()
