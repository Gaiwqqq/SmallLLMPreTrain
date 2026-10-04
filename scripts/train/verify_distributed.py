#!/usr/bin/env python3
"""真实 GPU/CPU rank 的首步梯度更新对照，包括空 rank 的尾部。"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from dummym.data.packed import PackedRows
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.training.distributed import ProcessContext, initialize
from dummym.training.engine import train_update
from dummym.training.recipe import TrainRecipe, make_optimizer
from dummym.utils.files import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    context = initialize(args.device)
    try:
        if context.primary:
            args.output.mkdir(parents=True, exist_ok=True)
            (np.arange(56, dtype=np.uint16) % 61 + 3).astype("<u2").tofile(
                args.output / "train.bin"
            )
        context.barrier()
        torch.manual_seed(7)
        torch.set_num_threads(1)
        config = MiniLlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        reference = MiniLlamaForCausalLM(config).to(context.device)
        replica = MiniLlamaForCausalLM(config).to(context.device)
        replica.load_state_dict(reference.state_dict())
        model = (
            DistributedDataParallel(
                replica,
                device_ids=[context.device.index] if context.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            if context.world_size > 1
            else replica
        )
        recipe = TrainRecipe(
            total_tokens=56, global_batch_size=8, micro_batch_size=1, precision="fp32"
        )
        dataset = PackedRows(args.output / "train.bin", 7, 8, 64, False)
        optimizer = make_optimizer(model, recipe, False)
        baseline = make_optimizer(reference, recipe, False)
        single = ProcessContext(0, 1, context.device)
        differences = []
        for indices in (np.arange(7), np.array([3])):
            loss, _ = train_update(model, optimizer, dataset, indices, recipe, context)
            expected, _ = train_update(
                reference, baseline, dataset, indices, replace(recipe, micro_batch_size=8), single
            )
            if abs(loss - expected) > 1e-5:
                raise AssertionError(f"Loss mismatch: {loss} vs {expected}")
            difference = 0.0
            for left, right in zip(replica.parameters(), reference.parameters()):
                torch.testing.assert_close(left, right, atol=3e-6, rtol=1e-4)
                difference = max(difference, (left - right).abs().max().item())
            differences.append(difference)
        if context.primary:
            write_json(
                args.output / "summary.json",
                {
                    "status": "passed",
                    "world_size": context.world_size,
                    "precision": "fp32",
                    "max_parameter_differences": differences,
                    "empty_rank_tail": True,
                    "atol": 3e-6,
                    "rtol": 1e-4,
                },
            )
    finally:
        context.close()


if __name__ == "__main__":
    main()
