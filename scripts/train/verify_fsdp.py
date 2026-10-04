#!/usr/bin/env python3
"""FSDP2 教学对照：根模型分片与单卡 FP32 SGD 更新的数值一致性。"""

import argparse
from pathlib import Path

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.training.distributed import initialize
from dummym.utils.files import write_json


class LossModel(torch.nn.Module):
    """FSDP2 遍历 Tensor 输出挂 backward hook；显式适配教学 dataclass 输出。"""

    def __init__(self, config):
        super().__init__()
        self.decoder = MiniLlamaForCausalLM(config)

    def forward(self, input_ids):
        return self.decoder(input_ids, labels=input_ids).loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = initialize()
    try:
        torch.manual_seed(7)
        config = MiniLlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        reference = MiniLlamaForCausalLM(config).to(context.device)
        sharded = LossModel(config).to(context.device)
        sharded.decoder.load_state_dict(reference.state_dict())
        mesh = init_device_mesh("cuda", (context.world_size,))
        fully_shard(sharded, mesh=mesh)
        optimizer = torch.optim.SGD(sharded.parameters(), lr=1e-3)
        baseline = torch.optim.SGD(reference.parameters(), lr=1e-3)
        ids = (
            torch.arange(context.world_size * 2 * 8, device=context.device).reshape(-1, 8) % 61 + 3
        )
        local = ids[context.rank :: context.world_size]
        loss = sharded(local)
        loss.backward()
        optimizer.step()
        reference(ids, labels=ids).loss.backward()
        baseline.step()
        differences = []
        for name, parameter in sharded.named_parameters():
            complete = parameter.full_tensor()
            expected = reference.get_parameter(name.removeprefix("decoder."))
            torch.testing.assert_close(complete, expected, atol=3e-6, rtol=1e-4)
            differences.append((complete - expected).abs().max().item())
        if context.primary:
            write_json(
                args.output / "summary.json",
                {
                    "status": "passed",
                    "world_size": context.world_size,
                    "max_parameter_difference": max(differences),
                    "optimizer": "SGD",
                    "precision": "fp32",
                    "note": "Teaching correctness check; production uses DDP/AdamW, not this script",
                },
            )
    finally:
        context.close()


if __name__ == "__main__":
    main()
