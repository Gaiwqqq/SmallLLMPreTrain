#!/usr/bin/env python3
"""正式架构的 eager/compile BF16 logits、loss 与梯度对照。"""

import argparse
import copy
from pathlib import Path

import torch
import yaml

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.utils.files import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(2026)
    config = MiniLlamaConfig.from_dict(
        yaml.safe_load(args.model_config.read_text())["model_config"]
    )
    reference = MiniLlamaForCausalLM(config).cuda()
    replica = copy.deepcopy(reference)
    compiled = torch.compile(replica)
    ids = torch.randint(6, config.vocab_size, (2, 128), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        left, right = reference(ids, labels=ids), compiled(ids, labels=ids)
    torch.testing.assert_close(left.logits, right.logits, rtol=0.03, atol=0.05)
    torch.testing.assert_close(left.loss, right.loss, rtol=0.002, atol=0.002)
    left.loss.backward()
    right.loss.backward()
    max_difference = 0.0
    for expected, actual in zip(reference.parameters(), replica.parameters()):
        torch.testing.assert_close(expected.grad, actual.grad, rtol=0.05, atol=0.002)
        max_difference = max(max_difference, (expected.grad - actual.grad).abs().max().item())
    write_json(
        args.output / "summary.json",
        {
            "status": "passed",
            "precision": "bf16",
            "max_logit_difference": (left.logits - right.logits).abs().max().item(),
            "loss_difference": abs(left.loss.item() - right.loss.item()),
            "max_gradient_difference": max_difference,
            "note": "BF16 tolerance permits roundoff; no bitwise equality claimed",
        },
    )


if __name__ == "__main__":
    main()
