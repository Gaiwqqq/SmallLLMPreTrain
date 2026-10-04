"""真实两个 CPU rank，与单进程对照；尾部仅一个 rank 有有效样本。"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dummym.data.packed import PackedRows
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.training.distributed import ProcessContext
from dummym.training.engine import train_update
from dummym.training.recipe import TrainRecipe, make_optimizer


def worker(rank, root, world_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=world_size
    )
    try:
        context = ProcessContext(rank, world_size, torch.device("cpu"))
        torch.manual_seed(7)
        config = MiniLlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        reference = MiniLlamaForCausalLM(config)
        replica = MiniLlamaForCausalLM(config)
        replica.load_state_dict(reference.state_dict())
        model = DistributedDataParallel(replica, broadcast_buffers=False)
        recipe = TrainRecipe(
            total_tokens=56, global_batch_size=8, micro_batch_size=2, precision="fp32"
        )
        dataset = PackedRows(Path(root) / "train.bin", 7, 8, 64, False)
        optimizer = make_optimizer(model, recipe, False)
        baseline = make_optimizer(reference, recipe, False)
        single = ProcessContext(0, 1, torch.device("cpu"))
        for indices in (np.arange(7), np.array([3])):
            loss, _ = train_update(model, optimizer, dataset, indices, recipe, context)
            expected, _ = train_update(
                reference, baseline, dataset, indices, replace(recipe, micro_batch_size=8), single
            )
            assert abs(loss - expected) < 1e-6
            for left, right in zip(replica.parameters(), reference.parameters()):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def test_two_rank_updates_match_single_process_with_empty_tail_rank(tmp_path):
    (np.arange(56, dtype=np.uint16) % 61 + 3).astype("<u2").tofile(tmp_path / "train.bin")
    mp.spawn(worker, args=(str(tmp_path), 2), nprocs=2, join=True)
