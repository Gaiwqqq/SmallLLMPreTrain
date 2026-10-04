"""分布式进程生命周期；训练算法不依赖隐式的全局 rank 状态。"""

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ProcessContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def sum(self, values: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(values)
        return values

    def close(self) -> None:
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def initialize(device_type: str = "cuda") -> ProcessContext:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if device_type == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return ProcessContext(rank, world_size, device)
