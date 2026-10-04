#!/usr/bin/env python3
"""从本次 TensorBoard 日志画基线曲线，不读取或复用历史报告数值。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def curve(root: Path, name: str, tag: str):
    events = EventAccumulator(str(root / name / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    tokens = {item.step: item.value for item in events.Scalars("train/tokens_seen")}
    values = events.Scalars(tag)
    x = [tokens.get(item.step, 0) / 1e6 for item in values]
    y = [item.value for item in values]
    return x, y


def smooth(values, beta=0.95):
    result, mean = [], values[0]
    for value in values:
        mean = beta * mean + (1 - beta) * value
        result.append(mean)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for name, label in (
        ("m01_39m", "39M baseline"),
        ("m02_99m_lr1e3_w300_s2026", "99M LR=1e-3, warmup=300"),
    ):
        x, y = curve(args.runs, name, "train/loss")
        axes[0].plot(x, y, alpha=0.15)
        axes[0].plot(x, smooth(y), label=label)
    for name, label in (
        ("m02_99m_lr6e4_w100_s2026", "LR=6e-4, warmup=100"),
        ("m02_99m_lr1e3_w100_s2026", "LR=1e-3, warmup=100"),
        ("m02_99m_lr1e3_w300_s2026", "LR=1e-3, warmup=300"),
    ):
        x, y = curve(args.runs, name, "validation/loss")
        axes[1].plot(x, y, marker="o", markersize=3, label=label)
    for axis in axes:
        axis.set_xlabel("Input tokens (millions)")
        axis.set_ylabel("Next-token cross entropy")
        axis.set_ylim(3.4, 6.0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_title("Training loss (EMA beta=0.95)")
    axes[1].set_title("99M validation, seed 2026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
