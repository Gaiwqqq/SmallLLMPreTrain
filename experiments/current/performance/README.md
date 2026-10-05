# 性能实验：batch、卡数与 compile 的选择

## 问题与原因

相同模型能跑，不代表配置快。目标是在数值正确、global batch 不变、显存有余量的前提下提高训练更新吞吐。

## 为什么这样设计

用合成 token 排除下载与数据质量影响，固定 global batch=256，扫 micro-batch=4/8/16/32/64，再比较单/双/四卡。先预热，再取多步更新墙钟时间中位数。

如果同时增大 global batch，就改变了更新条件；如果把单卡吞吐直接乘四，就漏掉了通信。因此吞吐按所有 rank 的总输入 token 和真实墙钟统计。

compile 单独先检查数值，再要求至少 10% 稳态收益。未通过不能临时放宽容限来追求漂亮速度。

## 实测与决定

四卡 eager、micro-batch=16 时约 217,856 token/s，每卡峰值 allocated 显存 44.70 GiB。compile 有 12 个 logits 超出容限，保留 eager。正式更新日志约 21.8 万 token/s，与合成基准相近。

这是计算性能证据，不是语言质量证据；验证、保存、上传及初次加载仍影响总耗时。未通过的 compile 也不代表所有配置都不适用。

证据 `runs/benchmark-*/summary.json`、`runs/performance-selection.json`、`logs/verify-compile.log`，完整解释见 [性能阶段](../../../docs/04_distributed.md)。
