# 本次运行记录

2026-10-04，起始 commit：a483269096538609f348cc04a0f23349edbba345。

| 检查 | 本次结果 | 证据 |
|---|---|---|
| GPU 与环境 | PyTorch 2.7.1+cu126，四张 H20，BF16 可用 | logs/bootstrap.log、envs/train.lock.txt |
| CPU 单元与集成测试 | 35 passed，37.50s | logs/tests.log |
| HF 导出对照 | FP32 logits、loss、梯度通过 | tests/unit/model/test_huggingface_export.py |
| DDP 尾部更新 | 两个真实 CPU rank 对照通过，包括空 rank | tests/integration/test_ddp.py |
| M0 tiny overfit | 第 223 step 达到 loss < 0.05；准确率 100% | runs/m00_tiny_overfit/summary.json |
| 扩展测试 | 38 passed，10.84s | logs/tests.log |
| 四张 H20 DDP 更新对照 | 最大参数差异约 9e-8，空 rank 尾部通过 | runs/m03_gpu_correctness/summary.json |
| 四卡 TRL SFT 接口 | 32 个合成会话，训练、验证、模型保存通过 | runs/sft-interface-smoke/summary.json |
| 四卡 BF16 中断恢复 | dropout=0.1、不均匀尾部；RNG 完全一致，参数最大差异 2.91e-11 | runs/m03_ddp_resume/summary.json |
| 四卡 FSDP2 教学对照 | FP32 SGD 更新差异 1.86e-9；Tensor 输出适配层通过 | runs/m03_fsdp/summary.json |
| M1 39M 完整预训练 | 99,999,744 输入 token，3052 更新；验证 loss 10.472692 → 4.219103 | runs/m01_39m/summary.json |
| ModelScope | 私有仓库已创建，根 README 和阶段说明已同步 | [模型仓库](https://modelscope.cn/models/GaiWeiqi/SmallLLMPreTrain-English) |
| 真实 pilot 数据审计 | 300,000 文档指纹、分割及跨来源精确去重通过；Tokenizer 字符往返、特殊 ID 和 assistant mask 通过 | runs/pilot-audit.json |

证据路径相对于远程 `/diff/gaiwq/llm_pretrain`。Tiny overfit 不衡量泛化，当前还没有通过聊天验收的模型。M1 与历史 loss 4.153138 的差异来自本次不同的抽样分片和运行栈，未使用历史验证集。六组 99M 对照已完成；正式预训练、正式 SFT 和能力评测仍待完成。

## 99M 对照与性能实测

每组读取 99,999,744 token，保持数据、模型和更新数一致。下表为本次独立运行的最终验证交叉熵；数值越低越好。

| 学习率 | warmup 更新数 | seed 2026 | seed 2027 |
|---|---:|---:|---:|
| 6e-4 | 100 | 3.769134 | 3.758341 |
| 1e-3 | 100 | 3.725739 | 3.704958 |
| 1e-3 | 300 | 3.664364 | 3.647846 |

两组 seed 均支持较长 warmup 在本次配置中更好；213M 使用新的混合语料和自训 tokenizer，仍需单独调参。这些结果不等同于聊天能力。

![本次基线训练与验证曲线](figures/baselines.png)

曲线从本次 TensorBoard 原始日志生成；训练粗线使用 EMA（0.95），细线保留原始波动。纵轴放大到 3.4–6.0，初始化约 10.5 的 loss 在图外。复绘入口为 `scripts/evaluation/plot_baselines.py`。

213M 四卡 eager 合成数据基准：micro-batch=16/global-batch=256，217,856 输入 token/s，峰值显存约 44.70 GiB/卡。该测量排除了下载、packing、验证和 checkpoint 上传，正式训练预算会留出余量。最新测试为 39 passed；保留 FP32 参数的四卡 BF16 TRL 接口复测也通过（`runs/sft-interface-smoke-v3/summary.json`）。

## 自训 Tokenizer 的 213M pilot

四组均从随机权重训练，读取 99,997,696 token、完成 191 次更新，使用相同混合语料和自训 BPE。

| 学习率 | seed | 最终验证 loss |
|---|---:|---:|
| 3e-4 | 2026 | 5.988814 |
| 6e-4 | 2026 | 6.031944 |
| 1e-3 | 2026 | 6.136201 |
| 6e-4 | 2027 | 6.013090 |

本轮正式训练选择 3e-4，依据见 `runs/lr-selection.json`。与 99M 的编码和语料不同，两个阶段的 loss 数值不能直接比较。

`exports/pilot-base-lr3e4` 已通过 HF 导出和四个纯文本续写样例测试。实际输出重复明显：例如 `Water is important because` 后重复生成 “the little girl” 等故事片段。这是尚未充分训练的 Base，不是合格聊天模型；原始样例保留在 `runs/pilot-completions.json`，后续应检验扩大预训练量和 SFT 是否带来改善。

四卡 SFT 恢复快照接口也已实测通过并上传：`sft-interface-smoke-v4` 含模型、optimizer、scheduler、trainer_state 和四份 RNG。它使用极小合成模型，仅验证接口。最新常规测试为 39 passed（11.46s）。

2026-10-05，本轮所选预训练原始分片（24/16/4）及 SFT 的 5 个分片全部下载完成；全量清洗正在进行。pilot 模型、原始续写样例和阶段指南已同步 ModelScope。正式预训练须等待唯一 token 容量统计，不能把下载字节数当作训练 token 数。

## 自动阶段进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| engineering-checks | complete | /diff/gaiwq/llm_pretrain/logs/engineering-checks.log |
| lint-checks | complete | /diff/gaiwq/llm_pretrain/logs/lint-checks.log |
| format-checks | complete | /diff/gaiwq/llm_pretrain/logs/format-checks.log |
| download-cosmopedia-pilot | complete | /diff/gaiwq/llm_pretrain/logs/download-cosmopedia-pilot.log |
| download-tinystories-pilot | complete | /diff/gaiwq/llm_pretrain/logs/download-tinystories-pilot.log |
| clean-pilot | complete | /diff/gaiwq/llm_pretrain/logs/clean-pilot.log |
| train-tokenizer | complete | /diff/gaiwq/llm_pretrain/logs/train-tokenizer.log |
| pack-pilot | complete | /diff/gaiwq/llm_pretrain/logs/pack-pilot.log |
| pipeline | failed | RuntimeError: pack-pilot failed with exit code 1 |
| pilot-capacity-fineweb | complete | /diff/gaiwq/llm_pretrain/logs/pilot-capacity-fineweb.log |
| pilot-capacity-cosmopedia | complete | /diff/gaiwq/llm_pretrain/logs/pilot-capacity-cosmopedia.log |
| pilot-capacity-tinystories | complete | /diff/gaiwq/llm_pretrain/logs/pilot-capacity-tinystories.log |
| pilot-previous-attempt | retained | Incomplete packing preserved at /diff/gaiwq/llm_pretrain/data/failed/pilot100m-1791131011 |
| full-data-background | running | /diff/gaiwq/llm_pretrain/logs/full-data.log |
| reproduce-baselines | complete | /diff/gaiwq/llm_pretrain/logs/reproduce-baselines.log |
| verify-ddp-resume | complete | /diff/gaiwq/llm_pretrain/logs/verify-ddp-resume.log |
| verify-trl-sft | complete | /diff/gaiwq/llm_pretrain/logs/verify-trl-sft.log |
| verify-four-gpu-ddp | complete | /diff/gaiwq/llm_pretrain/logs/verify-four-gpu-ddp.log |
| verify-fsdp2 | complete | /diff/gaiwq/llm_pretrain/logs/verify-fsdp2.log |
| benchmark-single | complete | /diff/gaiwq/llm_pretrain/logs/benchmark-single.log |
| benchmark-2-gpu | complete | /diff/gaiwq/llm_pretrain/logs/benchmark-2-gpu.log |
| benchmark-4-gpu | complete | /diff/gaiwq/llm_pretrain/logs/benchmark-4-gpu.log |
| verify-compile | failed | /diff/gaiwq/llm_pretrain/logs/verify-compile.log |
| compile-selection | fallback | Eager retained: RuntimeError |
| m04_lr_pilot_0 | complete | /diff/gaiwq/llm_pretrain/logs/m04_lr_pilot_0.log |
| m04_lr_pilot_1 | complete | /diff/gaiwq/llm_pretrain/logs/m04_lr_pilot_1.log |
| m04_lr_pilot_2 | complete | /diff/gaiwq/llm_pretrain/logs/m04_lr_pilot_2.log |
| m04_lr_pilot_3 | complete | /diff/gaiwq/llm_pretrain/logs/m04_lr_pilot_3.log |
