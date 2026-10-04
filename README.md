# SmallLLMPreTrain：从零训练英文小模型

这是一个面向学习和代码审阅的 Llama-like 训练工程：先理解原生 PyTorch 模型，再实现数据流水线、四卡 DDP、预训练、SFT 和聊天推理。正式模型从随机权重开始，Tokenizer 自行训练。

## 当前状态

基础工程与四卡验收已通过：39 项测试、DDP 更新与断点恢复、HF 导出、TRL SFT 和 FSDP2 对照均通过。39M 与六组 99M 基线各完成约 100M token 训练，自训 32K BPE 与混合数据 pilot 已生成。当前执行 213M 模型性能测量和完整语料下载，尚未完成正式预训练或聊天能力验收。具体结果和曲线见 [本次运行记录](docs/results.md)。原项目的 M0–M2 历史结论仍保留在 experiments/，不能混为本次结果。

## 阅读顺序

1. [运行与网络](docs/00_environment.md)：所有远程产物的归属、代理和环境。
2. [模型原理](docs/01_foundations.md)：从一个 batch 读懂 Transformer。
3. [基线复现](docs/02_reproduction.md)：先证明旧流程可以复现。
4. [数据与 Tokenizer](docs/03_data_tokenizer.md)：清洗、隔离、BPE 和 packing。
5. [分布式与性能](docs/04_distributed.md)：DDP、尾部 batch、恢复和测量。
6. [正式预训练](docs/05_pretraining.md)：约 213M 参数模型的训练预算。
7. [聊天后训练](docs/06_sft.md)：SFT 和 assistant-only loss。
8. [评测与推理](docs/07_evaluation_inference.md)：检验能力，使用 CLI/API。
9. [工程约定](docs/engineering.md)：源码组织、审阅重点、测试与自动提交。

远程根目录固定为 `/diff/gaiwq/llm_pretrain`；环境、缓存、临时文件、数据、checkpoint、日志和导出模型均位于该目录。源码在其 `repo/` 子目录，本机源码是当前 Git 仓库。

## 首版目标

英文为主，首轮最多约七天。交付 Base/Chat 模型、Tokenizer、真实评测与阶段说明。目标涵盖指令遵循、摘要、改写、常识和三轮对话；小数据模型能力不作保证。未达标时交付真实结果和恢复点，追加预算另行讨论。

关键代码完成验收后自动进行本地 Git commit；不会自动 push GitHub。关键 checkpoint 和阶段文档自动同步至 [ModelScope 私有模型仓库](https://modelscope.cn/models/GaiWeiqi/SmallLLMPreTrain-English)。令牌不进入 Git。

## 源码导航

- `src/dummym/models/llama_like/`：可直接阅读的 RMSNorm、RoPE、GQA、SwiGLU 和 causal loss。
- `src/dummym/data/`、`tokenizer/`：下载、清洗、BPE、packing 和 uint16 数据读取。
- `src/dummym/training/`：训练参数、进程生命周期、全局取样和显式更新循环。
- `src/dummym/checkpoint/`：完整恢复、Hugging Face 导出和 ModelScope 归档。
- `src/dummym/posttrain/`、`evaluation/`、`inference/`：SFT、能力评测和聊天。
- `scripts/train/pretrain.py`：保留的单卡教学入口，历史实验仍可按原命令复现。

## 后续学习

首版完成后再研究 FSDP2 的扩展、scaling law、Muon、MoE、DPO 和 GRPO。它们不作为首版聊天模型的前置条件。
