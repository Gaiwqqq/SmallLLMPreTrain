# SmallLLMPreTrain：从零训练英文小模型

这是一个面向学习和代码审阅的 Llama-like 训练工程：先理解原生 PyTorch 模型，再实现数据流水线、四卡 DDP、预训练、SFT 和聊天推理。正式模型从随机权重开始，Tokenizer 自行训练。

## 当前状态

213M Base 已完成约 77 亿 token 预训练；正式 Chat SFT 已完成 7,060 步，验证 loss 为 1.149027。推理依赖故障已修复，服务运行于训练机回环 8000 端口。**首轮聊天语义验收未达标：46/120，38.3%，多轮上下文 0%。** 结构正常率 99.2% 不能当成语义通过率。详见 [首轮交付报告](docs/first_round_delivery.md)。

用户已授权 [第二轮格式/状态课程实验](experiments/current/sft_curriculum/README.md)：同一 Chat 起点，两个两卡短跑比较 3e-5/1e-4，保留一般对话回放。运行状态以远程 `runs/sft-curriculum-v1/status.json` 和训练日志为准。历史 M0–M2 与旧自动阶段表不能当成当前状态。

## 阅读顺序

第一次接触 LLM，请先读 [初学者导读](docs/reading_guide.md)，遇到陌生词查 [术语表](docs/glossary.md)。每个实验的动机、设计、观察和结论都汇总在 [实验目录](experiments/README.md)。

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
