# 实验导航：从问题出发阅读

每个实验先回答一个明确问题，再决定下一步。初学者建议先读 [总阅读路线](../docs/reading_guide.md)，再按下面顺序看报告；不要先复制长命令。

## 保留的历史实验

这些报告来自项目早期运行，原始数据、命令与结论全部保留。它们帮助理解实验方法，但不是本次远程训练的完成记录。

| 顺序 | 实验 | 核心问题 |
|---|---|---|
| M0 | [固定小语料过拟合](m00_foundations/exp001_tiny_overfit/README.md) | 链路能否记住几条已知文本？ |
| M1 | [39M 真实文章基线](m01_pretraining/exp001_39m/README.md) | 能否完成训练、独立验证与恢复？ |
| M2a | [99M 学习率](m02_recipe/exp001_lr_sweep/README.md) | 相同预算下哪档 LR 合适？ |
| M2b | [99M warmup](m02_recipe/exp002_warmup/README.md) | 改变升温阶段后最终验证怎样变化？ |

历史 99M 选择 LR=1e-3、warmup=300，限定于其模型、数据和预算。本次 213M 使用新 Tokenizer 与混合语料，重新选择 LR=3e-4，两者不矛盾。

## 本次四卡学习与训练

| 实验 | 为什么在这一步做 |
|---|---|
| [数据与自训 Tokenizer](current/data_tokenizer/README.md) | 先确认输入和预算真实，避免训练了错误或重复的数据 |
| [DDP 更新与尾部](current/ddp_correctness/README.md) | 多卡算得对，才能相信速度与曲线 |
| [中断恢复](current/resume/README.md) | 长任务必需可恢复，不能只保存权重 |
| [HF 导出](current/export/README.md) | 换框架之前确认数学没有变 |
| [FSDP2 教学对照](current/fsdp2/README.md) | 分清分片与数据并行，不因复杂而盲目采用 |
| [性能与 compile](current/performance/README.md) | 固定训练条件实测速率，失败优化不强行启用 |
| [213M 学习率 pilot](current/p213m_lr/README.md) | 新规模、新数据不能直接沿用旧 LR |
| [77 亿 token 正式预训练](current/pretraining/README.md) | 验证更多唯一文本能否改善 Base |
| [SFT 接口与正式对照](current/sft/README.md) | 先确认监督边界，再学习真实对话 |
| [冻结题集与能力验收](current/evaluation/README.md) | 直接检查沟通目标，防止只看 loss |
| [格式与状态课程双 LR 对照](current/sft_curriculum/README.md) | 用精确答案检查课程能否改善格式与状态，观察一般回复是否退化 |
| [四卡纠正与正式 v2 SFT](current/sft_curriculum_v2/README.md) | 修复固定前缀偏差、增加原对话回放，检验更长训练是否泛化 |
| [最终 120 题验收](../docs/final_delivery.md) | 保留每题真实理由，明确 12.5% 通过率与未达标结论 |

## 读报告时的约定

每份 README 解释问题、设计理由、固定/变化条件、证据、结论边界和下一步。历史详细表格放在导读后；本次已收尾，最终状态以 [最终交付](../docs/final_delivery.md) 与 [交付清单](../docs/artifacts.md) 为准。[运行记录](../docs/results.md) 的自动阶段表保留旧控制器历史，不代表最终 v2 状态。

新实验从随机权重或明确规定的同一 Base 开始，不偷偷跨组续训。输出目录不同，产物留在远程 `runs/`；Git 保存代码与解释，ModelScope 保存模型和关键恢复点。

数据输入 token、有效预测位置、optimizer step、epoch 是不同计量单位。比较模型前，先确认 Tokenizer、验证集与预算口径是否一致。
