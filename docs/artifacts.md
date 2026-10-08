# 交付清单：从代码找到模型、恢复点和逐题证据

本轮已完成训练和最终语义审阅，正式模型通过 15/120（12.5%），未达到可靠交流目标。先读 [最终报告](final_delivery.md)，再按下面的路径检查产物；上传成功不代表能力达标。

## GitHub：代码、说明与可审阅的小型结果

目标仓库：[Gaiwqqq/SmallLLMPreTrain](https://github.com/Gaiwqqq/SmallLLMPreTrain)，分支 `main`。

| 内容 | 仓库位置 | 阅读目的 |
|---|---|---|
| 总入口 | `README.md` | 当前结论和阅读顺序 |
| 初学者说明 | `docs/reading_guide.md`、`docs/00_*.md` 至 `docs/07_*.md` | 学习数据、模型、并行、预训练、SFT 和验收 |
| 原生模型与训练 | `src/dummym/` | 查看核心算法、数据约定和训练实现 |
| 运行及复现入口 | `scripts/`、`configs/`、`requirements/` | 找到配置、控制器与依赖版本 |
| 实验原因与结果 | `experiments/README.md`、`experiments/current/` | 理解每次实验为什么做、结果支持什么 |
| 冻结题集 | `evaluation/chat_test_v2.jsonl` 与 manifest | 确认最终 120 项任务及其哈希 |
| 逐题原始回答 | `evaluation/reviews/formal_v2/fresh-chat-test.jsonl` | 直接检查模型说了什么 |
| 逐题判定与政策 | `evaluation/reviews/formal_v2/final-reviews.jsonl`、`review-method.json` | 复核每一项为什么通过或失败 |
| 聚合报告 | `evaluation/reviews/formal_v2/final-report.json` | 检查六类分数、accepted 与输入哈希 |
| 开发集对照 | `evaluation/reviews/formal_v2/general-*` | 单独查看 40 项 dev，不混入最终 test |

这些回复来自冻结的合成题集，属于可审阅的实验结果。原始大语料、训练环境、缓存、日志、凭据和大型 checkpoint 不进入 Git。`.gitignore` 保持这些边界。发布以远端 `main` 为父提交，并检查分支头是否变化，不强制覆盖远端历史。

## ModelScope：权重、恢复点与归档证据

项目：[GaiWeiqi/SmallLLMPreTrain-English](https://modelscope.cn/models/GaiWeiqi/SmallLLMPreTrain-English)，目前为私有，需要有权限的账号访问。

| 云端目录 | 产物 | 用途与边界 |
|---|---|---|
| `exports/base/` | 213M 预训练 Base 与 Tokenizer | 文章续写、检查预训练基础能力 |
| `exports/chat/` | 首轮正式 Chat | 首轮 SFT 结果，尚未通过可靠聊天验收 |
| `exports/sft_curriculum_lr3e-5/`、`exports/sft_curriculum_lr1e-4/` | 两组 v1 课程短跑 | 对照学习率；一般回复存在退化 |
| `exports/curriculum-v2/` | 四卡纠正实验候选 | 160K 会话，一轮；一般 dev 25% |
| `exports/curriculum-v2-formal/` | 最终正式模型、Tokenizer、模型卡 | 160K 会话，两轮、5,000 步；最终 test 12.5% |
| `checkpoints/` | 预训练和 SFT 关键恢复点 | 含训练状态；格式与恢复条件以各目录 README/ready 为准 |
| `reports/first-round/` | 首轮 120 题原始回复、理由与汇总 | 首轮 test 38.3%，不能与新 test 直接比较 |
| `reports/final-v2/` | 最终 120+40 项回复、判定、报告与哈希 | 完整验收证据；正式导出权重与生成模型的哈希已核对 |

正式 SFT 最后恢复点位于 `checkpoints/sft_curriculum_v2_formal/trainer-milestones/step-005000/`。推理导出仅供加载模型；恢复训练应使用完整 Trainer 快照及对应数据、配置和 world size，不能用推理权重冒充完整恢复点。

项目根目录也同步教学 README、文档和关联源码。GitHub 是代码版本入口，ModelScope 是大型训练产物入口；通过阶段名、配置、模型/题集哈希及结果文件交叉定位。

## 什么已经验证，什么需要本次上传？

2026-10-07 已实际完成最终报告、模型卡与文档的 ModelScope 归档；远程控制器写入 `completed_acceptance_failed` 和 `semantic_review_complete=true`。Git 本地保存最终验收提交 `c1dea7f`。

2026-10-08 用户授权整理后发布 GitHub，并继续同步权重与文档。本次整理修正了过时的运行状态和导航，补齐阶段产物、逐题证据和模型路径的对应关系。发布采用正常 Git 提交与推送；保留本地从 `a483269` 起的完整实验与验收历史，不压成只有最终代码的快照，也不强制覆盖远端历史。

ModelScope 已有权重与完整验收报告，本次同步整理后的教学材料。上传按文件清单和哈希核对；仅补传确实缺失的模型或恢复点，不重复训练、重生成评测或无条件重传大权重。
