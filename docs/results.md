# 本次运行记录

2026-10-04，起始 commit：a483269096538609f348cc04a0f23349edbba345。

| 检查 | 本次结果 | 证据 |
|---|---|---|
| GPU 与环境 | PyTorch 2.7.1+cu126，四张 H20，BF16 可用 | logs/bootstrap.log、envs/train.lock.txt |
| CPU 单元与集成测试 | 35 passed，37.50s | logs/tests.log |
| HF 导出对照 | FP32 logits、loss、梯度通过 | tests/unit/model/test_huggingface_export.py |
| DDP 尾部更新 | 两个真实 CPU rank 对照通过，包括空 rank | tests/integration/test_ddp.py |
| M0 tiny overfit | 第 223 step 达到 loss < 0.05；准确率 100% | runs/m00_tiny_overfit/summary.json |
| ModelScope | 私有仓库已创建，根 README 和阶段说明已同步 | [模型仓库](https://modelscope.cn/models/GaiWeiqi/SmallLLMPreTrain-English) |

证据路径相对于远程 `/diff/gaiwq/llm_pretrain`。Tiny overfit 不衡量泛化，当前还没有通过聊天验收的模型。M1、正式预训练、SFT 和能力评测仍待完成。
