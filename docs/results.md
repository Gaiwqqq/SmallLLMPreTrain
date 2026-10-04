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

证据路径相对于远程 `/diff/gaiwq/llm_pretrain`。Tiny overfit 不衡量泛化，当前还没有通过聊天验收的模型。M1 与历史 loss 4.153138 的差异来自本次不同的抽样分片和运行栈，未使用历史验证集。99M 对照正在四卡执行；正式预训练、正式 SFT 和能力评测仍待完成。

## 自动阶段进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| engineering-checks | complete | /diff/gaiwq/llm_pretrain/logs/engineering-checks.log |
| lint-checks | complete | /diff/gaiwq/llm_pretrain/logs/lint-checks.log |
| format-checks | complete | /diff/gaiwq/llm_pretrain/logs/format-checks.log |
| download-cosmopedia-pilot | complete | /diff/gaiwq/llm_pretrain/logs/download-cosmopedia-pilot.log |
| download-tinystories-pilot | complete | /diff/gaiwq/llm_pretrain/logs/download-tinystories-pilot.log |
