# 运行与网络

## 本阶段学什么

独立环境与反向隧道。

## 输入与阶段成果

环境、版本锁定、代理检查、四卡识别和目录报告。

## 设计与可执行步骤

所有命令先执行：

```bash
cd /diff/gaiwq/llm_pretrain/repo
source scripts/remote/environment.sh
bash scripts/remote/bootstrap.sh
python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

远程目录分为 repo、envs、cache、tmp、data、runs、exports、logs、tools。Conda、pip、Hugging Face、Torch、Triton、CUDA 和 uv 的缓存位置均显式设置，避免系统盘剩余空间不足。

本机 7890 已运行代理，SSH 反向转发将其映射为远程 127.0.0.1:17890：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:17890:127.0.0.1:7890 ubuntu@119.91.56.68
```

本次执行使用 Paramiko 建立相同的反向转发；连接需维持，下载中断可续传。训练只读取完整本地分片，不依赖隧道存活。

下载验证：`curl --fail --head https://huggingface.co`。四张 H20 各约 96GB，卡间 NVLink；/diff 是网络存储，需测吞吐，不能称为 NVMe。通过后将环境依赖写入 envs/train.lock.txt。

故障排查：连接拒绝检查反向隧道；CUDA 不可用检查是否使用 envs/train/bin/python；环境冲突使用隔离环境，禁止修改其他项目环境。

首周自动执行入口：

```bash
nohup python scripts/remote/week.py > "$PRETRAIN_ROOT/logs/week.log" 2>&1 < /dev/null &
nohup dummym-publish --docs --watch > "$PRETRAIN_ROOT/logs/modelscope.log" 2>&1 < /dev/null &
```

`workflow.json` 记录阶段、PID、命令和状态；`docs/results.md` 随阶段同步。执行器加文件锁，避免重复启动。重启时跳过完成阶段，训练从完整 checkpoint 恢复；数据/Tokenizer 若留下不完整目录，先审阅日志并保留失败目录再重做。推理与辅助评测分别使用 envs/inference、envs/evaluation。

## 本次结果

远程隔离环境已安装并锁定，PyTorch 2.7.1+cu126 识别四张 H20；反向隧道已用于环境和数据下载。产物均位于指定根目录。 实测证据见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
