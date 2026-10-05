# 运行与网络：把训练的“工作台”准备好

我们先解决环境和文件位置，再讨论模型。否则，一个下载失败、系统盘写满或错用 Python 环境，都可能被误认为训练代码有问题。

## 哪些工作在本机，哪些在训练机？

本机负责审阅源码和提供 7890 代理；训练机负责下载、数据处理、训练与保存。所有远程产物都归属 `/diff/gaiwq/llm_pretrain`。

| 子目录 | 放什么 | 为什么分开 |
|---|---|---|
| repo | 源码和 README | Git 只记录可审阅的代码与说明 |
| envs | Python、训练/评测/推理环境 | 各框架升级互不干扰 |
| cache、tmp | 下载缓存、编译缓存和临时文件 | 防止默认写入空间较小的系统盘 |
| data | 原始文本、清洗文本、Tokenizer、编码数据 | 能追溯每一步的输入输出 |
| runs | checkpoint、日志和运行摘要 | 保留训练现场，不塞进 Git |
| exports | 标准模型文件 | 供 Transformers、TRL、vLLM 加载 |
| logs、tools | 阶段日志和环境工具 | 出错时定位具体步骤 |

四张 H20 各约 96GB，有 NVLink。GPU 主要负责模型的矩阵计算；下载、清洗、去重和 Tokenizer 编码主要在 CPU 上进行。数据还没准备好时，强行让 GPU 做重复训练并不会增加有效数据。

`/diff` 是网络存储，不能当作本地 NVMe 描述；文件读取、保存也需要计入总体耗时。

## 反向隧道到底做了什么？

训练机访问 `127.0.0.1:17890` 时，请求通过 SSH 回到本机，再交给本机 `127.0.0.1:7890` 的代理。这是端口转发，不是把本机数据盘挂到远程。

本机建立连接的标准命令：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:17890:127.0.0.1:7890 ubuntu@119.91.56.68
```

本次自动执行使用 Paramiko 实现相同转发。隧道需要保持在线供下载和上传使用；模型训练只读取已经完整下载的本地文件，训练计算本身不经过代理。

## 为什么环境分成三个？

训练依赖固定版本，让恢复实验更容易一致。推理框架可能要求另一个 PyTorch/CUDA 组合，辅助评测也有自己的依赖；因此分别放在 `envs/train`、`envs/inference`、`envs/evaluation`。成熟框架帮助训练和部署，自写模型仍用于理解数学过程。

远程执行命令前先设置环境：

```bash
cd /diff/gaiwq/llm_pretrain/repo
source scripts/remote/environment.sh
python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

首次安装才执行 `bash scripts/remote/bootstrap.sh`。不要为了阅读文档重新安装正在使用的环境。`environment.sh` 不只是“激活 Python”，还设置了代理、缓存和临时文件路径。实际包版本保存在 `envs/train.lock.txt`。

## 长任务为什么交给控制器？

`week.py` 把每个阶段的命令明确列出来；完成状态、PID 和日志位置写入 `workflow.json`。`nohup` 让任务不依赖一个一直开着的终端，文件锁阻止同一控制器启动两份。

```bash
# 当前已经有任务运行；下面是首次启动方式，不需要重复执行。
nohup python scripts/remote/week.py > "$PRETRAIN_ROOT/logs/week.log" 2>&1 < /dev/null &
nohup dummym-publish --docs --watch > "$PRETRAIN_ROOT/logs/modelscope.log" 2>&1 < /dev/null &
```

重启控制器会跳过完成阶段、复用性能选择，并在确认 PID、命令和工作目录属于本工程后接管现有数据任务。训练从完整 checkpoint 恢复，不从一次算到一半的梯度恢复。

上传是独立任务。先保存完整目录，再写 `ready.json`，上传器只处理带标记的产物；上传失败不会要求模型重做已经完成的训练。

## 如何判断这一阶段通过？

环境能导入 PyTorch；识别四张卡；代理能访问下载源；产物落在指定根目录；版本可追踪。本次这些检查已通过，详见 [运行记录](results.md)。

遇到问题先按范围排查：连不上下载源看隧道和代理；找不到 GPU 看 Python 环境与设备编号；恢复失败看配置和数据指纹。下一阶段阅读 [模型原理](01_foundations.md)。
