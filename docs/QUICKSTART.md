# 启动与操作

## CPU 验证

按照根 README 执行 verify、smoke、公开入口测试和冻结评分复算。这些命令只需要 jsonschema，不加载模型、不初始化 CUDA、不调用真实工具。

Windows 请把仓库放在较短的路径中，例如 `C:\work\ticket-agent`。本次验证发现，深层目录下生成会话审计文件可能超过 Windows 路径长度限制；同一交付包解压到短路径后，22 项测试全部通过。

## 本地推理环境

真实模型 provider 保留服务器准入条件：Linux/Python 3.11、物理 GPU0 A100、至少 30000 MiB 空闲显存，GPU0 无其他计算进程。没有适配其他 GPU、量化或 CPU 模型推理。

以下为按既有版本整理的安装说明，本次没有在新的 Linux GPU 环境重装验证。CPU 检查不需要这些依赖：

```bash
python3.11 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c constraints.txt 'llamafactory==0.9.3' 'jsonschema==4.25.0'
python -m pip check
```

准备文件后，先核对版本和哈希：

```bash
CUDA_VISIBLE_DEVICES='' python -B scripts/run_public_demo.py --stage verify --server \
  --model-dir /absolute/path/to/Qwen2.5-7B-Instruct \
  --adapter-dir /absolute/path/to/retention-v1.2-adapter
```

之后按 README 的 local-qwen 命令启动。必须匹配原 tokenizer、配置和权重；任意最新上游快照不能代替核验。所有固定 SHA256 位于 `configs/ticket_interactive_demo_v1.json`，路径包装器不修改这个冻结文件，也不放宽哈希或后端限制。

## 通过 SSH 访问

服务只绑定 `127.0.0.1:8765`。保持服务器进程运行，在自己电脑另开终端：

```bash
ssh -N -L 8765:127.0.0.1:8765 -p YOUR_SSH_PORT YOUR_USER@YOUR_SSH_HOST
```

本机浏览器打开 `http://127.0.0.1:8765`。不要把个人 IP、端口、账号或密钥写进公开仓库。

## 演示顺序

1. 读取 DEMO-T101，观察 open、版本 1。
2. 更新为 in_progress，再次读取，检查版本递增。
3. 分配给 DEMO-A02，检查只改变负责人及版本。
4. 为 DEMO-U01 连续创建两个标题不同的 VPN 工单，检查新 ID 唯一。
5. 按 VPN、DEMO-U01、open 搜索，检查结果和条件匹配。
6. 查询 lin@demo.invalid，搜索 VPN 知识并读取正文证据。
7. 展开模型原文、工具提议、守卫决定及工单差异。

## 会话、日志与回滚

会话隔离，数据在当前服务进程内存中。重启不保留工单，新建会话回到 fixture。安全停机阻止后续请求，当前请求会先完成；它不撤销已经提交的数据。恢复需重新启动演示进程。

日志写入 Git 忽略的 `runs/ticket-interactive-demo-v1/`，可能含用户输入、会话 ID 和模型输出，不应直接公开。现有交互日志汇总没有对安全停机行为作出日志证明。
