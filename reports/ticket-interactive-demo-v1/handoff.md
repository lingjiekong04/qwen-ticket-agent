# 本地 Qwen 工单助手交互演示

运行组合：Qwen2.5-7B-Instruct + 冻结 retention v1.2 LoRA + v1.7 守卫运行时。
模拟数据使用新的 DEMO 实体；没有加载开发集、v3–v6 题目或答案。

## 1. 上传与 CPU 验证

把 `lora_sft_ticket_interactive_demo_v1.zip` 上传到项目目录，在服务器执行：

```bash
(
set -e
cd /konglingjie/sft_lora/lora_sft_toolcall_day1
unzip -n lora_sft_ticket_interactive_demo_v1.zip
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=/konglingjie/sft_lora/envs/toolcall-sft-py311/bin/python
"$PY" -B scripts/run_ticket_interactive_demo_v1.py --stage verify --server
"$PY" -B scripts/run_ticket_interactive_demo_v1.py --stage smoke
)
```

`verify --server` 用 CPU 校验源文件、框架版本、基座分片、tokenizer 和 LoRA 哈希，可能需要读取十几 GB 权重；不加载模型。
`smoke` 用显式脚本 provider 验证界面传输、8 个工具、连续创建、状态保持、重试与恢复。

## 2. 前台启动真实本地推理

```bash
(
set -e
cd /konglingjie/sft_lora/lora_sft_toolcall_day1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK NODE_RANK
unset ACCELERATE_USE_CPU ACCELERATE_MIXED_PRECISION
PY=/konglingjie/sft_lora/envs/toolcall-sft-py311/bin/python
"$PY" -u -B scripts/run_ticket_interactive_demo_v1.py \
  --stage serve --provider local-qwen --port 8765
)
```

等待输出 `"ready": true`。模型只加载一次，后续请求复用 GPU0；Ctrl+C 停止进程并释放显存。
若 GPU0 已被其他进程占用，启动会停止，不会终止其他进程。端口冲突会在加载模型前报错。

## 3. 从自己的电脑打开

服务只绑定服务器 `127.0.0.1`。使用你平时 SSH 连接服务器的同一地址、账号与端口建立转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 -p <SSH端口> <账号>@<服务器SSH地址>
```

然后在自己的浏览器打开 `http://127.0.0.1:8765`。
容器主机名不一定是公网 SSH 地址；这里使用服务器供应商提供的实际 SSH 连接信息。
若平台终端不支持 SSH 转发，需要使用平台提供的到容器本地端口的隧道功能；不要把应用改绑公网。

## 演示顺序

1. 页面应显示“本地 Qwen · GPU0”。“CPU 脚本验证模式”表示没有加载模型。
2. 读取 `DEMO-T101`，确认初始状态 `open`、版本 `1`。
3. 更新为 `in_progress`，勾选参数确认和模拟写入授权，执行后检查状态与版本。
4. 分配给 `DEMO-A02`，再读同一工单查看保留的变化。
5. 连续创建两张工单，检查生成不同 ID `N1000`、`N1001`。
6. 搜索 `VPN`、`DEMO-U01`、`open`，以及查询 `lin@demo.invalid`。
7. 搜索知识库 `VPN`，查看 `search_kb` 和 `get_kb_article` 证据。
8. 切换安全停机，验证后续请求停止；重启进程才能恢复。

每条请求由自然语言说明和用户确认的结构化参数构成。应用将完整契约附加到模型可见请求中，
模型负责工具提议，守卫负责范围、事务、恢复与证据答复。此版本没有自由文本意图解析器；
目标冲突时以页面中用户确认的结构化契约为准，页面可展开查看本次可见契约。

同一浏览器会话中的模拟工单连续保留，新会话独立。模型不接收旧对话聊天历史，当前数据库状态持续保留。
接口为同步请求，页面完成后展示全部模型轮次与工具事件；本版本不逐 token 流式输出。
每会话最多 100 个请求，进程最多 64 个会话。重复请求键只回放原结果；进程重启后内存状态重置。

## 证据与界限

每次请求保存原始模型输出、守卫决定、工具响应、前后状态、变化和哈希到
`runs/ticket-interactive-demo-v1/session-*.jsonl`，不会覆盖历史日志。
回滚开关在当前请求结束后阻止后续执行，不撤销已经提交的模拟变化。
网络异常时页面保留请求键，重试会复用原内容；参数校验拒绝时可直接修正后重新执行。

CPU 测试验证接入实现；真实 Qwen 推理需在服务器启动后按上述演示顺序验证。
现有 v6 正式成绩保持冻结，交互演示不产生新的最终验收成绩。

