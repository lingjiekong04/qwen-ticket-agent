# 本地 Qwen 工单助手

基于 **Qwen2.5-7B-Instruct 和 LoRA 微调**的本地工单助手，支持工单查询、创建、分配、状态更新及知识库检索。通过执行前校验、故障恢复和基于后端证据的答复，降低错误操作风险，并展示完整的工具调用与工单变化过程。

项目包含数据构建、训练配置、对照评测和交互演示，当前使用会话隔离的模拟工单后端。完整训练所需的权重、数据和部分历史前置产物尚未全部公开，详见[复现与资源说明](docs/TRAINING_AND_ASSETS.md)。

## 技术方案与版本含义

| 组成 | 当前采用版本 | 负责什么 |
|---|---|---|
| 基础模型 | Qwen2.5-7B-Instruct | 理解请求并提出工具调用 |
| 微调适配器 | retention v1.2 LoRA | 项目训练的工单适配器，兼顾工单任务与已有工具调用能力 |
| 工具执行层 | trusted completion v1.7 | 校验调用、执行受控恢复，并在任务完成后释放证据答复 |
| 正式测试集 | 独立 v6 | 覆盖 8 工具及正常/故障分层，验证冻结系统 |

这些版本号分别标识模型、适配器、执行层和测试集。下面的 v1.6/v1.7 对照使用相同模型和 LoRA，比较的是**执行层改进**。

| 微调适配器（模型侧） | 工具执行层（系统侧） |
|---|---|
| 通过 LoRA 训练更新一部分模型参数 | 通过普通代码实现校验、恢复和状态控制 |
| 影响模型如何提出工具调用和生成回答 | 决定提议是否可执行、失败后如何恢复、何时可以答复 |
| 当前冻结为 retention v1.2 | 在同一冻结适配器上逐步迭代到 trusted completion v1.7 |
| 模型效果需要专门对照评测，训练 loss 下降不能替代任务完成率 | 执行层收益可在相同模型权重下进行对照 |

## 已验证成果

| 指标 | v1.6 对照 | v1.7 候选 |
|---|---:|---:|
| 独立 v6 完成数 | 154/160 | **160/160** |
| search_tickets | 16/20 | 20/20 |
| update_ticket_status | 18/20 | 20/20 |
| 其余 6 工具 | 各 20/20 | 各 20/20 |
| 无效输出次数 | 1 | 0 |
| 无额外模型回合的特殊前缀释放 | 0/30 | 30/30 |
| 观察到的安全违规 | 0 | 0 |

两臂使用相同冻结 LoRA，差异来自运行时。**160/160 是这套冻结合成测试上的系统成绩，不代表模型或真实业务永远正确。** 每工具正常与故障路径各 10 题；逐条语义复核为 AI 辅助复核，不冒称人工逐条审核。v6 已消耗并公开，不可再次当作独立盲测。

已有实际演示日志覆盖 8 个工具、10 次完成请求，provider 为 `local-qwen`；本仓库只提供其汇总。CPU 演示验证使用脚本化输出，不能单独证明模型能力。

## 系统结构

```text
用户请求 + 用户确认的结构化参数（形成模型可见的可信契约）
        ↓
本地 Qwen + 冻结 LoRA → 模型工具提议
        ↓
解析器 → 契约守卫 → 精确恢复 → 会话内模拟工单后端
                                      ↓
                             后端证据 → 确定性答复
```

用户需要选择操作并确认参数；当前没有任意自由文本意图提取。最终状态答复由证据渲染器生成。部分故障注入路径可由已有契约直接恢复。

## 项目如何迭代

| 阶段 | 发现的问题 | 采取的改进 |
|---|---|---|
| 数据与微调 | 工具调用和工单回答需要领域适配 | 构建中英多轮工单数据，在已有工具调用 LoRA 上继续 SFT |
| 失败归因 | 低分可能来自动作、回答、评分口径或任务契约 | 拆分调用与状态证据，复核真实失败和任务参数可见性 |
| 可信执行 | 模型可能提出错误目标、身份或额外操作 | 在执行前校验契约、权限和版本，基于实际后端证据回答 |
| 通用恢复 | 错误调用被拦截后，任务仍可能无法完成 | 增加事务、身份、搜索条件和完整创建载荷恢复 |
| 可信完成 | 动作已完成，额外模型回合仍可能出错或循环 | 精确恢复目标，并在证据足够时直接答复 |
| 独立验收与演示 | 开发集改善需要独立验证和可运行交付 | 冻结 v6 对照评测，提供本地模型演示、CPU CI 和可复算证据 |

### 微调适配器：历史版本差在哪里

以下列出影响最终路径的主要版本和修复分支。它们并非每一行都继承上一行；修复分支以 retention v1.2 为共同起点。

| 版本或分支 | 相比前阶段的主要变化 | 在最终项目中的位置 |
|---|---|---|
| Hermes 工具调用 LoRA（r8） | 在通用工具调用数据上训练，建立基础调用能力 | 后续工单 SFT 的初始化来源 |
| ticket-sft-v3（Hermes r8 工单续训） | 在工具调用 LoRA 上继续学习工单任务 | retention 训练使用的直接初始化适配器 |
| retention v1.1 | 使用包含工单策略与能力保持主题的中英混合数据，形成领域与能力保持训练配方 | 较早的 retention 训练版本，未作为当前加载权重 |
| **retention v1.2** | 沿用 v1.1 的主要训练配方，使用独立产物目录，并补强 GPU 绑定、运行环境和训练结果核验 | **最终冻结的 LoRA；后续执行层对照均使用它** |
| repair-lr2p5e-8 / repair-lr5e-8 | 从 retention v1.2 继续低学习率修复，分别比较 2.5e-8、5e-8 | 开发候选，最终配置未采用 |
| answer-contract-lr1e-7 | 从 retention v1.2 继续训练，针对回答格式、字段完整性和证据表达修复 | 曾进入开发对照，最终配置未采用 |

retention v1.1/v1.2 的主要差异属于训练执行与核验工程，不能仅凭版本号推导模型质量提升。后两个修复方向是实验分支，其完整权重与历史评测没有全部纳入本公开快照；它们未替换最终冻结的 retention v1.2。训练配方和交付边界见[训练说明](docs/TRAINING_AND_ASSETS.md)。

### 工具执行层：每次增加了什么

下表记录从基础控制到可信完成的变化。多数旧层仍由新层组合调用，因此仓库保留其实现。

| 版本 | 主要新增能力 | 解决的典型问题 |
|---|---|---|
| controller v1.1 | 基础调用、状态和答复检查 | 建立工具执行与答复约束 |
| contract guard v1 | 绑定可信目标并基于证据答复；最初支持读取与状态更新 | 错误目标、额外状态更新及无证据陈述 |
| contract guard v1.1 | 扩展至分配、创建、用户查询、工单搜索及知识库，共 8 个工具 | 守卫覆盖不完整 |
| transaction recovery v1.2 | 事务拒绝与冲突后的受控恢复 | 错误事务虽未提交，但任务无法继续 |
| blocked action recovery v1.3 | 对错误负责人、身份及扩大搜索条件进行拦截后恢复 | 调用被拦截后陷入重复尝试 |
| general recovery v1.4 | 错误 KB 查询恢复和分配目标重新读取 | 查询偏离可信词、重读恢复不完整 |
| evidence completion v1.5 | 受限 schema 投影、补齐已返回知识条目的正文证据 | 多余参数导致调用失败、正文证据缺失 |
| exact create recovery v1.6 | 按完整可信 title、description、priority、requester_id 恢复创建 | 模型改写标题、描述、优先级或请求人 |
| **trusted completion v1.7** | 精确工单搜索、保持标识符大小写、可信目标重读、完成后立即释放证据 | 搜索与重读循环、动作完成后继续生成出错 |

controller v1.1 与 contract guard v1.1 是两个组件的版本号。正式 v6 比较的是最后两行：相同模型和 LoRA 下，完成数从 154/160 到 160/160；不是把不同测试集的历史分数串成同一条提升曲线。

完整说明见[项目迭代与设计决策](docs/ITERATION_HISTORY.md)，包括执行层各版本、对应故障、验证方法和现有边界。

## 快速验证：无需模型和 GPU

在仓库根目录执行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
python -B scripts/run_public_demo.py --stage verify
python -B scripts/run_public_demo.py --stage smoke
python -B -m unittest discover -s tests -p test_public_entrypoints.py -v
python -B scripts/verify_frozen_results.py
python -B scripts/verify_publication.py
```

仅检查界面和模拟闭环，可显式选择脚本化 provider：

```bash
python -B scripts/run_public_demo.py --stage serve --provider scripted
```

打开 `http://127.0.0.1:8765`。这不代表实际模型推理。

## 实际本地模型演示

需要 Linux、Python 3.11、物理 GPU0 A100，以及匹配的模型、tokenizer 和 LoRA。已验证栈为 torch 2.6.0+cu124、transformers 4.51.3、peft 0.15.2、LLaMA-Factory 0.9.3。

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -B scripts/run_public_demo.py --stage serve --provider local-qwen \
  --model-dir /absolute/path/to/Qwen2.5-7B-Instruct \
  --adapter-dir /absolute/path/to/retention-v1.2-adapter
```

新入口只迁移模型目录，原冻结文件和哈希不变。本仓库没有权重，也尚未提供 LoRA 公开下载地址；没有匹配权重时，只能运行 CPU 检查和现有证据复算。详见 [启动说明](docs/QUICKSTART.md)。

## 目录导航

```text
extensions/       演示、模拟后端、组合运行时及研究代码
scripts/          公开入口、只读证据校验、筛选保留的历史脚本
configs/          原始冻结配置，历史路径仅用于溯源
data/             模拟 fixture、已消耗 v6、训练数据元信息
results/          v6 轨迹及评分、训练元信息、交互日志汇总
reports/          原有复核摘要和离线集成验收记录
docs/             启动、结果解释、训练复现边界、开源范围
publication/      来源映射与公开文件哈希
tests/            公开入口与部分历史研究测试
third_party/      第三方许可证副本
```

历史运行时层是实际依赖，不能只留下最后一个 core.py。研究脚本含原服务器路径和历史前置产物，不应对全部历史测试直接 `unittest discover`，也不要用旧 blind 脚本重跑已消耗盲测。本 README 的命令是支持的独立 CPU 验证入口。

## 文档与许可

- [项目技术报告 PDF](docs/qwen-ticket-agent-project-report.pdf)
- [项目迭代与设计决策](docs/ITERATION_HISTORY.md)
- [启动与演示](docs/QUICKSTART.md)
- [结果和边界](docs/EVALUATION.md)
- [训练配方与权重/数据交付](docs/TRAINING_AND_ASSETS.md)
- [上传与排除清单](docs/PUBLICATION.md)
- [第三方来源](THIRD_PARTY_NOTICES.md)

原创代码选择 Apache-2.0。第三方组件、模型、数据保留各自许可。完整训练样本没有随本候选公开，尚需完成其实际来源和分发核验。
