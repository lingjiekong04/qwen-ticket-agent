# 训练配方与外部资源

本仓库包含训练配方、构建/校验源码、训练数据元信息和最终运行时；未打包模型、适配器权重、优化器或完整训练数据。

## 已完成的 retention v1.2 训练

| 项目 | 配方 |
|---|---|
| 基座 | Qwen2.5-7B-Instruct |
| 方法 | 在已有 Hermes 工具调用 LoRA 上继续 SFT，不是随机新建 LoRA |
| 数据 | 1800 条、900 中英配对、5202 个 assistant 回合 |
| 主题 | 知识链、权限、澄清、事实忠实、优先级、能力保持 |
| LoRA | rank 8、alpha 16、dropout 0.05，7 类线性投影 |
| 训练 | BF16、1 epoch、225 步、学习率 1e-5、cosine、warmup 0.05 |
| 有效 batch | 单设备 batch 1 × 累积 8 |
| 框架 | LLaMA-Factory 0.9.3，Python 3.11 |

原配置 `configs/ticket_policy_retention_train_v1_2.yaml` 实际为 YAML 兼容 JSON，保留原字节。训练与初始化元信息位于 `results/training/`。历史配置含原机器路径，不能直接在任意机器使用。

训练元信息注明：1800 条中有 1762 条唯一公开记录、38 条重复增量，不应宣传完全去重。完整样本未随审查包提供，本次不能直接确认其内容分发边界。

## 外部交付清单

| 资源 | 原项目位置/身份 | 大小 | 建议处理 |
|---|---|---:|---|
| 基础模型 | Qwen/Qwen2.5-7B-Instruct | 多 GB，4 分片 | 引用上游，核对本项目逐文件哈希 |
| 最终 LoRA | runs/ticket-policy-retention-sft-v1.2-gpu2/train/adapter | 权重约 77.05 MiB | HF 模型仓库，附卡片、固定版本和 SHA256 |
| 初始化 LoRA | runs/ticket-sft-v3-hermes-r8-v1.1/train/adapter | 权重约 77.05 MiB | 另一个 HF 版本，精确续训复现必需 |
| train.json | data/ticket_policy_retention_train_release_v1_1/train.json | 约 10.23 MiB | 审核后 HF Dataset，固定 revision 和下载校验 |
| 已消耗 v6 和两臂轨迹 | data/...holdout_v6 与 results/v6 | 数 MiB | Git 直接保存，保留字节/哈希 |
| optimizer.pt 等状态 | 历次 checkpoint | 百 MB 级 | 私有留存，通常不必公开 |

最终适配器 SHA256：

```text
1837ddf22cce6a614c360ac86e0c1d48b261334db4712ac10b7810472e4cb625
```

初始化适配器 SHA256：

```text
789144f12d35807f193974a8b753ada3e18327da64557e5551f96815553cb802
```

这两个权重和完整数据尚未提供公开下载地址。当前可独立复现 **CPU 演示闭环与冻结评分**；实际模型演示需匹配最终 LoRA，完整训练还需要初始化 LoRA、完整训练数据及历史准入产物。不能写成“克隆即可完整重训”。

公开目录保留了筛选后的训练源码，但历史 source gate 还引用未纳入的设计报告、旧训练阶段等前置产物。这些是研究档案，不是即用的一键训练入口。

初始化阶段涉及 Hermes 数据，上游数据卡许可不代替核验本项目具体转换、混合、来源和分发范围。完整数据发布前，须检查真实样本和来源记录。

本方案不使用 Git LFS：权重放专门模型仓库，Git 保持轻量。后续有真实外链后，再补充固定 revision、下载脚本及下载后 SHA256 校验；这里没有编造权重地址。
