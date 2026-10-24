# 第三方来源与许可边界

原创代码选择 Apache License 2.0（见 LICENSE）。第三方组件、模型与数据保留各自许可，此选择不将全部资源自动归为项目原创。

| 来源 | 用途 | 处理 |
|---|---|---|
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 基础模型，官方 LICENSE 标注 Apache-2.0 | 不分发基础权重，使用时保留上游声明 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 训练/推理，使用 0.9.3 | 不复制完整框架，third_party/ 保留原包所附许可证 |
| [Hermes function calling v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 历史工具调用初始化数据来源，上游数据卡标注 Apache-2.0 | 不分发其原始数据或完整混合训练集 |
| PyTorch、Transformers、PEFT、jsonschema 等 | 运行依赖 | 通过包管理器安装，遵守各自许可 |

模型和数据页面于 2026-10-24 核对。上游 main 可能更新，精确复现以本项目哈希为准。模型/数据并不随本代码 LICENSE 自动重新授权。

原适配器自动卡片含 `license: other`、本地路径与未完成字段，因此不直接作为公开模型卡。发布最终与初始化 LoRA 前，须补齐来源、训练数据、适用范围及准确许可，不能只上传旧自动卡片。
