# GitHub 上传范围

只从此候选目录初始化新仓库，不在原 11.328 GiB 实验目录执行 `git add .`。

| 分类 | 本项目内容 | 决定 |
|---|---|---|
| 必传 | README、docs、LICENSE、第三方说明、依赖、.gitignore/.gitattributes | 已纳入 |
| 必传 | 完整演示依赖链与 v1.7、冻结绑定 | 已纳入，不能只取最后一个 core.py |
| 必传 | 最终训练配方、构建/校验代码 | 筛选保留，前置产物不全已说明 |
| 建议传 | 已消耗 v6 原始合成轨迹、评分、题目与 AI 复核摘要 | 已纳入，不再称其为未来盲测 |
| 建议传 | 训练指标、来源哈希、实际演示汇总 | 已纳入 |
| 外链 | 最终/初始化 LoRA、完整训练数据 | 待发布 HF 资源 |
| 不传 | 原 runs 全目录、optimizer、缓存、checkpoints | 排除 |
| 不传 | zip/tar.gz 执行包、完整上游框架副本 | 排除，仅保留第三方许可 |
| 不传 | .env.local、密钥、SSH 信息、真实用户日志 | 排除；库存中上游 .env.local 未读取或纳入 |
| 私有归档 | 审查原包、1197 文件快照、原交互会话日志 | 留在候选目录之外 |

`publication/source-provenance.json` 逐文件记录复制来源、大小、SHA256，复制内容不变。新增入口、文档和清单不冒称旧冻结文件；`publication/files-sha256.json` 校验当前候选内容。绝对服务器路径作为历史配置元信息保留，不包含可登录凭据。

## 上传命令

仅在本候选根目录执行。将 YOUR_NEW_GITHUB_REPOSITORY_URL 换成自己新建的空仓库地址：

```bash
python -B scripts/verify_publication.py
python -B scripts/run_public_demo.py --stage smoke
python -B scripts/verify_frozen_results.py
git init -b main
git config core.autocrlf false
git add --all
git diff --cached --stat
git diff --cached --name-only
# 确认暂存列表后再提交
git commit -m "Release local Qwen ticket assistant with frozen evaluation evidence"
git remote add origin YOUR_NEW_GITHUB_REPOSITORY_URL
git push -u origin main
```

本次整理没有执行 Git 初始化、提交、远程配置或 push。没有接入真实工单后端。

## 安全检查边界

审查包的文本扫描覆盖私钥、GitHub/HF/OpenAI token 和云访问标识模式，未检出匹配项；仍需在发布最终权重和完整数据时核验真实内容。模式扫描不是绝无秘密的证明。原始交互日志、服务器授权文件和大权重未随候选发布。
