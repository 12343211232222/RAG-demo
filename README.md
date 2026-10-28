# kb-rag-demo —— 内部技术资料 RAG 演示

面向“员工根据内部技术资料提问”的场景，演示资料存在**旧版本、同名术语、跨段条件、冲突内容**时的
完整 RAG 链路：导入 → 分块 → 检索 → 上下文构造 → 生成引用 → 更新与撤回。

> **声明**：`docs/` 与 `updates/` 下全部文档均为**模拟材料**；回答由模板化摘录生成并标注
> 【模拟生成】，仅演示流程，**不能证明真实大模型的回答质量**。
> 纯 Python 3 标准库实现，**无需任何密钥或外部服务**。

## 快速开始

```bash
python kb.py import-all                 # 导入 docs/ 全部文档（幂等：重复执行不增加结果）
python kb.py list                       # 查看文档标识/版本/状态/来源与片段数
python kb.py ask "E1002 错误码表示什么？"
python kb.py ask "密码策略 1.0 版中，密码有效期是多少天？" --doc password-policy --version 1.0   # 指定旧版本
python kb.py ask "公司食堂午餐几点开始供应？"        # 证据不足 -> 如实回答
python kb.py ask "接口超时时间是多少？"              # 同名术语/冲突 -> 并列多来源
python kb.py import updates/deploy-guide_v3.0.md    # 更新：新版本生效，旧版自动 deprecated
python kb.py revoke data-retention                  # 撤回：之后不再作为有效依据
python kb.py evaluate                   # 运行 14 题评测 + 对照实验，生成 REPORT.md
python kb.py serve --port 8765          # 启动 Web 演示页：http://127.0.0.1:8765
```

Web 页面支持上传 `.md/.txt` 文档并选择解析算法（按段落 / 按章节 / 定长滑窗 500 字 /
**语义分块**），无头部的文本自动补全元数据；同一文档再次上传视为**更新**（自动升版本、
旧版转 deprecated），相同内容重复上传**幂等跳过**。

语义分块使用真实嵌入模型（OpenAI 兼容接口），通过环境变量配置，不内置密钥：

```bash
$env:KB_EMBED_BASE_URL="https://api.openai.com/v1"   # 或本地 Ollama: http://127.0.0.1:11434/v1
$env:KB_EMBED_API_KEY="sk-..."                        # 本地服务可任意填
$env:KB_EMBED_MODEL="text-embedding-3-small"          # 或 bge-m3 等
$env:KB_SEMANTIC_THRESHOLD="0.5"                      # 相邻段落余弦相似度切分阈值（可选）
python kb.py serve --port 8765
```

## 设计要点

- **分块与溯源**：按空行分段，标题作为上下文前缀；每个片段编号为 `doc_id@version#pN`，
  保留文档标识、来源、版本、状态与段落序号，可定位回原文。
- **幂等导入**：按 `doc_id + version` 的内容哈希去重；同版本重复导入跳过，同版本不同内容拒绝导入。
- **版本与状态**：默认每族只检索最新 `active` 版本；`--version` 可查 `deprecated` 旧版；
  `revoked` 版本永不参与检索与引用（显式指定也会拒绝）。
- **检索**：BM25（中文按字 bigram、编号/错误码整体成词并加权）；另提供 `--method vector|hybrid`
  （确定性哈希向量，无需密钥，仅作对照）。
- **证据进入上下文**：top-4 片段全文直接拼入；top1 低于阈值 → 声明“资料不足”；
  top-k 中出现多个高分来源 → 判为冲突并并列呈现，不擅自裁决。
- **不使用重排序**：语料小，单阶段 BM25 + 分数阈值 + 同族聚合已稳定（详见 REPORT.md 第 1 节）。

## 评测

- `evalset.json`：14 题，调优集 8 / 保留集 6，覆盖正常、证据不足、指定版本、冲突、更新、撤回、
  同名术语、重复内容八种题型；每题记录**检索命中、证据完整性、回答支持性、引用正确性**四项指标。
- `python kb.py evaluate` 输出分题结果、分组均值、幂等性校验，以及对照实验
  （单一变量：BM25 → 混合检索），失败案例如实记录，全部写入 [REPORT.md](REPORT.md)。
# RAG-demo
