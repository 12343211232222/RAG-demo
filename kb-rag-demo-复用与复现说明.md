# kb-rag-demo 复用与复现说明文档

> 本说明对应项目目录：`kb-rag-demo/`（内部技术资料 RAG 演示）。
> 项目内 `docs/` 与 `updates/` 下全部文档均为**模拟材料**；程序生成的回答标注【模拟生成】，
> 仅演示流程，不能证明真实大模型的回答质量。

## 1. 复用库说明（来源、版本、复用范围、本人完成部分）

**本项目未复用任何第三方库**，运行仅依赖 Python 3.13 标准库，因此无需密钥、无需联网、无需安装依赖。

| 复用项 | 来源 | 版本 | 复用范围 |
|---|---|---|---|
| Python 标准库 `argparse / hashlib / json / math / re / pathlib` | Python 官方 | 3.13.5 | CLI 解析、内容哈希（幂等去重）、存储序列化、BM25 数学计算、分词正则、文件读写 |

本人独立完成的部分（`kb-rag-demo/kb.py`，约 400 行）：

- Markdown 头部解析与按段分块（片段编号 `doc@version#pN`，保留标识/来源/版本/状态/段落对应关系）；
- 幂等导入（doc_id+version 内容哈希去重）、版本更新（旧版自动 deprecated）、撤回（revoked 不可检索不可引用）；
- BM25 检索（中文按字 bigram、编号/错误码整体成词加权）及无需密钥的哈希向量 / 混合检索（对照实验用）；
- 上下文构造与模板化生成（阈值判"资料不足"、多来源判"冲突"并列呈现、逐句引用片段编号）；
- 评测脚本（14 题评测集、调优/保留集划分、四项指标、幂等校验、对照实验、REPORT.md 自动生成）；
- `docs/` 16 份模拟演示文档与 `evalset.json` 评测集均为本人编写。

## 2. 代码结构、复用说明及可复现

```
kb-rag-demo/
├── kb.py            # 全部逻辑：解析分块 / 存储 / 检索 / 生成 / 评测（纯标准库）
├── evalset.json     # 14 题评测集（调优集 8 + 保留集 6，含金标与更新/撤回动作）
├── docs/            # 15 份模拟文档（多版本、冲突、同名术语、跨段条件、重复内容）
├── updates/         # 1 份模拟文档（deploy-guide v3.0，用于"更新生效"场景）
├── README.md        # 使用说明与设计要点
└── REPORT.md        # 评测报告（运行 evaluate 自动生成，含对照实验与失败案例）
```

**可复现**：代码为确定性实现（无随机、无外部服务），在任意装有 Python 3.10+ 的机器上
执行下文命令即可得到与 `REPORT.md` 一致的结果（评测在内存中建库，不影响本地数据）。

## 3. 启动命令；测试命令

启动（导入并交互式问答）：

```bash
cd kb-rag-demo
python kb.py import-all                                   # 导入全部文档（幂等）
python kb.py list                                         # 查看文档/版本/状态
python kb.py ask "E1002 错误码表示什么？"                  # 正常问答
python kb.py ask "密码策略 1.0 版中，密码有效期是多少天？" --doc password-policy --version 1.0
python kb.py import updates/deploy-guide_v3.0.md          # 更新：新版本生效
python kb.py revoke data-retention                        # 撤回：不再作为依据
```

测试（完整评测 + 对照实验，自动重新生成 REPORT.md）：

```bash
cd kb-rag-demo
python kb.py evaluate
```
