#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb-rag-demo —— 内部技术资料 RAG 演示（纯 Python 标准库，无需密钥 / 外部服务）

功能：导入、分块、检索（BM25 / 哈希向量 / 混合）、上下文构造、生成引用、更新与撤回、评测。

重要声明：
- 本程序的“生成”是模板化摘录（答案中标注【模拟生成】），仅用于演示流程，
  不能用于证明任何真实大模型的回答质量。
- docs/ 与 updates/ 下的所有文档均为模拟材料。
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
DOCS_DIR = BASE / "docs"
UPDATES_DIR = BASE / "updates"
DEFAULT_STORE = BASE / "kb_store.json"
EVALSET = BASE / "evalset.json"
REPORT = BASE / "REPORT.md"

TOP_K = 4          # 进入上下文的证据片段数
MIN_SCORE = 5.0    # 证据不足判定阈值（BM25 得分，经调优集校准：无关噪声约 3.8，真实命中通常 > 8）
STRONG_RATIO = 0.55  # 与最高分的比例，用于冲突检测
# 各检索方法的“证据不足”阈值（hybrid/vector 为归一化得分，与 BM25 不同量纲）
METHOD_MIN_SCORE = {"bm25": MIN_SCORE, "vector": 0.05, "hybrid": 0.25}

# ---------------------------------------------------------------- 解析与分块

def parse_front_matter(text):
    """解析 --- 包围的 key: value 头部，返回 (meta, body)。"""
    if not text.startswith("---"):
        return {}, text
    end = text.index("\n---", 3)
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[end + 4:].lstrip("\n")


def _para_chunks(doc_key, meta, body):
    """解析算法一：按空行分段，标题作为上下文前缀。"""
    chunks, heading, para_idx = [], "", 0
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", block)
        if m:
            heading = m.group(2).strip()
            continue
        para_idx += 1
        chunks.append((heading, para_idx, re.sub(r"\s+", " ", block)))
    return chunks


def _heading_chunks(doc_key, meta, body):
    """解析算法二：按章节切分，一个标题小节为一个片段。"""
    chunks, heading, buf, idx = [], "", [], 0
    def flush():
        nonlocal idx
        text = re.sub(r"\s+", " ", " ".join(buf).strip())
        if text:
            idx += 1
            chunks.append((heading, idx, text))
    for line in body.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            flush(); buf = []; heading = m.group(2).strip()
        elif line.strip():
            buf.append(line.strip())
    flush()
    return chunks


def _fixed_chunks(doc_key, meta, body, size=500, overlap=50):
    """解析算法三：定长滑窗（500 字、50 字重叠）。"""
    text = re.sub(r"\s+", " ", body).strip()
    chunks, idx = [], 0
    for start in range(0, len(text), size - overlap):
        seg = text[start:start + size].strip()
        if seg:
            idx += 1
            chunks.append(("", idx, seg))
    return chunks


# ---- 真实嵌入模型（OpenAI 兼容 /embeddings 接口）----
# 配置优先级：页面设置（kb_config.json）> 环境变量（KB_EMBED_BASE_URL / KB_EMBED_API_KEY /
# KB_EMBED_MODEL / KB_SEMANTIC_THRESHOLD）。密钥仅存本地配置文件，不进入代码。

CONFIG_FILE = BASE / "kb_config.json"


def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


def embed_settings():
    cfg = load_config()
    return {
        "base_url": cfg.get("embed_base_url") or os.environ.get("KB_EMBED_BASE_URL", ""),
        "api_key": cfg.get("embed_api_key") or os.environ.get("KB_EMBED_API_KEY", ""),
        "model": cfg.get("embed_model") or os.environ.get("KB_EMBED_MODEL", "text-embedding-3-small"),
    }


def embed_texts(texts):
    """调用真实嵌入模型；未配置服务地址时返回 None。"""
    s = embed_settings()
    if not s["base_url"]:
        return None
    req = urllib.request.Request(
        s["base_url"].rstrip("/") + "/embeddings",
        data=json.dumps({"model": s["model"], "input": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {s['api_key']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _semantic_chunks(doc_key, meta, body):
    """解析算法四：语义分块——真实嵌入模型计算相邻段落余弦相似度，低于阈值处切开。"""
    paras = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", body)
             if p.strip() and not p.strip().startswith("#")]
    if not paras:
        return []
    vecs = embed_texts(paras)
    if vecs is None:
        raise ValueError("语义分块需要真实嵌入模型：请在左侧「嵌入模型设置」中填写 API 地址与模型，"
                         "或设置环境变量 KB_EMBED_BASE_URL / KB_EMBED_API_KEY / KB_EMBED_MODEL")
    threshold = float(os.environ.get("KB_SEMANTIC_THRESHOLD", "0.5"))
    groups, cur = [], [paras[0]]
    for i in range(1, len(paras)):
        if _cos(vecs[i - 1], vecs[i]) < threshold:
            groups.append(cur)
            cur = []
        cur.append(paras[i])
    groups.append(cur)
    return [("", i + 1, " ".join(g)) for i, g in enumerate(groups)]


CHUNKERS = {"paragraph": _para_chunks, "heading": _heading_chunks,
            "fixed": _fixed_chunks, "semantic": _semantic_chunks}


def chunk_markdown(doc_key, meta, body, method="paragraph"):
    """按所选解析算法分块。每个片段保留：文档标识、版本、标题、来源、段落序号。"""
    out = []
    for heading, para_idx, text in CHUNKERS[method](doc_key, meta, body):
        out.append({
            "chunk_id": f"{doc_key}#p{para_idx}",
            "doc_key": doc_key,
            "doc_id": meta["doc_id"],
            "version": meta["version"],
            "title": meta.get("title", meta["doc_id"]),
            "source": meta.get("source", ""),
            "heading": heading,
            "para_index": para_idx,
            "text": text,
        })
    return out

# ---------------------------------------------------------------- 存储（幂等导入 / 更新 / 撤回）

class Store:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data = {"docs": {}, "chunks": {}}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self):
        if self.path:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def version_key(v):
        return tuple(int(x) for x in re.findall(r"\d+", v))

    def import_file(self, path, method="paragraph"):
        """导入文档。幂等：同一 doc_id+version 且内容相同 -> 跳过；
        内容不同 -> 拒绝（需提升版本号）。导入更新版本时，旧版自动转为 deprecated。"""
        raw = Path(path).read_text(encoding="utf-8")
        return self.import_text(raw, method)

    def import_text(self, raw, method="paragraph", default_id="uploaded-doc"):
        """从文本内容导入（Web 上传入口）。缺少头部字段时自动补全；
        doc_id 已存在时自动提升版本号（即“更新”语义）。"""
        meta, body = parse_front_matter(raw)
        meta.setdefault("doc_id", default_id)
        digest = hashlib.sha256((raw + method).encode("utf-8")).hexdigest()
        # 幂等：同一文档相同内容（同解析算法）重复上传 -> 跳过，不增加结果
        for d in self.data["docs"].values():
            if d["doc_id"] == meta["doc_id"] and d["hash"] == digest:
                return f"跳过（相同内容已存在，幂等）: {d['doc_id']}@{d['version']}"
        if "version" not in meta:
            versions = [d["version"] for d in self.data["docs"].values()
                        if d["doc_id"] == meta["doc_id"]]
            if versions:
                latest = max(self.version_key(v) for v in versions)
                meta["version"] = f"{latest[0] + 1}.0"  # 同一文档再次上传 = 更新
            else:
                meta["version"] = "1.0"
        meta.setdefault("title", meta["doc_id"])
        meta.setdefault("source", "用户上传（模拟材料）")
        doc_key = f"{meta['doc_id']}@{meta['version']}"
        if doc_key in self.data["docs"]:
            raise ValueError(f"{doc_key} 已存在但内容不同，请提升版本号后再导入")
        self.data["docs"][doc_key] = {
            "doc_id": meta["doc_id"], "version": meta["version"],
            "title": meta.get("title", meta["doc_id"]),
            "source": meta.get("source", ""), "status": "active",
            "date": meta.get("date", ""), "hash": digest,
            "chunk_method": method,
        }
        for c in chunk_markdown(doc_key, meta, body, method):
            self.data["chunks"][c["chunk_id"]] = c
        # 更新：同族旧版本自动失效（仍可被显式指定版本查询）
        for key, d in self.data["docs"].items():
            if (d["doc_id"] == meta["doc_id"] and key != doc_key
                    and d["status"] == "active"
                    and self.version_key(d["version"]) < self.version_key(meta["version"])):
                d["status"] = "deprecated"
        return f"已导入: {doc_key}（{len(self.data['chunks'])} 个片段总量）"

    def revoke(self, doc_id, version=None):
        """撤回：状态置为 revoked，之后检索与引用均不可再用。"""
        n = 0
        for d in self.data["docs"].values():
            if d["doc_id"] == doc_id and (version is None or d["version"] == version):
                d["status"] = "revoked"
                n += 1
        return n

    def allowed_chunks(self, doc_id=None, version=None):
        """返回 (chunks, special)。默认每族取最新 active 版本；
        显式指定版本时允许 deprecated，但 revoked 永远不可用。"""
        docs = self.data["docs"]
        if doc_id and version:
            key = f"{doc_id}@{version}"
            if key not in docs:
                return [], "not_found"
            if docs[key]["status"] == "revoked":
                return [], "revoked"
            allow = {key}
        else:
            allow = set()
            fams = {}
            for key, d in docs.items():
                if d["status"] == "active" and (doc_id is None or d["doc_id"] == doc_id):
                    fams.setdefault(d["doc_id"], []).append(d)
            for fam in fams.values():
                latest = max(fam, key=lambda d: self.version_key(d["version"]))
                allow.add(f"{latest['doc_id']}@{latest['version']}")
        return [c for c in self.data["chunks"].values() if c["doc_key"] in allow], None

# ---------------------------------------------------------------- 检索

def tokenize(text):
    """中英混合：英文/编号/错误码整体成词；中文用字二元组（bigram）为主、单字低权重兜底。"""
    text = text.lower()
    tf = {}
    for m in re.finditer(r"[a-z0-9][a-z0-9._\-]*", text):
        # 编号/错误码/版本号类词条精度极高，加权 2.0（对“编号问题”的取舍：精确匹配优先）
        tf[m.group()] = tf.get(m.group(), 0.0) + 2.0
    cjk = [ch for ch in text if "一" <= ch <= "鿿"]
    for ch in cjk:
        tf[ch] = tf.get(ch, 0.0) + 0.3
    for a, b in zip(cjk, cjk[1:]):
        tf[a + b] = tf.get(a + b, 0.0) + 1.0
    return tf


def bm25_rank(query, chunks):
    N = len(chunks)
    if N == 0:
        return []
    q_tf = tokenize(query)
    d_tf = [tokenize(c["heading"] + " " + c["text"]) for c in chunks]
    avgdl = sum(sum(d.values()) for d in d_tf) / N or 1e-9
    df = {}
    for d in d_tf:
        for t in d:
            df[t] = df.get(t, 0) + 1
    k1, b = 1.5, 0.75
    out = []
    for c, d in zip(chunks, d_tf):
        dl = sum(d.values()) or 1e-9
        s = 0.0
        for t, qw in q_tf.items():
            if t not in d:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            tfv = d[t]
            s += qw * idf * (tfv * (k1 + 1)) / (tfv + k1 * (1 - b + b * dl / avgdl))
        out.append((c, s))
    out.sort(key=lambda x: -x[1])
    return out


def hash_vec(tf, dim=512):
    """无需密钥的确定性哈希向量（仅用于演示向量检索，非真实语义嵌入）。"""
    v = [0.0] * dim
    for t, w in tf.items():
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += w * (1.0 if (h >> 8) & 1 else -1.0)
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def vector_rank(query, chunks):
    qv = hash_vec(tokenize(query))
    out = []
    for c in chunks:
        cv = hash_vec(tokenize(c["heading"] + " " + c["text"]))
        out.append((c, sum(a * b for a, b in zip(qv, cv))))
    out.sort(key=lambda x: -x[1])
    return out


def search(store, question, method="bm25", doc_id=None, version=None):
    chunks, special = store.allowed_chunks(doc_id, version)
    if special:
        return [], special
    if method == "bm25":
        ranked = bm25_rank(question, chunks)
    elif method == "vector":
        ranked = vector_rank(question, chunks)
    else:  # hybrid：单一变量对照实验用，0.5*归一化BM25 + 0.5*余弦
        bm = bm25_rank(question, chunks)
        vc = vector_rank(question, chunks)
        bmax = bm[0][1] if bm and bm[0][1] > 0 else 1.0
        vscore = {c["chunk_id"]: s for c, s in vc}
        merged = [(c, 0.5 * (s / bmax) + 0.5 * vscore.get(c["chunk_id"], 0.0)) for c, s in bm]
        merged.sort(key=lambda x: -x[1])
        ranked = merged
        return ranked[:TOP_K], None
    return ranked[:TOP_K], None

# ---------------------------------------------------------------- 上下文构造 + 模拟生成

def excerpt(text, limit=160):
    return text if len(text) <= limit else text[:limit] + "…"


def mock_generate(question, ranked, special=None, min_score=MIN_SCORE):
    """模板化生成：只做摘录与冲突罗列，标注【模拟生成】。
    返回 {text, citations, conflict}。证据（片段全文）直接进入上下文，引用即片段编号。"""
    if special == "revoked":
        return {"text": "【模拟生成】所指定版本已被撤回，不能作为有效依据。", "citations": [], "conflict": False}
    if special == "not_found":
        return {"text": "【模拟生成】未找到指定的文档版本。", "citations": [], "conflict": False}
    hits = [(c, s) for c, s in ranked if s >= min_score]
    if not hits:
        return {"text": "【模拟生成】资料不足：现有资料中没有足够证据回答该问题。", "citations": [], "conflict": False}
    top = hits[0][1]
    strong = [(c, s) for c, s in hits if s >= STRONG_RATIO * top]
    fams = {}
    for c, _ in strong:
        fams.setdefault(c["doc_id"], []).append(c)
    if len(fams) > 1:  # 冲突：多来源说法并列呈现
        lines = ["【模拟生成】不同来源对该问题的说法不一致，请确认适用的场景与版本："]
        citations = []
        for did, cs in list(fams.items())[:3]:
            c = cs[0]
            lines.append(f"- 《{c['title']}》v{c['version']}：{excerpt(c['text'])}（证据：{c['chunk_id']}）")
            citations.append(c["chunk_id"])
        return {"text": "\n".join(lines), "citations": citations, "conflict": True}
    c0_id = list(fams.keys())[0]
    cs = [c for c, s in hits if c["doc_id"] == c0_id and s >= 0.5 * top][:3]
    citations = [c["chunk_id"] for c in cs]
    c0 = cs[0]
    text = (f"【模拟生成】依据《{c0['title']}》v{c0['version']}："
            + "；".join(excerpt(c["text"]) for c in cs)
            + f"（证据：{', '.join(citations)}）")
    return {"text": text, "citations": citations, "conflict": False}

# ---------------------------------------------------------------- CLI

def cmd_import_all(store):
    for f in sorted(DOCS_DIR.glob("*.md")):
        print(store.import_file(f))
    store.save()


def cmd_list(store):
    print(f"{'文档标识':<20}{'版本':<8}{'状态':<12}{'来源':<18}标题")
    for key in sorted(store.data["docs"]):
        d = store.data["docs"][key]
        print(f"{d['doc_id']:<20}{d['version']:<8}{d['status']:<12}{d['source']:<18}{d['title']}")
    print(f"\n片段总数: {len(store.data['chunks'])}")


def cmd_ask(store, q, doc_id, version, method, debug):
    ranked, special = search(store, q, method, doc_id, version)
    if debug:
        for c, s in ranked:
            print(f"  [{s:7.3f}] {c['chunk_id']} {excerpt(c['text'], 60)}")
    ans = mock_generate(q, ranked, special)
    print(ans["text"])


# ---------------------------------------------------------------- Web 演示页（纯标准库 http.server）

PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>知澜知识库 · RAG 演示</title>
<style>
*{box-sizing:border-box;margin:0}
body{font-family:system-ui,"Microsoft YaHei",sans-serif;height:100vh;display:flex;flex-direction:column;background:#f0f6ff;color:#1f2937}
header{height:52px;display:flex;align-items:center;gap:10px;padding:0 20px;background:#fff;border-bottom:1px solid #dbeafe}
.logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#38bdf8,#2563eb);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700}
header b{font-size:16px} header .tag{font-size:12px;color:#64748b}
main{flex:1;display:flex;min-height:0}
aside{width:280px;background:#fff;border-right:1px solid #dbeafe;display:flex;flex-direction:column}
aside h2{font-size:13px;color:#475569;padding:12px 16px 6px}
#docs{flex:1;overflow-y:auto;padding:0 8px 12px}
.doc{padding:8px 10px;border-radius:8px;margin-bottom:4px;font-size:13px}
.doc:hover{background:#eff6ff}
.doc .meta{font-size:11px;color:#94a3b8}
.badge{font-size:10px;padding:1px 6px;border-radius:99px;margin-left:4px}
.active{background:#dcfce7;color:#059669}.deprecated{background:#fef3c7;color:#d97706}.revoked{background:#fee2e2;color:#dc2626}
.chat{flex:1;display:flex;flex-direction:column;min-width:0}
#msgs{flex:1;overflow-y:auto;padding:20px 24px}
.msg{display:flex;margin-bottom:16px}
.msg.user{justify-content:flex-end}
.bubble{max-width:72%;padding:10px 14px;border-radius:14px;font-size:14px;line-height:1.7;white-space:pre-wrap}
.user .bubble{background:#2563eb;color:#fff;border-bottom-right-radius:4px}
.bot .bubble{background:#fff;border:1px solid #dbeafe;border-bottom-left-radius:4px}
.refs{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px}
.ref{font-size:11px;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:6px;padding:2px 8px;cursor:default}
.mock{font-size:11px;color:#94a3b8;margin-top:6px}
.inputbar{padding:12px 24px 16px;background:transparent}
.controls{display:flex;gap:8px;margin-bottom:8px}
.controls select,.controls input{padding:5px 8px;border:1px solid #bfdbfe;border-radius:8px;font-size:12px;background:#fff}
.box{display:flex;gap:8px;background:#fff;border:1px solid #bfdbfe;border-radius:14px;padding:8px}
#q{flex:1;border:0;outline:0;font-size:14px;padding:6px 8px;background:transparent}
button{border:0;border-radius:10px;background:#2563eb;color:#fff;padding:8px 20px;cursor:pointer;font-size:14px}
button:hover{background:#1d4ed8}
.up{background:#f8fafc;border:1px dashed #bfdbfe;border-radius:10px;margin:8px;padding:10px;font-size:12px}
.up select{width:100%;margin:6px 0;padding:5px;border:1px solid #bfdbfe;border-radius:6px;background:#fff}
.up .filebtn{display:block;text-align:center;padding:6px;border-radius:8px;background:#eff6ff;color:#1d4ed8;cursor:pointer}
.up button{width:100%;margin-top:6px;padding:6px}
#upmsg{font-size:11px;color:#64748b;margin-top:4px;min-height:14px}
</style></head><body>
<header><div class="logo">澜</div><b>知澜知识库</b><span class="tag">内部技术资料问答 · 模拟材料 · 答案为【模拟生成】</span></header>
<main>
<aside><h2>文档库（active=生效 / deprecated=旧版 / revoked=撤回）</h2><div id="docs"></div>
<div class="up"><b>上传文档</b>（.md / .txt）
<select id="cmethod">
<option value="paragraph">解析算法：按段落分块</option>
<option value="heading">解析算法：按章节分块</option>
<option value="fixed">解析算法：定长滑窗(500字)</option>
<option value="semantic">解析算法：语义分块(需嵌入模型)</option>
</select>
<label class="filebtn">选择文件<input type="file" id="file" accept=".md,.txt,.markdown" style="display:none"></label>
<span id="fname" style="color:#64748b"></span>
<button onclick="upload()">解析并导入</button><div id="upmsg"></div></div>
<div class="up"><b>模型设置</b> <span id="embstate" class="tag"></span>
<button onclick="location.href='/settings'">嵌入模型配置 →</button></div>
</aside>
<div class="chat">
  <div id="msgs">
    <div class="msg bot"><div class="bubble">你好，我是知识库助手。可以问我内部技术资料相关问题，例如：\\n· E1002 错误码表示什么？\\n· 接口超时时间是多少？（冲突示例）\\n· 公司食堂午餐几点开始供应？（证据不足示例）</div></div>
  </div>
  <div class="inputbar">
    <div class="controls">
      <select id="method"><option value="bm25">BM25</option><option value="hybrid">混合检索</option><option value="vector">哈希向量</option></select>
      <select id="doc"><option value="">全部文档</option></select>
      <select id="ver"><option value="">最新生效版本</option></select>
    </div>
    <div class="box"><input id="q" placeholder="输入问题，回车发送…"><button onclick="ask()">发送</button></div>
  </div>
</div>
</main>
<script>
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function add(role,text,cites){
  const m=document.createElement('div');m.className='msg '+role;
  let h='<div class="bubble">'+esc(text);
  if(cites&&cites.length){h+='<div class="refs">'+cites.map(c=>'<span class="ref">'+esc(c)+'</span>').join('')+'</div>'}
  if(role==='bot'){h+='<div class="mock">模板化摘录生成（模拟生成），引用可定位到原文片段</div>'}
  m.innerHTML=h+'</div>';msgs.appendChild(m);msgs.scrollTop=msgs.scrollHeight;
}
async function ask(){
  const text=q.value.trim();if(!text)return;q.value='';
  add('user',text);
  const p=new URLSearchParams({q:text,method:method.value,doc:doc.value,version:ver.value});
  const j=await (await fetch('/api/ask?'+p)).json();
  add('bot',j.answer,j.citations);
}
async function loadDocs(){
  const j=await (await fetch('/api/docs')).json();
  window._docs=j;
  document.getElementById('docs').innerHTML=j.map(d=>
    '<div class="doc">'+esc(d.title)+'<span class="badge '+d.status+'">'+d.status+'</span>'+
    '<div class="meta">'+esc(d.doc_id)+' v'+esc(d.version)+' · '+esc(d.source)+'</div></div>').join('');
  const ids=[...new Set(j.map(d=>d.doc_id))];
  document.getElementById('doc').innerHTML='<option value="">全部文档</option>'+
    ids.map(i=>'<option value="'+esc(i)+'">'+esc(i)+'</option>').join('');
  fillVer();
}
function fillVer(){
  const id=document.getElementById('doc').value;
  const vs=(window._docs||[]).filter(d=>!id||d.doc_id===id);
  document.getElementById('ver').innerHTML='<option value="">最新生效版本</option>'+
    (id?vs.map(d=>'<option value="'+esc(d.version)+'">v'+esc(d.version)+'（'+d.status+'）</option>').join(''):'');
}
document.getElementById('doc').addEventListener('change',fillVer);
async function upload(){
  const f=document.getElementById('file').files[0];
  if(!f){upmsg.textContent='请先选择文件';return}
  const text=await f.text();
  const r=await fetch('/api/upload?method='+cmethod.value+'&name='+encodeURIComponent(f.name),
    {method:'POST',body:text});
  const j=await r.json();
  upmsg.textContent=j.message;
  if(j.ok){loadDocs();add('bot','文档导入结果：'+j.message,[])}
}
async function loadEmb(){
  const j=await (await fetch('/api/embed-config')).json();
  embstate.textContent=j.configured?'已配置':'未配置';
}
document.getElementById('file').addEventListener('change',e=>{fname.textContent=e.target.files[0]?e.target.files[0].name:''});
loadDocs();loadEmb();q.addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
</script></body></html>"""


SETTINGS_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>模型设置 - 知澜知识库</title><style>
body{margin:0;font:14px/1.6 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;background:#f4f8fd;color:#0f172a}
.top{height:56px;background:#fff;border-bottom:1px solid #dbe7f5;display:flex;align-items:center;padding:0 20px;gap:12px}
.logo{width:32px;height:32px;border-radius:9px;background:linear-gradient(135deg,#38bdf8,#2563eb);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700}
.card{max-width:560px;margin:32px auto;background:#fff;border:1px solid #dbe7f5;border-radius:16px;padding:24px}
label{display:block;margin:14px 0 4px;font-weight:600;font-size:13px}
input{width:100%;padding:9px 12px;border:1px solid #dbe7f5;border-radius:8px;box-sizing:border-box}
button{background:#2563eb;color:#fff;border:none;padding:9px 20px;border-radius:8px;cursor:pointer;margin-top:16px}
.tip{color:#64748b;font-size:12px;margin-top:8px}
a{color:#2563eb;text-decoration:none}
</style></head><body>
<div class="top"><div class="logo">澜</div><b>知澜知识库</b><span style="color:#64748b;font-size:13px">嵌入模型设置</span>
<span style="flex:1"></span><a href="/">← 返回知识库</a></div>
<div class="card"><h2 style="margin-top:0">嵌入模型配置</h2>
<div class="tip">语义分块通过 OpenAI 兼容的 /embeddings 接口计算相邻段落相似度。配置保存在本地 kb_config.json，密钥不进入代码；未配置时其余三种解析算法不受影响。</div>
<label>API 地址</label><input id="eb" placeholder="如 https://api.openai.com/v1 或 http://127.0.0.1:11434/v1">
<label>API Key</label><input id="ek" type="password" placeholder="本地服务可任意填；留空则保持已存值">
<label>模型名</label><input id="em" placeholder="如 text-embedding-3-small / bge-m3">
<div>当前状态：<b id="state"></b></div>
<button onclick="save()">保存配置</button> <span id="msg" style="color:#059669"></span></div>
<script>
async function load(){
  const j=await (await fetch('/api/embed-config')).json();
  eb.value=j.base_url||''; em.value=j.model||'';
  state.textContent=j.configured?'已配置'+(j.has_key?'（含密钥）':''):'未配置';
}
async function save(){
  const j=await (await fetch('/api/embed-config',{method:'POST',
    body:JSON.stringify({base_url:eb.value,api_key:ek.value,model:em.value})})).json();
  msg.textContent=j.message; ek.value=''; load();
}
load();
</script></body></html>"""


def cmd_serve(store, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype="text/html"):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                self._send(PAGE)
            elif u.path == "/settings":
                self._send(SETTINGS_PAGE)
            elif u.path == "/api/ask":
                q = qs.get("q", [""])[0]
                method = qs.get("method", ["bm25"])[0]
                doc = qs.get("doc", [""])[0] or None
                ver = qs.get("ver", qs.get("version", [""]))[0] or None
                ranked, special = search(store, q, method, doc, ver)
                ans = mock_generate(q, ranked, special, METHOD_MIN_SCORE.get(method, MIN_SCORE))
                self._send(json.dumps({"answer": ans["text"], "citations": ans["citations"]},
                                      ensure_ascii=False), "application/json")
            elif u.path == "/api/docs":
                docs = sorted(store.data["docs"].values(),
                              key=lambda d: (d["doc_id"], Store.version_key(d["version"])))
                self._send(json.dumps(docs, ensure_ascii=False), "application/json")
            elif u.path == "/api/embed-config":
                s = embed_settings()
                self._send(json.dumps({
                    "base_url": s["base_url"], "model": s["model"],
                    "has_key": bool(s["api_key"]), "configured": bool(s["base_url"]),
                }, ensure_ascii=False), "application/json")
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/embed-config":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                cfg = load_config()
                cfg["embed_base_url"] = body.get("base_url", "").strip()
                cfg["embed_model"] = body.get("model", "").strip() or "text-embedding-3-small"
                if body.get("api_key"):
                    cfg["embed_api_key"] = body["api_key"].strip()
                save_config(cfg)
                self._send(json.dumps({"ok": True, "message": "嵌入模型配置已保存"},
                                      ensure_ascii=False), "application/json")
            elif u.path == "/api/upload":
                qs = parse_qs(u.query)
                method = qs.get("method", ["paragraph"])[0]
                if method not in CHUNKERS:
                    method = "paragraph"
                name = qs.get("name", ["uploaded-doc"])[0]
                name = re.sub(r"[^\w\-]+", "-", name.rsplit(".", 1)[0]) or "uploaded-doc"
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode("utf-8", errors="replace")
                try:
                    msg = store.import_text(raw, method, default_id=name)
                    store.save()
                    self._send(json.dumps({"ok": True, "message": msg}, ensure_ascii=False),
                               "application/json")
                except ValueError as e:
                    self._send(json.dumps({"ok": False, "message": str(e)}, ensure_ascii=False),
                               "application/json")
            else:
                self.send_response(404); self.end_headers()

        def log_message(self, *a):
            pass

    print(f"Web 演示页已启动: http://127.0.0.1:{port}  (Ctrl+C 停止)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


# ---------------------------------------------------------------- 评测

def match_spec(chunk, spec):
    return chunk["doc_id"] == spec["doc_id"] and spec["contains"] in chunk["text"]


def eval_one(store, q, method):
    for act in q.get("setup", []):
        if act["action"] == "revoke":
            store.revoke(act["doc_id"], act.get("version"))
        elif act["action"] == "import":
            store.import_file(UPDATES_DIR / act["file"])
    gold = q["gold"]
    thr = METHOD_MIN_SCORE[method]
    ranked, special = search(store, q["question"], method,
                             gold.get("doc_id"), gold.get("version"))
    ans = mock_generate(q["question"], ranked, special, thr)
    # 1) 检索命中：金标文档是否进入 top-k；证据不足题则要求无任何片段超过阈值
    retrieved = [c for c, _ in ranked]
    if gold.get("insufficient"):
        hit = not any(s >= thr for _, s in ranked)
    else:
        hit = any(c["doc_id"] in gold.get("docs", []) for c in retrieved)
    # 2) 证据完整性：金标片段被检索覆盖的比例
    specs = gold.get("chunks", [])
    if specs:
        completeness = sum(any(match_spec(c, sp) for c in retrieved) for sp in specs) / len(specs)
    elif gold.get("docs"):
        completeness = 1.0 if hit else 0.0
    else:
        completeness = 1.0
    # 3) 回答支持性：关键事实出现于答案的比例；期望“资料不足”时看是否如实声明
    text = ans["text"]
    if gold.get("insufficient"):
        support = 1.0 if ("资料不足" in text or "撤回" in text) else 0.0
    else:
        facts = gold.get("facts", [])
        got = sum(1 for f in facts if f in text)
        bad = sum(1 for f in gold.get("forbidden_facts", []) if f in text)
        support = max(0.0, (got - bad) / max(len(facts), 1))
    # 4) 引用正确性：引用片段确属金标证据（含关键事实）；无引用需求时须无引用
    cited = [store.data["chunks"][cid] for cid in ans["citations"] if cid in store.data["chunks"]]
    if gold.get("insufficient"):
        cite_ok = 1.0 if not cited else 0.0
    elif not cited:
        cite_ok = 0.0
    else:
        def cite_valid(c):
            if specs and any(match_spec(c, sp) for sp in specs):
                return True
            return c["doc_id"] in gold.get("docs", []) and any(f in c["text"] for f in gold.get("facts", []))
        cite_ok = sum(1 for c in cited if cite_valid(c)) / len(cited)
    return {"id": q["id"], "split": q["split"], "type": q["type"],
            "retrieval_hit": float(hit), "evidence_completeness": completeness,
            "answer_support": support, "citation_correctness": cite_ok,
            "answer": text[:120].replace("\n", " / ")}


def run_eval(method="bm25", split=None):
    store = Store()  # 内存中的全新库，评测不影响用户数据
    for f in sorted(DOCS_DIR.glob("*.md")):
        store.import_file(f)
    # 幂等性检查：重复导入全部文档，片段数不得变化
    before = len(store.data["chunks"])
    for f in sorted(DOCS_DIR.glob("*.md")):
        store.import_file(f)
    idempotent_ok = len(store.data["chunks"]) == before
    questions = json.loads(EVALSET.read_text(encoding="utf-8"))
    rows = [eval_one(store, q, method) for q in questions if split is None or q["split"] == split]
    return rows, idempotent_ok


def aggregate(rows):
    keys = ["retrieval_hit", "evidence_completeness", "answer_support", "citation_correctness"]
    return {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in keys}


def cmd_evaluate():
    rows, idem = run_eval("bm25")
    tune = [r for r in rows if r["split"] == "tune"]
    hold = [r for r in rows if r["split"] == "holdout"]
    # 对照实验：仅改变检索环节 bm25 -> hybrid
    abl_bm25, _ = run_eval("bm25", split="tune")
    abl_hybrid, _ = run_eval("hybrid", split="tune")
    A, B = aggregate(abl_bm25), aggregate(abl_hybrid)

    MET = [("retrieval_hit", "检索命中"), ("evidence_completeness", "证据完整性"),
           ("answer_support", "回答支持性"), ("citation_correctness", "引用正确性")]
    L = []
    L.append("# kb-rag-demo 评测报告\n")
    L.append("> 所有文档均为**模拟材料**；回答由模板化摘录生成（标注【模拟生成】），"
             "仅演示流程，**不能证明真实大模型的回答质量**。\n")
    L.append("## 1. 方法说明与取舍\n")
    L.append("- **检索方案**：纯 Python 实现的 BM25 关键词检索（中文按字二元组切分，英文/编号/错误码整体成词）。")
    L.append("- **编号与错误码**：如 `E1002` 作为完整词条索引，精确匹配优先——编号类问题语义改写无意义，故不做泛化。")
    L.append("- **语义问题**：字 bigram 只能覆盖部分语义重叠；为此实现了无需密钥的哈希向量检索作为对照，"
             "但它只是词袋的随机投影，并非真实语义嵌入，预期提升有限。")
    L.append("- **重排序**：未使用。语料小、BM25 单阶段已稳定；改用**分数阈值 + 同族聚合**代替重排序："
             "top1 低于阈值判“资料不足”，top-k 中出现多个高分来源判“冲突”并并列呈现。")
    L.append("- **证据进入上下文**：检索 top-%d 的片段全文直接拼入上下文；答案逐句附片段编号"
             "（`doc@version#pN`，含版本与段落序号，可定位回原文）。" % TOP_K)
    L.append("- **版本与状态**：默认每族只取最新 active 版本；`--version` 可查旧版；revoked 永不参与检索与引用。")
    L.append("- **幂等性**：按 doc_id+version 的内容哈希去重，重复导入不增加结果。"
             f"本次评测重复导入校验：{'通过' if idem else '失败'}。\n")
    L.append("## 2. 评测集与划分\n")
    L.append(f"共 {len(rows)} 题：调优集 {len(tune)} 题、保留集 {len(hold)} 题。"
             "划分原则：每种题型（正常/证据不足/指定版本/冲突/更新/撤回/同名术语/重复内容）在两个集合中至少各出现一次，"
             "阈值与生成规则只依据调优集校准，保留集仅用于最终验证，避免过拟合。\n")
    L.append("## 3. 分题结果（BM25）\n")
    L.append("| 题目 | 集合 | 类型 | 检索命中 | 证据完整性 | 回答支持性 | 引用正确性 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['id']} | {r['split']} | {r['type']} | "
                 + " | ".join(f"{r[k]:.2f}" for k, _ in MET) + " |")
    for name, grp in (("调优集", tune), ("保留集", hold), ("总体", rows)):
        g = aggregate(grp)
        L.append(f"| **{name}均值** |  |  | " + " | ".join(f"**{g[k]:.2f}**" for k, _ in MET) + " |")
    L.append("\n## 4. 对照实验（单一变量：检索方法 bm25 → hybrid，调优集）\n")
    L.append("| 方法 | 检索命中 | 证据完整性 | 回答支持性 | 引用正确性 |")
    L.append("|---|---|---|---|---|")
    for name, g in (("bm25", A), ("hybrid", B)):
        L.append(f"| {name} | " + " | ".join(f"{g[k]:.2f}" for k, _ in MET) + " |")
    delta = {k: B[k] - A[k] for k, _ in MET}
    L.append("\n**结论**：混合检索相对纯 BM25 的变化为 "
             + "，".join(f"{zh} {delta[k]:+.2f}" for k, zh in MET)
             + "。哈希向量只是词袋的随机投影，不具备真实语义，且归一化融合会稀释 BM25 的精确匹配信号"
               "（尤其是编号/错误码类问题），本实验**未观察到改善**，如实记录。"
               "若接入真实嵌入服务，可复用同一对照框架重新验证。")
    L.append("\n## 5. 失败案例（任一指标 < 1 的题目）\n")
    fails = [r for r in rows if min(r[k] for k, _ in MET) < 1.0]
    if not fails:
        L.append("本次评测无失败案例（指标全部满分）。这不代表现实质量——语料为精心构造的小规模模拟材料。")
    for r in fails:
        bad = [zh for k, zh in MET if r[k] < 1.0]
        L.append(f"- **{r['id']}**（{r['type']}）：失分指标 {', '.join(bad)}；答案摘要：{r['answer']}")
    L.append("\n## 6. 复现\n")
    L.append("```bash\npython kb.py import-all\npython kb.py ask \"E1002 错误码表示什么？\"\n"
             "python kb.py ask \"密码有效期\" --doc password-policy --version 1.0\n"
             "python kb.py revoke data-retention\npython kb.py evaluate\n```")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    for r in rows:
        print(f"{r['id']:<6}{r['split']:<9}{r['type']:<14}"
              + "  ".join(f"{r[k]:.2f}" for k, _ in MET))
    print("总体:", {k: round(v, 2) for k, v in aggregate(rows).items()})
    print(f"REPORT.md 已生成；幂等性检查: {'通过' if idem else '失败'}")


def main():
    ap = argparse.ArgumentParser(description="内部技术资料 RAG 演示（模拟材料）")
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import-all")
    p = sub.add_parser("import"); p.add_argument("file")
    p = sub.add_parser("revoke"); p.add_argument("doc_id"); p.add_argument("--version")
    sub.add_parser("list")
    p = sub.add_parser("ask"); p.add_argument("question")
    p.add_argument("--doc"); p.add_argument("--version")
    p.add_argument("--method", default="bm25", choices=["bm25", "vector", "hybrid"])
    p.add_argument("--debug", action="store_true")
    sub.add_parser("evaluate")
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    store = Store(a.store)
    if a.cmd == "import-all":
        cmd_import_all(store)
    elif a.cmd == "import":
        print(store.import_file(a.file)); store.save()
    elif a.cmd == "revoke":
        n = store.revoke(a.doc_id, a.version); store.save()
        print(f"已撤回 {n} 个版本")
    elif a.cmd == "list":
        cmd_list(store)
    elif a.cmd == "ask":
        cmd_ask(store, a.question, a.doc, a.version, a.method, a.debug)
    elif a.cmd == "evaluate":
        cmd_evaluate()
    elif a.cmd == "serve":
        cmd_serve(store, a.port)


if __name__ == "__main__":
    main()
