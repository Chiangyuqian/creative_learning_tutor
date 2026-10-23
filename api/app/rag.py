"""RAG：切塊、嵌入與索引、檢索。

兩個版本，以環境變數 RAG_VERSION 切換：
  A（基本版）：固定長度切塊（字元）＋ 向量檢索
  B（改良版）：依 ### 段落切塊（附小節標題）＋ 混合檢索（BM25 + 向量，RRF 融合）
"""

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx


def _default_data_dir() -> Path:
    # 容器內教材掛在 /data；本機直接執行時用 repo 裡的 data/
    if Path("/data").is_dir():
        return Path("/data")
    return Path(__file__).resolve().parents[2] / "data"


DATA_DIR = Path(os.environ.get("DATA_DIR") or _default_data_dir())
TEXTBOOK_DIR = DATA_DIR / "textbook"

# 版本 A 的固定切塊大小（字元）
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
TOP_K = int(os.environ.get("TOP_K", "3"))
RAG_VERSION = os.environ.get("RAG_VERSION", "B").upper()
CACHE_DIR = Path(os.environ.get("CACHE_DIR") or Path(__file__).resolve().parents[1] / ".cache")


# ---------- 讀取教材 ----------

@dataclass
class Section:
    id: str      # 段落 id，例如 1-3-3
    title: str   # 段落標題，例如「例題：解 3x + 5 = 20」
    parent: str  # 所屬小節，例如「1-3 移項法則」
    text: str    # 段落內文（不含標題）


SECTION_RE = re.compile(r"^###\s+(\d+(?:-\d+)+)\s+(.*)$")
PARENT_RE = re.compile(r"^##\s+(.*)$")


def load_sections(directory: Path = TEXTBOOK_DIR) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(directory.glob("*.md")):
        parent = ""
        current: Section | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if m := SECTION_RE.match(line):
                current = Section(id=m.group(1), title=m.group(2).strip(), parent=parent, text="")
                sections.append(current)
            elif m := PARENT_RE.match(line):
                parent = m.group(1).strip()
                current = None  # 例如「本章範圍說明」沒有段落 id，不納入
            elif current is not None:
                current.text += line + "\n"
    for s in sections:
        s.text = s.text.strip()
    return sections


# ---------- 切塊 ----------

@dataclass
class Chunk:
    text: str                                              # 拿去嵌入／比對的文字
    section_ids: list[str] = field(default_factory=list)   # 這塊涵蓋的段落 id


def chunk_fixed(sections: list[Section], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """版本 A：全部段落接成一條長字串，每 size 字元切一塊，相鄰兩塊重疊 overlap 字元。"""
    full = ""
    owner: list[str] = []  # owner[i] = 第 i 個字元屬於哪個段落
    for s in sections:
        piece = f"{s.id} {s.title}\n{s.text}\n\n"
        full += piece
        owner += [s.id] * len(piece)

    chunks = []
    for start in range(0, len(full), size - overlap):
        end = min(start + size, len(full))
        ids = list(dict.fromkeys(owner[start:end]))  # 依出現順序去重
        chunks.append(Chunk(text=full[start:end], section_ids=ids))
        if end == len(full):
            break
    return chunks


def chunk_by_section(sections: list[Section]) -> list[Chunk]:
    """版本 B：一個 ### 段落一塊，前面加上所屬小節與段落標題。"""
    return [Chunk(text=f"{s.parent}｜{s.id} {s.title}\n{s.text}", section_ids=[s.id]) for s in sections]


# ---------- 嵌入 ----------

def embed(texts: list[str]) -> list[list[float]]:
    """呼叫地端 Ollama 嵌入模型，回傳正規化後的向量（長度 1）。"""
    resp = httpx.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    resp.raise_for_status()
    return [_normalize(v) for v in resp.json()["embeddings"]]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # 兩邊都已正規化，內積即 cosine


# ---------- BM25（版本 B 的關鍵字檢索） ----------

def tokenize(text: str) -> list[str]:
    """中文取單字＋相鄰兩字；英數字（含 x、數字）整段當一個詞。"""
    tokens = []
    for run in re.findall(r"[一-鿿]+|[A-Za-z0-9]+", text.lower()):
        if run[0].isascii():
            tokens.append(run)
        else:
            tokens += list(run)
            tokens += [run[i:i + 2] for i in range(len(run) - 1)]
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]          # 每塊裡每個詞出現幾次
        self.lens = [sum(tf.values()) for tf in self.tfs]        # 每塊的詞數
        self.avg = sum(self.lens) / len(self.lens)               # 平均詞數
        df = Counter(t for tf in self.tfs for t in tf)           # 每個詞出現在幾塊裡
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, length in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


# ---------- 索引與檢索 ----------

@dataclass
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, version: str):
        self.version = version
        self.sections = {s.id: s for s in load_sections()}
        secs = list(self.sections.values())
        self.chunks = chunk_fixed(secs) if version == "A" else chunk_by_section(secs)
        self.vectors = self._load_or_build_vectors()
        self.bm25 = BM25([c.text for c in self.chunks]) if version == "B" else None

    def _load_or_build_vectors(self) -> list[list[float]]:
        """切塊內容或嵌入模型沒變時，直接讀快取，不重算。"""
        texts = [c.text for c in self.chunks]
        key = hashlib.sha1(json.dumps([EMBED_MODEL, texts]).encode()).hexdigest()[:12]
        path = CACHE_DIR / f"index-{self.version}-{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        vectors = embed(texts)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vectors), encoding="utf-8")
        return vectors

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        qv = embed([query])[0]
        dense = [cosine(qv, v) for v in self.vectors]
        if self.bm25 is None:  # 版本 A：只用向量
            order = sorted(range(len(self.chunks)), key=lambda i: -dense[i])
            return [Hit(self.chunks[i], dense[i]) for i in order[:k]]

        # 版本 B：Reciprocal Rank Fusion，score = Σ 1 / (60 + 名次)
        sparse = self.bm25.scores(query)
        fused = [0.0] * len(self.chunks)
        for ranking in (dense, sparse):
            order = sorted(range(len(self.chunks)), key=lambda i: -ranking[i])
            for rank, i in enumerate(order, start=1):
                fused[i] += 1 / (60 + rank)
        order = sorted(range(len(self.chunks)), key=lambda i: -fused[i])
        return [Hit(self.chunks[i], fused[i]) for i in order[:k]]

    @staticmethod
    def hit_section_ids(hits: list[Hit]) -> list[str]:
        """命中切塊涵蓋的段落 id，依名次去重。"""
        return list(dict.fromkeys(sid for h in hits for sid in h.chunk.section_ids))


_indexes: dict[str, Index] = {}


def get_index(version: str = RAG_VERSION) -> Index:
    if version not in _indexes:
        _indexes[version] = Index(version)
    return _indexes[version]
