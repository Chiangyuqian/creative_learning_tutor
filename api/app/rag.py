"""RAG：切塊、嵌入與索引、檢索。

兩個版本，以環境變數 RAG_VERSION 切換：
  A（基本版）：固定長度切塊（字元）＋ 向量檢索
  B（改良版）：依 ### 段落切塊（附小節標題）＋ 混合檢索（BM25 + 向量，RRF 融合）
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


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
