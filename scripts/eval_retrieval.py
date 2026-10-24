"""檢索評測：對 A、B 兩版跑同一份題目，比較 Top-K 命中段落與命中率。

只呼叫 rag.py 的檢索，不經過 LLM 生成。題目取自 jsonl 中帶有
expect.sources_include 的題目（以最後一輪訊息當查詢）。

用法（在容器內執行，需開著 Ollama；cmd 寫法）：
    docker compose run --rm -v "%cd%\\scripts:/srv/scripts:ro" -v "%cd%\\tests:/srv/tests:ro" api python scripts/eval_retrieval.py --detail

參數：
    --file tests/tutor.jsonl   換一份題目
    --k 5                      改 Top-K
    --detail                   列出未命中題目在向量／BM25 的名次
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 本機：app 在 api/app；容器：app 直接在 /srv/app
sys.path.insert(0, str(ROOT / "api" if (ROOT / "api").is_dir() else ROOT))

from app.rag import TOP_K, cosine, embed, get_index  # noqa: E402

VERSIONS = ("A", "B")


def load_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            want = case.get("expect", {}).get("sources_include")
            if want:
                cases.append({"id": case["id"], "tier": case.get("tier", "-"),
                              "query": case["turns"][-1], "want": want})
    return cases


def rank_of(scores: list[float], positions: list[int]) -> int:
    """positions 這幾塊裡，排最前面的是第幾名（1 起算）。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return min(order.index(p) + 1 for p in positions)


def detail(idx, query: str, want: list[str]) -> str:
    """應命中段落所在的切塊，在向量與 BM25 中各排第幾名。"""
    positions = [i for i, c in enumerate(idx.chunks) if set(c.section_ids) & set(want)]
    qv = embed([query])[0]
    out = f"向量第 {rank_of([cosine(qv, v) for v in idx.vectors], positions)} 名"
    if idx.bm25 is not None:
        out += f"、BM25 第 {rank_of(idx.bm25.scores(query), positions)} 名"
    return f"{out}（共 {len(idx.chunks)} 塊）"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # 輸出導到檔案時，中文和符號才不會出錯

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "tests" / "my_tests.jsonl"))
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--detail", action="store_true", help="列出未命中題目在向量／BM25 的名次")
    args = ap.parse_args()

    cases = load_cases(args.file)
    indexes = {v: get_index(v) for v in VERSIONS}
    stats = {v: {"hit": 0, "hit1": 0, "hit3": 0, "ids": 0} for v in VERSIONS}
    by_tier = defaultdict(lambda: {v: 0 for v in VERSIONS} | {"n": 0})
    misses = []

    print(f"| # | 題號 | 問題 | 應命中 | A 實際命中（Top-{args.k}） | A 命中？ | B 實際命中（Top-{args.k}） | B 命中？ |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for n, case in enumerate(cases, 1):
        row = [str(n), case["id"], case["query"], ", ".join(case["want"])]
        by_tier[case["tier"]]["n"] += 1
        for v in VERSIONS:
            idx = indexes[v]
            ids = idx.hit_section_ids(idx.search(case["query"], k=args.k))
            rank = next((i for i, sid in enumerate(ids, 1) if sid in case["want"]), None)  # 第一個正確段落排第幾
            ok = rank is not None
            s = stats[v]
            s["hit"] += ok
            s["hit1"] += rank == 1
            s["hit3"] += ok and rank <= 3
            s["ids"] += len(ids)
            by_tier[case["tier"]][v] += ok
            row += [", ".join(ids), f"✅ {rank}" if ok else "❌"]
            if not ok:
                misses.append((v, case))
        print("| " + " | ".join(row) + " |")

    print("\n> 命中？欄：✅ 後的數字為第一個正確段落在回傳段落中的名次。")

    n = len(cases)
    pct = lambda x: f"{x}/{n}（{x / n:.0%}）"
    print("\n| | 版本 A | 版本 B |")
    print("| --- | --- | --- |")
    for tier, t in by_tier.items():
        print(f"| {tier} | " + " | ".join(f"{t[v]}/{t['n']}" for v in VERSIONS) + " |")
    print("| **命中率**（Top-K 切塊內任一段命中） | " + " | ".join(f"**{pct(stats[v]['hit'])}**" for v in VERSIONS) + " |")
    print("| 平均回傳段落數 | " + " | ".join(f"{stats[v]['ids'] / n:.1f}" for v in VERSIONS) + " |")
    print("| 首位命中率（第 1 段即命中） | " + " | ".join(pct(stats[v]["hit1"]) for v in VERSIONS) + " |")
    print("| 前 3 段命中率（同段數比較） | " + " | ".join(pct(stats[v]["hit3"]) for v in VERSIONS) + " |")

    if args.detail and misses:
        print("\n未命中題目的名次分析")
        for v, case in misses:
            print(f"  [{v}] {case['id']} {case['query']}")
            print(f"      應命中 {', '.join(case['want'])}：{detail(indexes[v], case['query'], case['want'])}")


if __name__ == "__main__":
    main()
