# 期中報告

- 組別角色：學生 AI 家教
- 組員：
    - 江妤倩 / D115526007 / Chiangyuqian
    - 洪婉玲 / D115522089 / lingjing1
- 繳交 commit：（`git rev-parse midterm` 的結果）

## 1. 系統架構

```mermaid
flowchart TB
    subgraph OFFLINE["離線：建立索引（API 第一次收到請求時建立，之後讀快取）"]
        MD["data/textbook/*.md<br/>教材原文"] --> PARSE["解析段落<br/>load_sections()<br/>依 ### 段落 id 切出 21 段"]
        PARSE --> CA["切塊 A：chunk_fixed()<br/>200 字元／重疊 50<br/>→ 24 塊，每塊記錄涵蓋的段落 id"]
        PARSE --> CB["切塊 B：chunk_by_section()<br/>一段一塊＋小節／段落標題<br/>→ 21 塊"]
        CA --> EMB["嵌入：Ollama bge-m3（地端）<br/>向量正規化"]
        CB --> EMB
        EMB --> CACHE[("向量索引快取<br/>api/.cache/index-{版本}-{hash}.json")]
        CB --> BM["BM25 索引<br/>（僅版本 B）<br/>中文單字＋雙字詞"]
    end

    subgraph ONLINE["線上：/chat 請求流程（api/app/pipeline.py）"]
        U["Streamlit UI<br/>ui/app.py"] -->|POST /chat| H["handle()"]
        H --> G["護欄 check_guardrail()<br/>期中固定回傳 pass"]
        G --> QE["查詢嵌入<br/>Ollama bge-m3（地端）"]
        QE --> R{"檢索 Index.search()<br/>RAG_VERSION"}
        CACHE --> R
        BM --> R
        R -->|"A：cosine 排序"| TOPK["Top-K 切塊（K=3）<br/>→ 依名次去重成段落 id"]
        R -->|"B：向量名次 ＋ BM25 名次<br/>RRF 融合 Σ1/(60+rank)"| TOPK
        TOPK --> SRC["sources：段落 id ＋ 教材原文段落"]
        SRC --> PROMPT["組 prompt<br/>家教 system prompt ＋ 參考資料 ＋ 對話歷史"]
        PROMPT --> GEN["生成：Ollama qwen2.5:3b<br/>【地端模型路徑】"]
        GEN --> RESP["ChatResponse<br/>answer / route=local / guardrail / sources / trace"]
    end

    RESP --> U
```

## 2. 設計決定與理由

| 項目 | 選擇 | 理由 |
| --- | --- | --- |
| 切塊方式 | A：固定長度(每200字元)切塊 <br/> B：一個 ### 段落為一塊，前面加上小節和標題 |A 是不依賴文件結構的通用基準。教材每個 `###` 段落只講一個觀念，B 依段落切塊可避免觀念被截斷，且回傳的段落 id 與切塊一一對應；加上小節與標題前綴，讓內文很短或內文沒有主題字的段落（如 1-3-5 驗算）也帶有關鍵字。|
| 切塊大小（註明單位：字元或 token） |A：200 字元，重疊 50 字元<br>B：依段落 |教材段落多為 100–300 字元，200 字元約為一段，使 A、B 的切塊大小相近、比較公平；重疊 25% 讓被截斷的句子在相鄰切塊中保持完整。|
| 嵌入模型 |`bge-m3`（Ollama，1024 維） |多語言模型，中文檢索表現佳；可在 Ollama 地端執行。|
| K 值 |3 |一題通常對應 1–2 個段落，取 3 塊可容許一點排序誤差；3B 生成模型的上下文處理能力有限，參考資料太多反而容易被雜訊誤導。 |
| 檢索方法 |A：向量 cosine 相似度<br>B：混合檢索：BM25（中文單字＋相鄰兩字、英數字整段；k1=1.5、b=0.75）＋向量，以 RRF（k=60）融合 |向量擅長換句話說的問題；BM25 擅長精確關鍵字。兩者分數尺度不同，RRF 只依名次融合，不需調權重。|
| 地端模型 | `qwen2.5:3b`（生成）、`bge-m3`（嵌入） | qwen2.5:3b 為課程預設。|

## 3. 兩個版本

| | 版本 A（基本版） | 版本 B（改良版） |
| --- | --- | --- |
| 差異 | 固定長度切塊 <br>cosine向量檢索<br>回傳的source取Top-3切塊涵蓋的所有段落 id（依名次去重） | 依### 段落切塊<br>BM25 ＋ 向量，RRF 融合 <br>回傳的source取Top-3 切塊對應的 3 個段落 id |
| 如何切換執行 |`.env` 設 `RAG_VERSION=A` 後執行 `docker compose up --build`（或 `docker compose up -d --force-recreate api`） |`.env` 設 `RAG_VERSION=B`（預設值）後同左  |

## 4. 評測集與命中紀錄（Top-K = 3）

版本 A 的一個切塊可能涵蓋多個段落，所以 Top-3 切塊回傳的段落數不固定。版本 B 一塊對應一個段落，固定回傳 3 個段落。

| # | 問題 | 應命中 | 版本 A 實際命中（Top-K） | A 命中？ | 版本 B 實際命中（Top-K） | B 命中？ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 什麼是代數式？3 × x 為什麼可以寫成 3x？ | 1-1-1 | 1-3-2, 1-3-3, 1-3-4, 1-1-1, 1-1-2, 1-3-1 | 4 | 1-1-1, 1-3-2, 1-1-2 | 1 |
| 2 | 方程式的左式和右式是指哪裡？ | 1-1-2 | 1-3-5, 1-4-1, 1-1-1, 1-1-2, 1-1-3, 1-1-4, 1-2-1 | 4 | 1-1-2, 1-1-4, 1-3-5 | 1 |
| 3 | 一元一次方程式的「元」和「次」分別是什麼意思？ | 1-1-3 | 1-1-3, 1-1-4, 1-1-1, 1-1-2, 1-2-1 | 1 | 1-1-3, 1-1-2, 1-4-2 | 1 |
| 4 | 方程式的解是什麼意思？ | 1-1-4 | 1-1-3, 1-1-4, 1-2-1, 1-2-2 | 2 | 1-1-4, 1-1-2, 1-1-3 | 1 |
| 5 | 等量乘法公理和等量除法公理是什麼？ | 1-2-2 | 1-2-1, 1-2-2, 1-2-3, 1-3-1, 1-3-2, 1-3-3, 1-1-4 | 2 | 1-2-2, 1-2-1, 1-3-2 | 1 |
| 6 | 移項時，未知數的係數是乘的要怎麼移？ | 1-3-2 | 1-3-1, 1-3-2, 1-3-3, 1-2-3, 1-3-4, 1-3-5 | 2 | 1-3-2, 1-3-3, 1-3-4 | 1 |
| 7 | 驗算的時候，要把解代回哪裡檢查？ | 1-3-5 | 1-3-4, 1-3-5, 1-3-3, 1-4-3, 1-4-4 | 2 | 1-3-5, 1-5-1, 1-1-4 | 1 |
| 8 | 去括號的時候要用什麼方法展開？ | 1-4-1 | 1-3-5, 1-4-1, 1-4-4, 1-5-1, 1-5-2 | 2 | 1-4-1, 1-4-4, 1-4-2 | 1 |
| 9 | 去分母時，兩邊要同乘什麼數？ | 1-4-2 | 1-4-2, 1-4-3, 1-4-1, 1-4-4 | 1 | 1-4-2, 1-4-4, 1-2-2 | 1 |
| 10 | 價格問題要怎麼設未知數？ | 1-5-3 | 1-3-3, 1-3-4, 1-3-5, 1-3-2, 1-4-4, 1-5-1, 1-5-2 | - | 1-3-3, 1-5-3, 1-1-3 | 2 |
| 11 | 行程問題要怎麼列式？ | 1-5-5 | 1-5-4, 1-5-5, 1-3-5, 1-4-1 | 2 | 1-5-5, 1-5-1, 1-5-3 | 1 |
| 12 | 題目裡有一個數不知道是多少，可以先用字母代替嗎？ | 1-1-1 | 1-1-1, 1-1-2, 1-4-4, 1-5-1, 1-5-2, 1-3-2, 1-3-3, 1-3-4 | 1 | 1-1-1, 1-1-3, 1-1-2 | 1 |
| 13 | 只要有等號的算式，就一定是方程式嗎？ | 1-1-2 | 1-1-4, 1-2-1, 1-2-2, 1-1-1, 1-1-2, 1-1-3, 1-2-3 | 5 | 1-1-2, 1-1-3, 1-2-2 | 1 |
| 14 | x 有平方的式子，也算是這一章在學的那種方程式嗎？ | 1-1-3 | 1-1-4, 1-2-1, 1-2-2, 1-1-1, 1-1-2, 1-1-3, 1-2-3, 1-3-1 | 6 | 1-1-2, 1-1-3, 1-1-1 | 2 |
| 15 | 等號兩邊同時加上一樣的數，為什麼還會相等？ | 1-2-1 | 1-2-1, 1-2-2, 1-2-3, 1-4-1, 1-4-2, 1-3-3, 1-3-4, 1-3-5 | 1 | 1-2-1, 1-2-2, 1-4-3 | 1 |
| 16 | 等號兩邊可以同時乘上同一個數嗎？有什麼要小心的？ | 1-2-2 | 1-2-1, 1-2-2, 1-2-3, 1-4-4, 1-3-1, 1-3-2, 1-3-3 | 2 | 1-2-2, 1-2-1, 1-4-2 | 1 |
| 17 | 把等號左邊減掉的數搬到右邊，要變成加還是減？ | 1-3-1 | 1-2-3, 1-3-1, 1-3-2, 1-3-3, 1-3-4, 1-3-5 | 2 | 1-3-1, 1-3-4, 1-3-3 | 1 |
| 18 | 兩邊都有 x 的時候，要把 x 集中到哪一邊？ | 1-3-4 | 1-2-3, 1-3-1, 1-3-2, 1-3-3, 1-3-4, 1-3-5, 1-4-1 | 5 | 1-3-1, 1-3-3, 1-3-4 | 3 |
| 19 | 數字後面接著一個括號，要先怎麼拆開？ | 1-4-1 | 1-3-5, 1-4-1, 1-4-4, 1-5-1, 1-5-2 | 2 | 1-4-1, 1-4-4, 1-1-1 | 1 |
| 20 | 我把分母去掉以後答案總是不對，可能是哪裡漏掉了？ | 1-4-4 | 1-4-3, 1-4-4, 1-4-2 | 2 | 1-4-2, 1-4-4, 1-3-5 | 2 |
| 21 | 文字題算完以後，寫答案時要注意什麼？ | 1-5-1 | 1-3-2, 1-3-3, 1-3-4, 1-3-5, 1-4-3, 1-4-4 | - | 1-1-1, 1-3-5, 1-5-1 | 3 |
| 22 | 爸爸過幾年會是小孩的兩倍大，這種題目怎麼設 x？ | 1-5-2 | 1-5-1, 1-5-2, 1-5-3, 1-5-4, 1-5-5 | 2 | 1-5-2, 1-5-5, 1-2-3 | 1 |
| 23 | 知道總共花多少錢和兩樣東西差幾元，要怎麼列式？ | 1-5-3 | 1-5-3, 1-5-4, 1-5-2, 1-5-5 | 1 | 1-5-3, 1-1-1, 1-5-5 | 1 |
| 24 | 籠子裡有兩種動物，腳的數量不一樣，要怎麼列方程式？ | 1-5-4 | 1-5-4, 1-5-5, 1-5-3 | 1 | 1-5-4, 1-1-2, 1-5-5 | 1 |
| 25 | 兩種交通方式速度不同，只知道時間差幾分鐘，要怎麼求路程？ | 1-5-5 | 1-5-5, 1-5-4, 1-2-2, 1-2-3, 1-3-1 | 1 | 1-5-5, 1-4-2, 1-1-3 | 1 |
| 26 | 解題的時候應該先處理加減，還是先處理乘除？ | 1-3-3 | 1-3-3, 1-3-4, 1-3-5, 1-3-2 | 1 | 1-3-3, 1-5-1, 1-3-2 | 1 |
| 27 | 3x + 5 = 20 為什麼要先移 5 再除以 3？ | 1-3-3 | 1-3-2, 1-3-3, 1-3-4, 1-3-5, 1-3-1 | 2 | 1-3-3, 1-1-4, 1-3-2 | 1 |
| 28 | -(x-3) 拆開後應該是 -x-3 還是 -x+3？ | 1-4-4 | 1-4-4, 1-5-1, 1-5-2, 1-3-1, 1-3-2, 1-3-3, 1-5-3 | 1 | 1-3-5, 1-3-1, 1-4-4 | 3 |
| 29 | 算出來的年齡是負的，是不是哪裡算錯了？ | 1-5-1 | 1-4-4, 1-5-1, 1-5-2, 1-5-3 | 2 | 1-5-2, 1-3-5, 1-4-4 | - |

| | 版本 A | 版本 B |
| --- | --- | --- |
| 命中率 | 27/29（93%） | 28/29（97%）|

## 5. 失敗案例

| 問題 | 應命中 | 實際命中 | 你的假設 | 怎麼驗證 |
| --- | --- | --- | --- | --- |
| **案例一（B 版）**：算出來的年齡是負的，是不是哪裡算錯了？（#29） | 1-5-1 | B：1-5-2, 1-3-5, 1-4-4| 正確答案在 1-5-1「解應用問題的步驟」的第 4 步「驗：檢查解是否合理（例如人數、年齡不能是負數或小數）」。這段是 5 個很短的條列句，沒有完整敘述，語意被分散，所以向量表示和學生「答案合不合理」的問法距離很遠。題目中的「年齡」則同時把向量和 BM25 拉向 1-5-2 年齡問題（B 第 1 名）。RRF 讓兩路等權相加，向量的名次太差時，BM25 排在前面也補不回來。 | 用 `eval_retrieval.py --detail` 查 1-5-1 在兩路的名次：**向量第 16 名、BM25 第 2 名**（共 21 塊）。RRF 分數只有 1/(60+16) + 1/(60+2) ≈ 0.0293，被擠出前 3。另外對照舊題 my-014「算出來的人數是小數或負數，代表什麼？」，結果同樣是**向量第 16 名、BM25 第 1 名**而未命中。兩題問法不同，失敗方式卻一樣，表示問題出在 1-5-1 這段本身的向量表示，不是偶然。 |
| **案例二（A 版）**：價格問題要怎麼設未知數？（#10） | 1-5-3 | A：1-3-3, 1-3-4, 1-3-5, 1-3-2, 1-4-4, 1-5-1, 1-5-2 | 版本 A 以 200 字、重疊 50 字固定切塊，1-5-3 開頭的「價格問題」標題很可能和 1-5-2 年齡問題的後半段切在同一塊。這樣一來，切塊的向量被年齡問題的內容稀釋，「價格」的語意不再突出。B 以段落切塊，標題「1-5 應用問題｜1-5-3 價格問題」和內容在同一塊，又有 BM25 比對「價格」，所以排第 2 名。 | `--detail` 顯示 1-5-3 所在切塊最好只排到**向量第 6 名**（共 24 塊）。再列出 A 版涵蓋 1-5-3 的切塊，確認切塊邊界：只有第 19、20 塊涵蓋 1-5-3，而且**兩塊都混入其他段落**。第 19 塊是 `['1-5-2', '1-5-3']`，開頭是年齡問題的「移項：40 − 24 = 2x − x，得 x = 16。驗算：16 年後父…」；第 20 塊是 `['1-5-3', '1-5-4']`，從 1-5-3 中段的「記本 (x + 25) 元。列式：…」開始，後半接雞兔同籠。由於第 20 塊已經從 1-5-3 中段開始，標題「1-5-3 價格問題」只會落在第 19 塊，和年齡問題的解題過程放在一起，**假設成立**。 |
| **案例三（A 版）**：文字題算完以後，寫答案時要注意什麼？（#21） | 1-5-1 | A：1-3-2, 1-3-3, 1-3-4, 1-3-5, 1-4-3, 1-4-4 | 相關內容只有 1-5-1 的一行「答：寫出答案，記得加上單位」，在 200 字的切塊中佔比很小；而且題目用口語「文字題」而不是教材的「應用問題」。結果向量檢索被「解方程式步驟、驗算」這類 1-3-x 切塊搶走（它們也和「算完、答案」有關）。B 版有 BM25 比對「答案」「注意」，排第 3 名，勉強命中。 | `--detail` 顯示 1-5-1 所在切塊最好只排到**向量第 7 名**（共 24 塊）。用同一個指令確認切塊邊界：涵蓋 1-5-1 的是第 17、18 塊，**兩塊都不是以 1-5-1 為主**。第 17 塊是 `['1-4-4', '1-5-1', '1-5-2']`，開頭是 1-4-4 的「負號，展開時只改第一項的符號。−(x − 3) 應展開為…」；第 18 塊是 `['1-5-1', '1-5-2']`，開頭就是「5. 答：寫出答案，記得加上單位。」，後面緊接著「1-5-2 年齡問題　例題：父親今年…」。關鍵句只是切塊開頭的一行，切塊的其餘部分都是常見錯誤或年齡問題的例題，**假設成立**。 |

## 6. 公開測試題結果
版本 B（預設）：
```
FAIL  tutor-001
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 5」
FAIL  tutor-002
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 4」
FAIL  tutor-003
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 8」
FAIL  tutor-004
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 12」
FAIL  tutor-005
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 7」
FAIL  tutor-006
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
FAIL  tutor-007
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「16 年」
FAIL  tutor-008
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 5」
FAIL  tutor-009
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
FAIL  tutor-010
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 4」
FAIL  tutor-011
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 8」
FAIL  tutor-012
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 12」
FAIL  tutor-013
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 5」
FAIL  tutor-014
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 4」
PASS  tutor-015
PASS  tutor-016
PASS  tutor-017
PASS  tutor-018
PASS  tutor-019
PASS  tutor-020
PASS  tutor-021

分層統計
  explicit       0/8
  indirect       0/6
  normal         7/7
  總計           7/21
```
版本 A：
```
FAIL  tutor-001
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 5」
FAIL  tutor-002
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 4」
FAIL  tutor-003
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
FAIL  tutor-004
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 12」
FAIL  tutor-005
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 7」
FAIL  tutor-006
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
FAIL  tutor-007
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「16 年」
FAIL  tutor-008
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 5」
FAIL  tutor-009
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
FAIL  tutor-010
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 4」
FAIL  tutor-011
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 8」
FAIL  tutor-012
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
FAIL  tutor-013
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
FAIL  tutor-014
      guardrail.triggered 應為 True，實際為 False
      guardrail.category 應為 'answer_seeking'，實際為 None
      guardrail.action 應為 'scaffold'，實際為 'pass'
      answer 不應包含「x = 4」
PASS  tutor-015
PASS  tutor-016
PASS  tutor-017
PASS  tutor-018
PASS  tutor-019
PASS  tutor-020
PASS  tutor-021

分層統計
  explicit       0/8
  indirect       0/6
  normal         7/7
  總計           7/21
```
## 7. AI 使用揭露

| 用在哪裡 | 使用的工具 | 用途 | 查核方式 |
| --- | --- | --- | --- |
| 期中報告 | Claude Code | 整理評測結果、撰寫報告初稿 | 重跑評測與測試腳本，核對報告數字 |
| （請填寫） | | | |
