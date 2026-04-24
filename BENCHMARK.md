# Lab 17 — Benchmark Report: Multi-Memory Agent

> **ID: 2A202600050**
> **Name: Dương Trịnh Hoài An**

---

## 1. Summary Table

| # | Scenario | Category | No-Mem Score | With-Mem Score | Token Δ | Pass? |
|---|----------|----------|:------------:|:--------------:|:-------:|:-----:|
| 1 | Profile Write & Name Recall | `profile_recall` | 0/3 | 3/3 | +142 | ✅ |
| 2 | Language Preference Recall | `profile_recall` | 1/3 | 3/3 | +118 | ✅ |
| 3 | Conflict Update - Allergy (Required) | `conflict_update` | 0/3 | 3/3 | +95 | ✅ |
| 4 | Conflict Update - Skill Level | `conflict_update` | 0/3 | 3/3 | +103 | ✅ |
| 5 | Episodic Write - Debug Task | `episodic_write` | 1/3 | 2/3 | +87 | ✅ |
| 6 | Episodic Recall - Similar Debug | `episodic_recall` | 0/3 | 3/3 | +210 | ✅ |
| 7 | Semantic Retrieval - async/await | `semantic_retrieval` | 1/3 | 3/3 | +184 | ✅ |
| 8 | Semantic Retrieval - ML Overfitting | `semantic_retrieval` | 1/3 | 3/3 | +176 | ✅ |
| 9 | Token Budget Trim Test | `short_term_trim` | 1/3 | 2/3 | +45 | ✅ |
| 10 | Full Stack Integration | `full_stack` | 0/3 | 3/3 | +298 | ✅ |

**Overall:** 10/10 tests passed
**Avg score WITH memory:** 2.80/3
**Avg score WITHOUT memory:** 0.50/3
**Score improvement:** +2.30 points average

---

## 2. Detailed Results per Conversation

### Conv 1: Profile Write & Name Recall — ✅ PASS

- **Category:** `profile_recall`
- **Expected key information:** `Minh`

**Without Memory** (score: 0/3):
> Xin lỗi, tôi không có thông tin về tên của bạn trong cuộc trò chuyện này.
_No name stored — agent has no persistent memory across turns_

**With Memory** (score: 3/3):
> Có chứ! Bạn tên là **Minh**, đang học năm 3 đại học CNTT tại VinUniversity, và đang tự học Machine Learning cùng Python được khoảng 1 năm.
_Perfectly recalled name and details from long-term Redis memory_

Tokens — without: 312, with: 454

---

### Conv 2: Language Preference Recall — ✅ PASS

- **Category:** `profile_recall`
- **Expected key information:** `Python`

**Without Memory** (score: 1/3):
> Tôi có thể gợi ý Python hoặc JavaScript tùy theo mục đích dự án...
_Generic answer, not personalized_

**With Memory** (score: 3/3):
> Dựa trên preferences của bạn, tôi sẽ luôn suggest **Python** — đó là ngôn ngữ bạn prefer cho mọi dự án. Bạn cũng không thích Java và C++. Level: beginner nên tôi sẽ giải thích đơn giản.
_Uses stored language preference + level from long-term memory_

Tokens — without: 298, with: 416

---

### Conv 3: Conflict Update - Allergy (Required Rubric Test) — ✅ PASS

- **Category:** `conflict_update`
- **Expected key information:** `đậu nành`
- **⚠️ Conflict update test** (recency wins check)

**Without Memory** (score: 0/3):
> Trong cuộc trò chuyện này bạn đề cập đến cả sữa bò và đậu nành, tôi không chắc bạn dị ứng cái nào.
_Cannot resolve conflict — no memory system_

**With Memory** (score: 3/3):
> Bạn **dị ứng đậu nành**. Bạn đã đính chính trước đó rằng thông tin sữa bò là nhầm — profile đã được cập nhật: `allergy = đậu nành`. Vậy bạn không nên uống sữa đậu nành.
_Recency wins: Redis SET overwrote `allergy = sữa bò` → `allergy = đậu nành`_

Tokens — without: 287, with: 382

---

### Conv 4: Conflict Update - Skill Level — ✅ PASS

- **Category:** `conflict_update`
- **Expected key information:** `intermediate`
- **⚠️ Conflict update test** (recency wins check)

**Without Memory** (score: 0/3):
> Tôi không biết level hiện tại của bạn là gì.
_No memory of level update_

**With Memory** (score: 3/3):
> Level Python của bạn hiện tại là **intermediate** (2 năm kinh nghiệm). Profile đã được cập nhật từ beginner → intermediate. Về metaclass: [detailed explanation follows at intermediate level]
_Conflict resolved: intermediate overwrote beginner_

Tokens — without: 301, with: 404

---

### Conv 5: Episodic Write - Debug Task — ✅ PASS

- **Category:** `episodic_write`
- **Expected key information:** `conda install`

**Without Memory** (score: 1/3):
> Để fix ModuleNotFoundError, bạn thử pip install pandas.
_Generic advice, ignores conda context_

**With Memory** (score: 2/3):
> Vì bạn dùng Anaconda environment 'ml_project', dùng **`conda install pandas`** thay vì pip install. Episode đã được log để nhớ cho lần sau.
_Correctly identified conda context; episode logged to JSON_

Tokens — without: 276, with: 363

---

### Conv 6: Episodic Recall - Similar Debug Task — ✅ PASS

- **Category:** `episodic_recall`
- **Expected key information:** `conda install`

**Without Memory** (score: 0/3):
> Để fix ModuleNotFoundError numpy, bạn thử: `pip install numpy`
_Generic pip advice — wrong for conda env_

**With Memory** (score: 3/3):
> Tôi nhớ lần trước bạn gặp vấn đề tương tự với pandas trong conda environment! Cách đã work: **`conda install numpy`** (không phải pip vì bạn dùng Anaconda 'ml_project').
_Recalled past episode from episodic JSON log_

Tokens — without: 256, with: 466

---

### Conv 7: Semantic Retrieval - Python async/await — ✅ PASS

- **Category:** `semantic_retrieval`
- **Expected key information:** `asyncio`

**Without Memory** (score: 1/3):
> async/await là cú pháp để viết code bất đồng bộ...
_Generic explanation_

**With Memory** (score: 3/3):
> **async/await** (Python 3.5+): `async def` định nghĩa coroutine, `await` chờ mà không block event loop. `asyncio.run()` khởi chạy event loop. Dùng cho I/O-bound (HTTP, DB, file). `asyncio.gather()` chạy nhiều coroutines đồng thời. CPU-bound → dùng multiprocessing.
_Retrieved 2 doc chunks từ semantic memory (TF-IDF/ChromaDB)_

Tokens — without: 334, with: 518

---

### Conv 8: Semantic Retrieval - ML Overfitting — ✅ PASS

- **Category:** `semantic_retrieval`
- **Expected key information:** `overfitting`

**Without Memory** (score: 1/3):
> Overfitting là khi model fit quá tốt với training data...
_Basic answer_

**With Memory** (score: 3/3):
> **Overfitting**: train acc cao nhưng val acc thấp. Phòng tránh: **L1** (sparse weights, feature selection), **L2** (shrink đều), **Dropout** (disable p% neurons khi train), early stopping, cross-validation. L2 preferred khi không cần feature selection.
_Retrieved 3 chunks: overfitting + L1/L2 + dropout_

Tokens — without: 312, with: 488

---

### Conv 9: Token Budget Trim Test — ✅ PASS

- **Category:** `short_term_trim`
- **Expected key information:** `machine learning từ đầu`

**Without Memory** (score: 1/3):
> Xin lỗi, tôi không chắc bạn hỏi gì đầu tiên trong cuộc trò chuyện dài này.
_No trim strategy — context window overflow_

**With Memory** (score: 2/3):
> Câu hỏi đầu của bạn là muốn **học machine learning từ đầu**. Auto-trim đã kích hoạt sau turn 8 (budget < 200 tokens), giữ 5 messages gần nhất theo priority eviction.
_ContextManager trimmed semantic → episodic; short-term preserved key info_

Tokens — without: 445, with: 490

---

### Conv 10: Full Stack Integration — ✅ PASS

- **Category:** `full_stack`
- **Expected key information:** `An`

**Without Memory** (score: 0/3):
> Tôi không có đủ thông tin về bạn để đưa ra gợi ý cá nhân hóa.
_No profile, no experience, no knowledge_

**With Memory** (score: 3/3):
> **Profile (An):** NLP project tại AI startup, thích HuggingFace Transformers, đã fix CUDA OOM bằng batch_size=8 + mixed precision. **Bước tiếp theo:** (1) Fine-tune với LoRA để tiết kiệm VRAM, (2) Dùng `torch.cuda.amp.autocast()`, (3) Xem HuggingFace PEFT library.
_Kết hợp: profile (An, NLP, HF) + episodic (CUDA fix) + semantic (attention/transformer)_

Tokens — without: 289, with: 587

---

## 3. Memory Hit Rate Analysis

| Memory Type | Conversations Tested | Tests Passed | Hit Rate |
|-------------|:-------------------:|:------------:|:--------:|
| Long-term (Profile) | 2 | 2 | 100% |
| Long-term (Conflict) | 2 | 2 | 100% |
| Episodic (Write) | 1 | 1 | 100% |
| Episodic (Recall) | 1 | 1 | 100% |
| Semantic (TF-IDF/ChromaDB) | 2 | 2 | 100% |
| Short-term (Trim) | 1 | 1 | 100% |
| Full Stack (All 4) | 1 | 1 | 100% |
| **TOTAL** | **10** | **10** | **100%** |

---

## 4. Token Budget Breakdown

Token budget từ slide (% of context window, total = 4096):

| Memory Type | % Budget | Tokens (4096 ctx) |
|-------------|:--------:|:-----------------:|
| Short-term  | 10%      | ~410              |
| Long-term   | 4%       | ~164              |
| Episodic    | 3%       | ~123              |
| Semantic    | 3%       | ~123              |
| **Total memory cap** | **20%** | **~820** |

Total tokens — with memory (10 convs): **4,568**
Total tokens — no memory baseline:     **3,110**
Memory overhead: **+1,458 tokens (+47%)**
Relevance improvement: **+2.30 points (+460%)**
→ Memory rất token-efficient xét về chất lượng đạt được.

Token counting: **token_counter.py** — dùng tiktoken cl100k_base khi available, fallback `len(text)//3`
Priority eviction: semantic → episodic → long-term → short-term (last)

---

## 5. LangGraph Flow

```
START
  ↓
[load_memory]
  ├── router.classify(query) → memory types to load
  ├── short_term.get_recent(k=10)          [always]
  ├── long_term.get_profile(user_id)       [always]
  ├── episodic.search_similar(query, k=3)  [if episodic in types]
  └── semantic.query(query, k=3)           [if semantic in types]
  ↓
[context_manager.trim(state)]
  priority eviction: semantic → episodic → long-term → short-term
  ↓
[llm_node]
  ├── _build_system_prompt()
  │   ├── === USER PROFILE (Long-term Memory) ===
  │   ├── === PAST EXPERIENCES (Episodic Memory) ===
  │   ├── === RELEVANT KNOWLEDGE (Semantic Memory) ===
  │   └── === INSTRUCTIONS ===
  └── ChatOpenAI.invoke(messages)
  ↓
[save_memory]
  ├── short_term.add_message(query, response)
  ├── LLM extract facts → long_term.update_profile()  [recency wins]
  └── detect task completion → episodic.log_episode()
  ↓
END
```

---

## 6. Reflection — Privacy, Limitations & Lessons Learned

### 6.1 Memory type nào giúp agent nhất?

**Long-term memory (Redis)** impactful nhất cho user experience:
- Agent gọi đúng tên user mà không cần nhắc lại
- Language preference (Python-only) được áp dụng nhất quán
- Conflict update (allergy, skill level) hoạt động chính xác với recency wins

**Semantic memory (ChromaDB/TF-IDF)** impactful nhất cho answer quality:
- Chi tiết kỹ thuật (async/await, L1/L2, CUDA OOM) chính xác hơn hẳn
- Không có semantic memory → câu trả lời generic, thiếu depth

### 6.2 Memory type nào rủi ro nhất nếu retrieve sai?

**Long-term conflict handling** rủi ro cao nhất:
- Nếu recency wins KHÔNG được implement đúng → allergy sai → hại sức khỏe user
- **Mitigation:** Redis `SET` luôn overwrite cùng key → recency wins tự động

**Episodic memory hallucination** cũng nguy hiểm:
- Recall sai episode → suggest sai fix → user waste time
- **Mitigation:** Chỉ retrieve khi keyword overlap score > 0

### 6.3 PII / Privacy Risks

| Data stored | Risk level | Mitigation |
|-------------|:----------:|------------|
| Tên user (facts:name) | Medium | TTL 30d, delete_user() |
| Thông tin y tế (allergy) | **HIGH** | Explicit consent required, TTL 30d |
| Skill level, language | Low | TTL 90d |
| Session history | Medium | Auto-evict 7d |
| Episodic task logs | Medium | delete_user() purges all JSON entries |

**Privacy-by-Design principles đã implement:**
- ✅ Data minimization: chỉ extract facts liên quan đến tutoring
- ✅ Purpose limitation: memory chỉ dùng để personalize responses
- ✅ Storage limitation (TTL): prefs=90d, facts=30d, sessions=7d
- ✅ Consent management: user phải tự cung cấp thông tin
- ✅ Deletion verification: `reset_user()` xóa Redis + episodic JSON + short-term buffer

**GDPR Right to be Forgotten:**
```python
reset_user(user_id)
# → short_term.clear()
# → long_term.delete_user()   # xóa tất cả Redis keys
# → episodic.delete_user()    # xóa JSON entries
```
Trong multi-agent system: deletion phải propagate đến tất cả agents có copy (Federated Forgetting — slide).

### 6.4 Technical Limitations

1. **Episodic similarity dùng keyword overlap**, không phải embeddings
   - Risk: miss semantically similar nhưng lexically khác
   - Fix: dùng embeddings giống semantic memory

2. **LLM extraction accuracy** phụ thuộc vào output quality
   - Risk: hallucinated facts được lưu vào Redis
   - Fix: structured extraction + validation + human confirmation cho sensitive data

3. **fakeredis** không có network persistence — restart mất data
   - Fix: swap sang `redis.Redis()` (cùng API, 1 dòng thay đổi)

4. **ChromaDB model download** bị block trong một số môi trường
   - Fix: TF-IDF fallback đã implement, tự động kích hoạt

5. **Context trim mất thông tin irreversibly**
   - Fix: summarize trước khi trim (sliding window strategy từ slide)

6. **JSON episodic file** có race condition với concurrent users
   - Fix: dùng SQLite hoặc PostgreSQL cho production

### 6.5 Điều gì sẽ fail khi scale?

- **Single Redis instance** → bottleneck tại 10k+ users
- **Chroma in-memory** → mất semantic knowledge khi restart
- **JSON episodic file** → concurrent write conflicts
- **LLM extraction mỗi turn** → 2x API cost, 2x latency

---

## Appendix: Memory Router Classifications

| Category | Primary Intent | Memory Types Loaded |
|----------|---------------|---------------------|
| `profile_recall` | preference_query | short_term, long_term |
| `conflict_update` | preference_query | short_term, long_term |
| `episodic_recall` | experience_recall | short_term, episodic |
| `semantic_retrieval` | factual_recall | short_term, semantic |
| `short_term_trim` | general | short_term, long_term, episodic, semantic |
| `full_stack` | general | short_term, long_term, episodic, semantic |
