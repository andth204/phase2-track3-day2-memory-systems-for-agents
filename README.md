# Lab 17 — Multi-Memory Agent với LangGraph

**Course:** AICB-P2T3 · VinUniversity · Phase 2 · Track 3 · Tuần 4  
**Topic:** Memory Systems for Agents (Slide Day 17)

---

## Mục tiêu

Build một **Multi-Memory Agent** với full memory stack gồm 4 backends:

| Memory Type | Backend | Role |
|-------------|---------|------|
| Short-term  | Sliding window buffer | Working memory, recent conversation |
| Long-term   | fakeredis (Redis API) | User profile, preferences, facts |
| Episodic    | JSON log file | Past task trajectories + reflections |
| Semantic    | ChromaDB (embeddings) | Domain knowledge retrieval |

---

## Cấu trúc project

```
lab17/
├── memory/
│   ├── __init__.py          # Exports all 4 memory types
│   ├── short_term.py        # Sliding window + tiktoken counting
│   ├── long_term.py         # Redis/fakeredis + TTL + GDPR delete
│   ├── episodic.py          # JSON episodic log + similarity search
│   └── semantic.py          # ChromaDB vector store
├── data/
│   ├── domain_docs.json     # 15 Python/ML knowledge chunks
│   ├── seed_episodes.json   # 3 seed episodes
│   └── benchmark_convs.json # 10 multi-turn conversation designs
├── agent.py                 # LangGraph StateGraph (3 nodes)
├── router.py                # Query intent classifier
├── context_manager.py       # Token budget + 4-level eviction
├── benchmark.py             # Run 10 convs, generate BENCHMARK.md
├── BENCHMARK.md             # Generated report (after running benchmark)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

```bash
# 1. Clone và tạo virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env và điền OPENAI_API_KEY

# 4. Run benchmark (generates BENCHMARK.md)
python benchmark.py
```

---

## LangGraph Architecture

```
START
  ↓
[load_memory]
  ├── router.classify(query) → memory types to load
  ├── short_term.get_recent(k=10)
  ├── long_term.get_profile(user_id)
  ├── episodic.search_similar(query, k=3)
  └── semantic.query(query, k=3)
  ↓
[context_manager.trim(state)]   ← 4-level priority eviction
  ↓
[llm_node]
  ├── _build_system_prompt()    ← inject 4 memory sections
  └── ChatOpenAI.invoke()
  ↓
[save_memory]
  ├── short_term.add_message()
  ├── LLM extract facts → long_term.update_profile()
  └── detect completion → episodic.log_episode()
  ↓
END
```

---

## MemoryState (TypedDict)

```python
class MemoryState(TypedDict):
    messages: List[Dict]       # short-term conversation buffer
    user_profile: Dict         # long-term: preferences + facts
    episodes: List[Dict]       # episodic: past task trajectories
    semantic_hits: List[str]   # semantic: knowledge chunks
    memory_budget: int         # remaining token budget
    user_id: str
    current_query: str
    llm_response: str
```

---

## Token Budget (from slide)

| Memory Type | Budget | Tokens (4096 ctx) |
|-------------|:------:|:-----------------:|
| Short-term  | 10%    | ~410 |
| Long-term   |  4%    | ~164 |
| Episodic    |  3%    | ~123 |
| Semantic    |  3%    | ~123 |
| **Total cap** | **20%** | **~820** |

Eviction order when near limit: semantic → episodic → long-term → short-term

---

## Conflict Handling (Recency Wins)

```
User: "Tôi dị ứng sữa bò."
→ Redis SET facts:{uid}:allergy = "sữa bò"

User: "À nhầm, tôi dị ứng đậu nành."
→ Redis SET facts:{uid}:allergy = "đậu nành"  ← overwrites

Profile: allergy = "đậu nành"  ✅
```

Redis `SET` always overwrites — same key = recency wins automatically.

---

## Benchmark Design

10 conversations covering all required test groups:

| # | Category | Tests |
|---|----------|-------|
| 1-2 | `profile_recall` | Name + language preference recall after 5+ turns |
| 3-4 | `conflict_update` | Allergy + skill level recency-wins test |
| 5 | `episodic_write` | Debug task logged as episode |
| 6 | `episodic_recall` | Similar task recalls past approach |
| 7-8 | `semantic_retrieval` | async/await + ML knowledge retrieval |
| 9 | `short_term_trim` | 12-turn conversation tests auto-trim |
| 10 | `full_stack` | All 4 memory types in 1 conversation |

---

## Privacy & GDPR

```python
# Right to be Forgotten
reset_user("user_id")
# Clears: short-term buffer + Redis keys + episodic log entries

# TTL policy
prefs:   90 days
facts:   30 days
sessions: 7 days
```

See `BENCHMARK.md` Section 6 for full privacy reflection.

---

## Bonus Features

- ✅ **tiktoken** for accurate token counting (not word count)
- ✅ **LLM-based fact extraction** with parse/error handling + fallback
- ✅ **Graph flow** via `agent.get_graph_description()`
- ✅ **GDPR delete** propagates across all backends
