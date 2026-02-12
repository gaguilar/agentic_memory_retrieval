# Project Explanation

A conversational AI agent with **long-term memory** that persists across sessions. The system extracts, consolidates, and retrieves user memories to produce contextually aware responses. It supports three retrieval modes — from simple recency to semantic search and graph traversal.

---

## Major Components

### 1. CLI Entry Point (`src/main.py`)

Interactive chat loop built with [Rich](https://github.com/Textualize/rich). Handles user input, dispatches commands (`/quit`, `/new`, `/history`, `/memories`, `/graph`, `/help`), and renders responses with metadata (RAG queries, active memories, token usage). Accepts `--rag-mode` and `--username` CLI arguments.

### 2. Configuration (`src/config.py`)

Pydantic-based settings loaded from a `.env` file. Central knobs include:

| Setting | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | LLM backend (`openai` or `anthropic`) |
| `RAG_MODE` | `none` | Retrieval strategy: `none`, `semantic`, `graph` |
| `MAX_MEMORIES` | `10` | Cap on memories injected into the prompt |
| `MAX_PROMPT_TOKENS` | `4000` | Token budget for the full prompt |
| `MAX_RESPONSE_TOKENS` | `1000` | Tokens reserved for the LLM response |
| `MAX_CONVERSATION_TURNS` | `20` | Historical turns to consider |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.5` | Cosine threshold for semantic retrieval |
| `TAG_SIMILARITY_THRESHOLD` | `0.3` | Cosine threshold for tag-based consolidation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformer model |

### 3. Agent Layer (`src/agents/`)

Thin abstraction over LLM providers.

- **`BaseAgent`** — Abstract interface: `generate(messages, max_tokens, temperature) -> Response`
- **`OpenAIAgent`** — GPT wrapper (default model: `gpt-4o`)
- **`AnthropicAgent`** — Claude wrapper (default model: `claude-3-5-sonnet`)

Every LLM call in the system (conversation, extraction, consolidation, linking, query generation) goes through this unified interface.

### 4. Conversation Manager (`src/conversation/manager.py`)

The **orchestrator**. Ties together every other component:

- Manages user sessions and conversation lifecycle (`start_conversation`, `end_conversation`)
- Retrieves memories according to the active RAG mode
- Builds the XML-templated system prompt with `{memories}`, `{conversation_history}`, and `{current_conversation}` placeholders
- Enforces token limits — truncates only historical turns, never the current session
- Persists all user/assistant turns and inference logs
- Triggers the memory pipeline (extraction → consolidation → embedding → linking) when a conversation ends

### 5. Database Layer (`src/database/`)

SQLite-backed persistence via SQLAlchemy ORM.

**`models.py`** — Six tables:

| Table | Purpose |
|---|---|
| `User` | User identity (`id`, `username`, `created_at`) |
| `Conversation` | Session container (`user_id`, `started_at`, `ended_at`, `is_active`) |
| `Turn` | Individual message (`role`, `content`, `turn_number`, `conversation_id`) |
| `Memory` | Extracted fact (`statement`, `memory_type`, `tags` as JSON, `embedding` as BLOB, `is_active`, `superseded_by`) |
| `MemoryEdge` | Graph relationship (`source_memory_id`, `target_memory_id`, `label`, `is_active`) |
| `Inference` | LLM call log (`task`, `prompt`, `response`, `model`, `tokens`, `turn_id`) |

**`repository.py`** — CRUD operations with session management: user/conversation lifecycle, turn persistence, memory CRUD (including soft-delete via `is_active`), edge management, and inference logging.

### 6. Memory System (`src/memory/`)

A four-stage pipeline that runs when a conversation ends:

#### 6a. Extractor (`extractor.py`)

Uses an LLM (`EXTRACTION_PROMPT`) to analyze conversation turns and produce structured `ExtractedMemoryData` objects, each containing a `statement`, `type`, and `tags` list.

**Memory types:**
- `semantic` — Stable facts about the user
- `episodic` — Specific past events
- `procedural` — How the user typically works
- `affective` — Preferences, values, beliefs

#### 6b. Consolidator (`consolidator.py`)

Prevents duplication and keeps memories up-to-date. For each batch of candidates:

1. Pre-fetches all active memories for the user
2. Finds relevant existing memories via **tag matching** (exact overlap + cosine similarity on tag embeddings)
3. Pools and deduplicates them
4. Sends a **single** LLM call (`CONSOLIDATION_PROMPT`) that decides per candidate:
   - **SKIP** — Duplicate of an existing memory; do nothing
   - **UPDATE** — Contradicts an existing memory; supersede old, store new
   - **MERGE** — Complements an existing memory; enrich its statement/tags
   - **STORE** — Novel fact; insert as a new active memory
5. Executes the database operations (including deactivating graph edges for superseded/merged memories)

#### 6c. Embedder (`embedder.py`) + Tag Embedder (`tag_embedder.py`)

- **`TagEmbedder`** — Loads a `sentence-transformers` model and provides tag-level similarity search (used during consolidation to find memories with semantically similar tags)
- **`MemoryEmbedder`** — Wraps `TagEmbedder` to generate full-statement embeddings, serialize them to `numpy` BLOBs, and compute batch cosine similarities (used during semantic/graph retrieval)

#### 6d. Linker (`linker.py`)

*Only active in graph mode.* After consolidation:

1. Takes newly created/updated memories + all existing active memories
2. Uses an LLM (`MEMORY_LINKING_PROMPT`) to propose directed edges with labels like `HEALTH_LIFESTYLE`, `CULTURAL_ASPECT`, `LOCATION_CONSTRAINT`, etc.
3. Validates proposals (no self-loops, IDs must exist) and stores them as `MemoryEdge` records

### 7. RAG Retrievers (`src/rag/`)

Three implementations behind a shared `BaseRetriever` interface:

- **`NoOpRetriever`** (`noop.py`) — Returns empty results; used as a placeholder when `RAG_MODE=none` (the manager queries the DB directly instead)
- **`SemanticRetriever`** (`semantic.py`) — Embeds a query, loads all user memory embeddings, computes cosine similarity, filters by threshold, returns top-k
- **`GraphRetriever`** (`graph.py`) — Accepts multiple entry queries; for each one runs semantic search for seeds, then performs 1-hop graph traversal via `MemoryEdge`, merges/deduplicates, and scores (seeds=1.0, hops=0.7 decaying)

### 8. Prompt Templates (`src/prompts/`)

All LLM prompts live here as Python string constants:

| Module | Prompt | Used By |
|---|---|---|
| `system.py` | `SYSTEM_PROMPT` | Conversation Manager (main chat) |
| `memory_extraction.py` | `EXTRACTION_PROMPT` | Memory Extractor |
| `memory_consolidation.py` | `CONSOLIDATION_PROMPT` | Memory Consolidator |
| `memory_linking.py` | `MEMORY_LINKING_PROMPT` | Memory Linker |
| `semantic_query.py` | `SEMANTIC_QUERY_PROMPT` | Conversation Manager (semantic mode) |
| `graph_query.py` | `GRAPH_QUERY_PROMPT` | Conversation Manager (graph mode) |

### 9. Utilities (`src/utils/`)

- **`tokens.py`** — Token counting via `tiktoken` for accurate prompt budgeting
- **`print_artifacts.py`** — Rich-based display: conversation history, memories, graph visualization (ASCII art), RAG queries, user stats
- **`quiet.py`** — Log suppression for noisy libraries

---

## RAG Setup Diagrams

Each diagram shows what happens during a **single `send_message()` call** — how the user's message leads to memory retrieval, prompt construction, and a response.

### Setup 1: Vanilla (`RAG_MODE=none`)

No LLM query generation, no embeddings, no graph. Memories are retrieved by recency alone.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER MESSAGE                                        │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION MANAGER                                      │
│                                                                             │
│  1. Persist user turn to DB                                                 │
│  2. Retrieve memories ──────────────────────────────┐                       │
│  3. Get historical turns (past sessions, limited)   │                       │
│  4. Get current conversation turns (unlimited)      │                       │
│  5. Truncate history to fit token budget            │                       │
│  6. Build system prompt from XML template           │                       │
│  7. Call LLM agent                                  │                       │
│  8. Persist assistant turn + inference log           │                       │
└──────────────────┬──────────────────────────────────┘                       │
                   │                                                          │
                   │                                  ┌───────────────────────┘
                   │                                  │
                   │                                  ▼
                   │                  ┌──────────────────────────────────┐
                   │                  │         DATABASE (SQLite)        │
                   │                  │                                  │
                   │                  │  SELECT * FROM memory            │
                   │                  │  WHERE user_id = ?               │
                   │                  │    AND is_active = 1             │
                   │                  │  ORDER BY created_at DESC        │
                   │                  │  LIMIT {max_memories}            │
                   │                  │                                  │
                   │                  │  → Returns N most recent         │
                   │                  │    active memories               │
                   │                  └──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM PROMPT (XML Template)                          │
│                                                                             │
│  <memories>                                                                 │
│    - (semantic) User works as a data engineer ...                           │
│    - (episodic) User went to Japan last summer ...                          │
│    - (affective) User prefers Python over Java ...                          │
│  </memories>                                                                │
│                                                                             │
│  <conversation_history>                                                     │
│    [truncated past-session turns]                                           │
│  </conversation_history>                                                    │
│                                                                             │
│  <current_conversation>                                                     │
│    [all turns from this session]                                            │
│  </current_conversation>                                                    │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LLM AGENT (OpenAI / Anthropic)                        │
│                                                                             │
│  Generates response using the system prompt + message history               │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ASSISTANT RESPONSE                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key characteristic:** Memory retrieval is a single SQL query — no additional LLM calls, no embeddings, no vector math. Fast but naive; the most recent memories may not be the most relevant to the current topic.

---

### Setup 2: Semantic-Based Memory Retrieval (`RAG_MODE=semantic`)

Adds an **LLM query generation step** and **cosine similarity search** over memory embeddings.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER MESSAGE                                        │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION MANAGER                                      │
│                                                                             │
│  1. Persist user turn to DB                                                 │
│  2. Generate semantic query ─────────────────────┐                          │
│                                                  │                          │
│                                                  ▼                          │
│                              ┌──────────────────────────────────┐           │
│                              │  LLM CALL #1: Query Generation   │           │
│                              │                                  │           │
│                              │  Input:                          │           │
│                              │   - SEMANTIC_QUERY_PROMPT         │           │
│                              │   - Current conversation turns   │           │
│                              │   - User's latest message        │           │
│                              │                                  │           │
│                              │  Output:                         │           │
│                              │   "user dietary preferences      │           │
│                              │    cooking habits Japan travel"   │           │
│                              └──────────────┬───────────────────┘           │
│                                             │                               │
│  3. Retrieve memories via SemanticRetriever ◄┘                              │
│     │                                                                       │
│     ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                    SEMANTIC RETRIEVER                           │         │
│  │                                                                │         │
│  │  a. Load all active memories for user (with embeddings)        │         │
│  │  b. Back-fill embeddings for any memories missing them         │         │
│  │  c. Embed the generated query ──────┐                          │         │
│  │                                     │                          │         │
│  │                                     ▼                          │         │
│  │                  ┌──────────────────────────────────┐          │         │
│  │                  │  SENTENCE-TRANSFORMER MODEL       │          │         │
│  │                  │  (all-MiniLM-L6-v2)              │          │         │
│  │                  │                                  │          │         │
│  │                  │  query → 384-dim embedding        │          │         │
│  │                  └──────────────────────────────────┘          │         │
│  │                                                                │         │
│  │  d. Compute cosine similarity:                                 │         │
│  │     query_embedding · memory_embedding[i]                      │         │
│  │     ─────────────────────────────────────                      │         │
│  │     ║query║ × ║memory[i]║                                     │         │
│  │                                                                │         │
│  │  e. Filter: score >= similarity_threshold (0.5)                │         │
│  │  f. Sort by score DESC, cap at max_memories (10)               │         │
│  │                                                                │         │
│  │  Memory Pool          Similarity      Result                   │         │
│  │  ────────────         ──────────      ──────                   │         │
│  │  "likes sushi"          0.82          ✅ included              │         │
│  │  "works at Acme"        0.31          ❌ below threshold       │         │
│  │  "visited Tokyo"        0.76          ✅ included              │         │
│  │  "prefers Python"       0.22          ❌ below threshold       │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  4. Get historical turns + current turns                                    │
│  5. Truncate history to fit token budget                                    │
│  6. Build system prompt from XML template                                   │
│  7. Call LLM agent (conversation) ──────────────────┐                       │
│  8. Persist assistant turn + inference log           │                       │
└──────────────────────────────────────────────────────┘                       │
                                                                              │
                          ┌───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LLM CALL #2: Conversation                             │
│                                                                             │
│  System prompt now contains only the topically relevant memories            │
│  instead of the most recent ones                                            │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ASSISTANT RESPONSE                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key characteristic:** Two LLM calls per message (query generation + conversation). The retrieval is **content-aware** — it finds memories that are semantically related to the current topic, even if they were created long ago. The trade-off is latency from the extra LLM call and embedding computation.

---

### Setup 3: Graph-Based Memory Retrieval (`RAG_MODE=graph`)

Extends semantic with **multiple entry queries** and **1-hop graph traversal** to surface memories that are structurally related but may not match the query semantically.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER MESSAGE                                        │
│                  "What should I cook for dinner?"                            │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION MANAGER                                      │
│                                                                             │
│  1. Persist user turn to DB                                                 │
│  2. Generate graph queries ──────────────────────┐                          │
│                                                  │                          │
│                                                  ▼                          │
│                              ┌──────────────────────────────────┐           │
│                              │  LLM CALL #1: Query Generation   │           │
│                              │                                  │           │
│                              │  Input:                          │           │
│                              │   - GRAPH_QUERY_PROMPT            │           │
│                              │   - Current conversation turns   │           │
│                              │   - User's latest message        │           │
│                              │                                  │           │
│                              │  Output: 2-4 query objects       │           │
│                              │  ┌────────────────────────────┐  │           │
│                              │  │ Query 1:                   │  │           │
│                              │  │  query: "dietary prefs"    │  │           │
│                              │  │  seed_types: [affective]   │  │           │
│                              │  │  hop_types: [procedural]   │  │           │
│                              │  │                            │  │           │
│                              │  │ Query 2:                   │  │           │
│                              │  │  query: "cooking habits"   │  │           │
│                              │  │  seed_types: [procedural]  │  │           │
│                              │  │  hop_types: [semantic]     │  │           │
│                              │  │                            │  │           │
│                              │  │ Query 3:                   │  │           │
│                              │  │  query: "health goals"     │  │           │
│                              │  │  seed_types: [semantic]    │  │           │
│                              │  │  hop_types: [affective]    │  │           │
│                              │  └────────────────────────────┘  │           │
│                              └──────────────┬───────────────────┘           │
│                                             │                               │
│  3. Retrieve memories via GraphRetriever ◄──┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                     GRAPH RETRIEVER                            │         │
│  │                                                                │         │
│  │  For EACH query:                                               │         │
│  │                                                                │         │
│  │  ┌──────────────────────────────────────────────────────────┐  │         │
│  │  │ STEP A: Semantic Seed Retrieval                          │  │         │
│  │  │                                                          │  │         │
│  │  │ Uses SemanticRetriever internally:                       │  │         │
│  │  │  - Embed query with sentence-transformer                 │  │         │
│  │  │  - Cosine similarity against all memory embeddings       │  │         │
│  │  │  - Filter by seed_types if specified                     │  │         │
│  │  │  - Return top-k seeds (score: 1.0 → 0.95 → 0.90 ...)   │  │         │
│  │  └──────────────────────┬───────────────────────────────────┘  │         │
│  │                         │                                      │         │
│  │                         ▼                                      │         │
│  │  ┌──────────────────────────────────────────────────────────┐  │         │
│  │  │ STEP B: 1-Hop Graph Traversal                            │  │         │
│  │  │                                                          │  │         │
│  │  │ From each seed, follow MemoryEdge records:               │  │         │
│  │  │                                                          │  │         │
│  │  │ ┌──────────┐  HEALTH_LIFESTYLE  ┌──────────────────┐    │  │         │
│  │  │ │ "likes   │ ─────────────────► │ "runs 3x/week"   │    │  │         │
│  │  │ │  sushi"  │                    │ (procedural)      │    │  │         │
│  │  │ │ (seed)   │  CULTURAL_ASPECT   ├──────────────────┤    │  │         │
│  │  │ │          │ ─────────────────► │ "visited Tokyo"   │    │  │         │
│  │  │ └──────────┘                    │ (episodic)        │    │  │         │
│  │  │                                 └──────────────────┘    │  │         │
│  │  │                                                          │  │         │
│  │  │ - Prioritize hops matching hop_types                     │  │         │
│  │  │ - Score hops: 0.7 → 0.67 → 0.64 ...                     │  │         │
│  │  └──────────────────────────────────────────────────────────┘  │         │
│  │                                                                │         │
│  │  After all queries:                                            │         │
│  │  ┌──────────────────────────────────────────────────────────┐  │         │
│  │  │ STEP C: Merge & Deduplicate                              │  │         │
│  │  │                                                          │  │         │
│  │  │  - Union all seeds + hops across queries                 │  │         │
│  │  │  - Deduplicate by memory ID (highest score wins)         │  │         │
│  │  │  - Sort by score DESC                                    │  │         │
│  │  │  - Cap at max_memories (10)                              │  │         │
│  │  └──────────────────────────────────────────────────────────┘  │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  4. Get historical turns + current turns                                    │
│  5. Truncate history to fit token budget                                    │
│  6. Build system prompt from XML template                                   │
│  7. Call LLM agent (conversation) ──────────────────┐                       │
│  8. Persist assistant turn + inference log           │                       │
└──────────────────────────────────────────────────────┘                       │
                                                                              │
                          ┌───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LLM CALL #2: Conversation                             │
│                                                                             │
│  System prompt now contains seeds (directly relevant memories)              │
│  AND hops (structurally related memories the user didn't ask about          │
│  but that provide useful context — e.g. dietary restrictions                │
│  linked to health goals)                                                    │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ASSISTANT RESPONSE                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key characteristic:** Surfaces **structurally adjacent** memories that pure semantic search would miss. For example, if the user asks about dinner, graph traversal can pull in health constraints or cultural preferences connected to food memories via labeled edges — even if those memories don't contain the word "dinner." The cost is higher latency (extra LLM call for multi-query generation + graph DB lookups).

---

## Memory Pipeline (End of Conversation)

This pipeline runs when the user ends a session (`/quit`, `/new`, or exiting). It is identical across all three RAG modes, except that steps 3 and 4 are only active in semantic/graph and graph modes respectively.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION ENDS                                         │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: EXTRACTION                                                         │
│                                                                             │
│  MemoryExtractor + LLM (EXTRACTION_PROMPT)                                  │
│                                                                             │
│  Input:  All turns from the conversation (with timestamps)                  │
│  Output: List of ExtractedMemoryData:                                       │
│          { statement, type (semantic|episodic|procedural|affective), tags }  │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: CONSOLIDATION                                                      │
│                                                                             │
│  MemoryConsolidator + LLM (CONSOLIDATION_PROMPT)                            │
│                                                                             │
│  For each candidate:                                                        │
│   a. Find existing memories with matching/similar tags (via TagEmbedder)    │
│   b. Pool and deduplicate relevant existing memories                        │
│   c. Single LLM call decides per candidate:                                 │
│      ┌──────────────────────────────────────────────┐                       │
│      │  SKIP   → Duplicate       → do nothing       │                       │
│      │  UPDATE → Contradiction   → supersede old     │                       │
│      │  MERGE  → Complementary   → enrich existing   │                       │
│      │  STORE  → Novel           → insert new        │                       │
│      └──────────────────────────────────────────────┘                       │
│   d. Execute DB operations (including deactivating edges for UPDATE/MERGE)  │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: EMBEDDING GENERATION  [semantic + graph modes only]                │
│                                                                             │
│  MemoryEmbedder (sentence-transformers/all-MiniLM-L6-v2)                    │
│                                                                             │
│  For each new/updated memory:                                               │
│   - Generate 384-dimensional embedding from statement text                  │
│   - Serialize as numpy float32 bytes                                        │
│   - Store as BLOB in memory.embedding column                                │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: GRAPH LINKING  [graph mode only]                                   │
│                                                                             │
│  MemoryLinker + LLM (MEMORY_LINKING_PROMPT)                                 │
│                                                                             │
│  Input:  New memories + all existing active memories                        │
│  Output: Directed edges with labels:                                        │
│                                                                             │
│    Memory A ──[HEALTH_LIFESTYLE]──► Memory B                                │
│    Memory C ──[CULTURAL_ASPECT]──► Memory D                                 │
│    Memory E ──[LOCATION_CONSTRAINT]──► Memory F                             │
│                                                                             │
│  Edges stored as MemoryEdge records in the database                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison of the Three Setups

| Aspect | Vanilla | Semantic | Graph |
|---|---|---|---|
| LLM calls per message | **1** (conversation) | **2** (query gen + conversation) | **2** (multi-query gen + conversation) |
| Retrieval mechanism | SQL `ORDER BY created_at DESC LIMIT N` | Cosine similarity on embeddings | Cosine similarity + 1-hop traversal |
| Memory relevance | Recency-based (may miss old relevant facts) | Topic-based (finds old but relevant facts) | Topic + structural (finds indirectly related facts) |
| Embedding model | Not needed | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| Graph edges | Not created | Not created | Created by MemoryLinker |
| Extra LLM calls at end-of-conversation | 2 (extraction + consolidation) | 2 (extraction + consolidation) | 3 (extraction + consolidation + linking) |
| Latency per message | Lowest | Medium | Highest |
| Best for | Simple use cases, low latency | Topic-focused recall | Complex interconnected knowledge |
