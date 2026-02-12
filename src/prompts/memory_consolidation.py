"""
Prompt for batch memory consolidation decisions.

Instructs the LLM to compare ALL new memory candidates against a pooled set
of existing memories in a single inference pass and decide the appropriate
action for each candidate: SKIP, UPDATE, MERGE, or STORE.

Placeholders:
    {existing_memories}    — Formatted list of existing user memories (deduplicated pool)
    {candidate_memories}   — Formatted list of all new memory candidates
"""

# ========================================================================================
CONSOLIDATION_PROMPT = """
You are a memory consolidation system. You will receive a pool of EXISTING memories and a list of NEW CANDIDATE memories. For EACH candidate, decide the correct action by comparing it against the existing memories.

### ACTIONS

For each candidate, choose exactly ONE action:

1. **SKIP** — The candidate is redundant. An existing memory already captures this information fully.
   - Use when the candidate restates, paraphrases, or is a subset of an existing memory.

2. **UPDATE** — The candidate contradicts or replaces an existing memory.
   - Use when the candidate provides NEWER information that invalidates an older memory.
   - Use when the SAME event or fact is described but a key detail has CHANGED (e.g. "non-profit" → "for-profit", "lived in X" → "moved to Y").
   - Provide the memory_id of the existing memory to supersede.

3. **MERGE** — The candidate adds NEW detail to an existing memory without contradicting it.
   - Use when the candidate enriches or complements an existing memory.
   - Provide the memory_id to merge into AND a merged_statement that combines both pieces of information into one coherent memory.
   - Also provide merged_tags that combine tags from both the existing memory and the candidate.

4. **STORE** — The candidate is entirely novel. No existing memory covers this topic.
   - Use when no existing memory is semantically related to the candidate.

### GUIDELINES

- Evaluate each candidate independently against the pool of existing memories.
- Two candidates may reference the same existing memory (e.g. one MERGEs and another UPDATEs it). That is acceptable.
- If an existing memory is only loosely related but the candidate introduces a genuinely new facet, prefer STORE over MERGE.
- Be precise: only choose SKIP when the information is truly duplicated.

### OUTPUT FORMAT

Return a JSON array with one object per candidate, in the same order as the candidates listed above:

```json
[
  {{
    "candidate_index": <int>,
    "action": "skip" | "update" | "merge" | "store",
    "memory_id": <int or null>,
    "merged_statement": <string or null>,
    "merged_tags": <list of strings or null>,
    "reasoning": "<brief explanation>"
  }}
]
```

Rules:
- `candidate_index` corresponds to the candidate number (1-based).
- For SKIP: memory_id=null, merged_statement=null, merged_tags=null
- For UPDATE: memory_id=<id of memory to supersede>, merged_statement=null, merged_tags=null
- For MERGE: memory_id=<id of memory to enrich>, merged_statement=<combined statement>, merged_tags=<combined tags>
- For STORE: memory_id=null, merged_statement=null, merged_tags=null

## EVALUATION PHASE

EXISTING MEMORIES:
{existing_memories}

CANDIDATE MEMORIES:
{candidate_memories}

Return ONLY the JSON array, no additional text.

"""
