"""
Prompt for extracting structured memories from conversations.

Instructs the LLM to analyze conversation turns and produce
a JSON array of memory objects with temporal grounding and tags.

Placeholders:
  - conversation_text — formatted conversation text to analyze 
"""

# ===============================================================================================================
EXTRACTION_PROMPT = """
# Memory Extraction Prompt

You are a memory extraction system. Your task is to extract **durable, reusable user memories** from the conversation below.

Your objective is to identify a **small set of high-signal memory units** useful for long-term reasoning and personalization.
Do **NOT** maximize memory count.

## 1. SOURCE OF TRUTH (NON-NEGOTIABLE)

- Extract **ONLY** information explicitly stated by the **USER** in their own turns.
- Treat **ASSISTANT** turns as invisible for extraction.
- User agreement with assistant suggestions (e.g. "sounds good", "I like that") is **NOT** new information.
- Ignore assistant memory recalls (e.g. "I remember you said...").
- If you cannot point to a specific USER quote, **do not extract** the memory.

## 2. ATOMICITY RULE (MOST IMPORTANT)

Each memory must capture **exactly ONE atomic fact**.

**Atomic = one subject + one predicate**
- No conjunctions ("and", "also", commas joining facts).
- If it can be split, it **must** be split.

**Hard constraint:**  
- Maximum **15 words** per memory statement.

**Self-check before emitting a memory:**
1. Is there more than one independent fact? → SPLIT  
2. Is word count > 15? → SHORTEN or SPLIT  

### Atomicity examples

BAD:  
"User is a graphic designer who specializes in branding and lives in Austin."

GOOD:  
- "User is a graphic designer."  
- "User specializes in branding."  
- "User lives in Austin."

BAD:  
"User moved to New York in January 2026 for a startup job."

GOOD:  
- "User moved to New York in January 2026."  
- "User started a startup job in January 2026."

## 3. MEMORY GRANULARITY

- Prefer **one canonical memory** over near-duplicates.
- Do **not** create multiple memories that differ only by:
  - examples vs abstraction
  - tools vs instances
  - paraphrasing
- Split memories **only if** the aspects would be retrieved independently later.
- Before emitting, ask:
  **"Would removing this memory reduce future personalization?"**
  - If NO → discard.

**Soft cap:** Extract at most **5-7 memories per conversation** unless clearly necessary.

## 4. MEMORY TYPES

Use exactly one type per memory:

- **semantic** — stable facts about the user
- **episodic** — specific past events
- **procedural** — how the user generally works
- **affective** — preferences, values, beliefs

Notes:
- Prefer one memory per event or stable fact.
- Capture general strategies, not step-by-step procedures.
- Do not extract both liking and believing unless clearly distinct.

## 5. TEMPORAL GROUNDING (REQUIRED)

Resolve all relative time references using the conversation timestamp. 

Examples:
If the timestamp of the conversation is 10:00 AM on February 11, 2026:
- "last month" → "in January 2026"
- "last week" → "in the week of February 4, 2026"
- "yesterday" → "on February 10, 2026"
- "today" → "on February 11, 2026"
- or any other specific timeframe available that can be resolved to avoid ambiguity over time

If exact resolution is impossible, use the most specific timeframe available.

## 6. TAGGING RULES

For each memory, generate **2-6 tags**:
- lowercase
- underscore_separated
- include entities, topics, and time markers when relevant
- avoid over-specific versions (use "python", not "python_3_11")

## 7. OUTPUT FORMAT

Return a valid JSON array. Each object must contain:

- "statement" — atomic memory (less than 15 words)
- "type" — one of ["semantic","episodic","procedural","affective"]
- "tags" — list of 2-6 tags

If no high-signal memories are found, return:
[]

## CONVERSATION TO ANALYZE:

{conversation_text}
"""
