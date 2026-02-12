"""
Prompt for generating edges between memories in the memory graph.

Instructs the LLM to analyze new and existing memories and propose
meaningful edges that capture situational, causal, and contextual
relationships. Edges should connect memories that are relevant to
each other beyond simple topic overlap.

Placeholders:
    {new_memories}      — Formatted list of newly extracted memories
    {existing_memories}  — Formatted list of existing active memories
"""

# ========================================================================================
MEMORY_LINKING_PROMPT = """
You are a memory graph linking system. Your task is to identify meaningful relationships between memories and propose edges for a memory graph.

## PURPOSE

The memory graph enables 1-hop retrieval: when a user's query matches one memory via semantic search, the graph edges allow retrieval of related memories that semantic search alone would miss. Edges should connect memories that are **situationally, causally, or contextually related** — not merely topically similar.

## EDGE GUIDELINES

Good edges connect memories that:
- Have a causal or explanatory relationship (e.g., health condition → dietary restriction)
- Share situational context (e.g., location → activity preferences affected by that location)
- Represent different facets of the same life aspect (e.g., diet + exercise = health lifestyle)
- Would help an assistant give better, more contextualized responses when one memory is recalled

Bad edges (avoid):
- Pure topic overlap that adds no new retrieval value (e.g., two facts about the same topic that would both be retrieved by the same semantic query anyway)
- Trivially obvious connections
- Edges between every pair — be selective. Only create edges where the connection adds real retrieval value.

## EDGE LABELS

Use descriptive UPPER_SNAKE_CASE labels that capture the nature of the relationship:
- HEALTH_LIFESTYLE (diet ↔ exercise)
- CULTURAL_ASPECT (language/culture ↔ location)
- LOCATION_CONSTRAINT (location → activity affected by it)
- FAMILY_HEALTH_CONSTRAINT (family health condition → dietary needs)
- TRAVEL_CONTEXT (travel habits ↔ location changes)
- PROFESSIONAL_CONTEXT (job/skills ↔ work habits)
- DIETARY_CONTEXT (diet ↔ food needs)
- EMOTIONAL_TRIGGER (experience → emotional response)

You may create new labels as appropriate — these are examples, not an exhaustive list.

## INSTRUCTIONS

1. For each new memory, evaluate potential connections to ALL other memories (both new and existing).
2. Also consider edges between new memories themselves if a meaningful relationship exists.
3. Only propose edges where the relationship provides genuine retrieval value.
4. Each edge should be directed (source → target), but the relationship is considered bidirectional for traversal.

## OUTPUT FORMAT

Return a JSON array of edge objects. Each object must have:
- "source_id": integer ID of the source memory
- "target_id": integer ID of the target memory
- "label": UPPER_SNAKE_CASE relationship label
- "reasoning": brief explanation of why this edge is valuable

If no meaningful edges exist, return an empty array: []

Example:
```json
[
  {{"source_id": 12, "target_id": 5, "label": "HEALTH_LIFESTYLE", "reasoning": "Diet preference connects to exercise habits for holistic health context"}},
  {{"source_id": 12, "target_id": 8, "label": "FAMILY_HEALTH_CONSTRAINT", "reasoning": "Family health condition explains dietary restrictions"}}
]
```

## INPUT

### NEW MEMORIES (just extracted from the latest conversation)
{new_memories}

### EXISTING MEMORIES (previously stored, currently active)
{existing_memories}

Return ONLY the JSON array. No additional text.
"""
